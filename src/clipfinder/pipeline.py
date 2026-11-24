"""Pipeline: candidates → score → retention → dedupe → campaign → top-K.

Blueprint §1.4: reject clips with time overlap above 30% or transcript
similarity above 0.85; return the top 10 per hour of source. Every clip
carries its score breakdown, retention prediction, hook-lab variants,
reframe plan, captions and campaign check — the full explainable ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .campaign import CampaignBrief
from .candidates import Candidate, build_candidates
from .captions import build_captions, captions_to_srt
from .features import FeatureTrack, neutral_track
from .hooklab import HookVariant, hook_lab
from .reframe import CropPlan, plan_reframe
from .retention import RetentionModel
from .scoring import Score, score_candidate
from .transcript import Transcript

TOP_K_PER_HOUR = 10
MAX_OVERLAP = 0.30
MAX_SIMILARITY = 0.85
_WORD = re.compile(r"[a-z0-9']+")


@dataclass
class Clip:
    candidate: Candidate
    score: Score
    retention_proba: float
    hook_variants: list[HookVariant]
    reframe: CropPlan
    campaign_violations: list[str]
    title: str

    @property
    def start(self) -> float:
        return self.candidate.start

    @property
    def end(self) -> float:
        return self.candidate.end

    @property
    def duration(self) -> float:
        return self.candidate.duration

    @property
    def text(self) -> str:
        return self.candidate.text

    @property
    def campaign_ok(self) -> bool:
        return not self.campaign_violations

    def captions_srt(self) -> str:
        return captions_to_srt(build_captions(list(self.candidate.words)))

    def sidecar(self) -> dict:
        return {
            "title": self.title,
            "start": round(self.start, 2), "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "score": round(self.score.total, 2),
            "score_parts": {k: round(v, 2) for k, v in self.score.parts.items()},
            "payoff": round(self.score.payoff, 2),
            "retention_proba": round(self.retention_proba, 3),
            "reason": self.score.reason,
            "evidence": self.score.evidence,
            "risks": self.score.risks,
            "campaign_violations": self.campaign_violations,
            "hook_variants": [{"name": v.name, "opening": v.opening,
                               "hook_score": round(v.hook_score, 2),
                               "delta": round(v.delta_vs_original, 2)}
                              for v in self.hook_variants],
            "hook_text": self.candidate.first_sentence,
            "transcript": self.text,
            "output": "1080x1920, 30fps, H.264, AAC 192kbps",
        }


def _overlap(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = min(a.duration, b.duration)
    return inter / shorter if shorter else 0.0


def _similarity(a: Candidate, b: Candidate) -> float:
    ta = set(_WORD.findall(a.text.lower()))
    tb = set(_WORD.findall(b.text.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedupe(scored: list[tuple[Candidate, Score]]) -> list[tuple[Candidate, Score]]:
    """Keep the best clip, drop anything overlapping >30% or similar >0.85."""
    ranked = sorted(scored, key=lambda cs: cs[1].total, reverse=True)
    kept: list[tuple[Candidate, Score]] = []
    for cand, score in ranked:
        if any(_overlap(cand, k[0]) > MAX_OVERLAP or
               _similarity(cand, k[0]) > MAX_SIMILARITY for k in kept):
            continue
        kept.append((cand, score))
    return kept


def suggest_title(cand: Candidate) -> str:
    first = cand.first_sentence.strip().rstrip(".!")
    title = first[:60]
    return title[0].upper() + title[1:] if title else "Untitled clip"


def find_clips(transcript: Transcript, features: FeatureTrack | None = None,
               retention_model: RetentionModel | None = None,
               brief: CampaignBrief | None = None,
               top_k_per_hour: int = TOP_K_PER_HOUR) -> list[Clip]:
    features = features or neutral_track(transcript.duration)
    campaign_kw = tuple(w.lower() for w in (brief.required_hashtags if brief else ()))
    scored = [(c, score_candidate(c, features, campaign_kw))
              for c in build_candidates(transcript)]
    kept = dedupe(scored)
    hours = max(1.0, transcript.duration / 3600)
    top_k = max(1, int(top_k_per_hour * hours))
    clips: list[Clip] = []
    for cand, score in sorted(kept, key=lambda cs: cs[1].total, reverse=True)[:top_k]:
        if retention_model is not None:
            feats = [[score.parts["hook"] / 10, score.parts["emotion"] / 10,
                      score.parts["insight"] / 10,
                      score.parts["audio_energy"] / 10, cand.length_fit]]
            retention = float(retention_model.predict_proba(feats)[0])
        else:
            retention = 0.5
        title = suggest_title(cand)
        violations = brief.check(cand.text, cand.duration, title) if brief else []
        clips.append(Clip(
            candidate=cand, score=score, retention_proba=retention,
            hook_variants=hook_lab(cand),
            reframe=plan_reframe(features.face_x, cand.start, cand.end,
                                 features.energy),
            campaign_violations=violations, title=title))
    # Final order: score first, retention breaks ties — both are shown.
    clips.sort(key=lambda c: (c.score.total, c.retention_proba), reverse=True)
    return clips


@dataclass
class ReviewDecision:
    clip_index: int
    action: str  # approve | reject | nudge
    nudge_seconds: float = 0.0
    note: str = ""


class ReviewQueue:
    """Keyboard-shortcut review queue (Part 3 upgrade): a/r/n per clip.

    ``apply`` returns the clips with approved ones first, rejected ones
    removed, and nudged cut points shifted — the human stays in charge.
    """

    def __init__(self, clips: list[Clip]):
        self.clips = list(clips)
        self.decisions: list[ReviewDecision] = []

    def decide(self, clip_index: int, action: str,
               nudge_seconds: float = 0.0, note: str = "") -> None:
        if action not in {"approve", "reject", "nudge"}:
            raise ValueError(f"unknown action: {action!r} (a/r/n)")
        if not 0 <= clip_index < len(self.clips):
            raise IndexError("clip index out of range")
        self.decisions.append(ReviewDecision(clip_index, action,
                                             nudge_seconds, note))

    def apply(self) -> list[Clip]:
        rejected = {d.clip_index for d in self.decisions if d.action == "reject"}
        out: list[Clip] = []
        for i, clip in enumerate(self.clips):
            if i in rejected:
                continue
            for d in self.decisions:
                if d.clip_index == i and d.action == "nudge":
                    cand = clip.candidate
                    clip = Clip(candidate=Candidate(
                        max(0.0, cand.start + d.nudge_seconds),
                        cand.end + d.nudge_seconds, cand.words,
                        cand.first_sentence, cand.last_sentence),
                        score=clip.score, retention_proba=clip.retention_proba,
                        hook_variants=clip.hook_variants,
                        reframe=clip.reframe,
                        campaign_violations=clip.campaign_violations,
                        title=clip.title)
            out.append(clip)
        approved = [c for i, c in enumerate(self.clips)
                    if any(d.clip_index == i and d.action == "approve"
                           for d in self.decisions)]
        rest = [c for c in out if c not in approved]
        return approved + rest

    @property
    def acceptance(self) -> float:
        decided = {d.clip_index for d in self.decisions}
        if not decided:
            return 0.0
        approved = {d.clip_index for d in self.decisions if d.action == "approve"}
        return len(approved) / len(decided)
