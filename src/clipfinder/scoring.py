"""Deterministic rubric scoring — the blueprint's LLM judge, made offline.

Blueprint §1.4 weights (0–10 per criterion, weighted sum):
hook 0.25, self-contained 0.20, emotion 0.15, insight 0.15, humor 0.10,
audio energy 0.05, speaker dynamics 0.05, campaign relevance 0.05.
The blueprint calls for an LLM judge; this package replaces it with a
transparent lexicon + signal rubric so scoring is reproducible, free and
offline — and says so in the README instead of pretending an LLM ran.
Every clip also gets a one-line reason and the exact words that drove the
score (the blueprint's "explainable ranking" upgrade).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .candidates import Candidate
from .features import FeatureTrack

WEIGHTS = {
    "hook": 0.25, "self_contained": 0.20, "emotion": 0.15, "insight": 0.15,
    "humor": 0.10, "audio_energy": 0.05, "speaker_dynamics": 0.05,
    "campaign_relevance": 0.05,
}

CURIOSITY = {"secret", "mistake", "nobody", "never", "nobody", "prove", "changed",
             "warning", "stop", "truth", "lie", "exactly", "insane", "free"}
EMOTION = {"love", "hate", "scary", "amazing", "terrible", "brilliant", "painful",
           "excited", "angry", "proud", "ashamed", "funny", "insane", "crazy"}
INSIGHT = {"framework", "lesson", "rule", "system", "learned", "discovered",
           "numbers", "percent", "revenue", "conversion", "trick", "secret",
           "sweet", "spot", "filter"}
HUMOR = {"joke", "funny", "laugh", "cache", "broke", "hilarious"}
CONTEXT_DEPENDENT = {"as i said", "that thing", "this one", "remember earlier",
                     "like i told", "as mentioned", "that guy", "you know what i mean"}
RISKY = {"damn", "hell", "shit", "stupid", "idiot", "kill", "died", "scam"}

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _first_seconds_words(cand: Candidate, seconds: float = 3.0) -> list[Word]:
    t0 = cand.start
    return [w for w in cand.words if w.start < t0 + seconds]


def score_hook(cand: Candidate) -> tuple[float, list[str]]:
    opening = " ".join(w.text for w in _first_seconds_words(cand))
    toks = _tokens(opening)
    score, evidence = 3.0, []
    if opening.strip().endswith("?"):
        score += 2.5
        evidence.append("opens with a question")
    if any(t in CURIOSITY for t in toks):
        score += 2.5
        evidence.append("curiosity word in first 3 s: "
                        + ", ".join(sorted({t for t in toks if t in CURIOSITY})))
    if re.search(r"\d+", opening):
        score += 1.5
        evidence.append("number in the opening")
    if toks and toks[0] in {"stop", "never", "do", "here", "what", "why", "how"}:
        score += 1.0
        evidence.append("imperative/question opening")
    return min(10.0, score), evidence


def score_self_contained(cand: Candidate) -> tuple[float, list[str]]:
    text = cand.text.lower()
    score, evidence = 8.0, []
    for phrase in CONTEXT_DEPENDENT:
        if phrase in text:
            score -= 2.0
            evidence.append(f"needs context: {phrase!r}")
    if re.search(r"\b(he|she|they|it|that)\b.{0,20}$", cand.first_sentence.lower()):
        score -= 1.0
    if len(_tokens(cand.text)) >= 20:
        evidence.append("complete thought in one clip")
    return max(0.0, min(10.0, score)), evidence


def score_lexicon(cand: Candidate, lexicon: set[str], base: float = 2.0,
                  per_hit: float = 1.6, cap: float = 10.0) -> tuple[float, list[str]]:
    toks = _tokens(cand.text)
    hits = sorted({t for t in toks if t in lexicon})
    return min(cap, base + per_hit * len(hits)), hits


def score_payoff(cand: Candidate) -> tuple[float, list[str]]:
    last = cand.last_sentence.lower()
    if re.search(r"(recap|so,|remember|next week|pick|do just)", last):
        return 8.5, ["ends on a takeaway"]
    if re.search(r"\d+", last):
        return 8.0, ["ends on a number"]
    if last.strip().endswith("?"):
        return 7.0, ["ends on a question"]
    return 5.0, []


@dataclass
class Score:
    total: float
    parts: dict[str, float]
    payoff: float
    reason: str
    evidence: dict[str, list[str]]
    risks: list[str] = field(default_factory=list)


def score_candidate(cand: Candidate, features: FeatureTrack | None = None,
                    campaign_keywords: tuple[str, ...] = ()) -> Score:
    hook, ev_hook = score_hook(cand)
    self_c, ev_self = score_self_contained(cand)
    emotion, ev_emo = score_lexicon(cand, EMOTION)
    insight, ev_ins = score_lexicon(cand, INSIGHT)
    humor, ev_hum = score_lexicon(cand, HUMOR, base=1.0, per_hit=2.0)
    if features is not None:
        humor = min(10.0, humor + 2.0 * features.laughter_seconds(cand.start, cand.end))
        audio = 10.0 * features.mean_energy(cand.start, cand.end)
        changes = features.speaker_changes(cand.start, cand.end)
        speaker = min(10.0, 4.0 + 1.5 * changes)
    else:
        audio, speaker = 5.0, 5.0
    if campaign_keywords:
        toks = set(_tokens(cand.text))
        hits = [k for k in campaign_keywords if k.lower() in toks]
        campaign = min(10.0, 2.0 + 2.0 * len(hits))
    else:
        campaign, hits = 5.0, []
    payoff, ev_pay = score_payoff(cand)
    parts = {"hook": hook, "self_contained": self_c, "emotion": emotion,
             "insight": insight, "humor": humor, "audio_energy": audio,
             "speaker_dynamics": speaker, "campaign_relevance": campaign}
    total = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)
    # Payoff shapes the final score slightly: a clip that fizzles out
    # loses up to half a point, one that lands gains up to half a point.
    total = max(0.0, min(10.0, total + (payoff - 5.0) * 0.1))
    risks = []
    low = cand.text.lower()
    if any(r in low for r in RISKY):
        risks.append("profanity/sensitive language — review before publishing")
    if self_c < 5.0:
        risks.append("needs-context — may mislead if cut alone")
    if any(t in _tokens(cand.text) for t in {"scam", "guarantee", "money back"}):
        risks.append("claim to verify (money-back promise)")
    top = max(parts, key=parts.get)
    reason = (f"Top clip because its {top.replace('_', ' ')} scores "
              f"{parts[top]:.1f}/10 — opens with: "
              f"“{cand.first_sentence[:70]}”")
    evidence = {"hook": ev_hook, "self_contained": ev_self,
                "emotion": ev_emo, "insight": ev_ins, "humor": ev_hum,
                "payoff": ev_pay,
                "campaign_hits": hits}
    return Score(total, parts, payoff, reason, evidence, risks)
