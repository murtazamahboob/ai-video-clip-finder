"""Transcript model: word-level timestamps, SRT/VTT parsing, sentences.

The clip finder works on a transcript with word timestamps plus a
per-second feature track (see ``features.py``). That is exactly what a real
ingest step produces — Whisper/faster-whisper with word timestamps, VAD and
a face detector — but this package never calls those tools itself: the core
is offline, deterministic and stdlib-only, and the bundled demo transcript
is synthetic, written for this project. Adapters for real ffmpeg/Whisper
output are a clearly labelled extension point, not a hidden dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

SENTENCE_END = re.compile(r"[.!?]+$")


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds
    speaker: str = "A"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    title: str = ""
    source: str = ""  # e.g. "synthetic demo" — never claim a real video

    def __len__(self) -> int:
        return len(self.words)

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def words_in_range(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.start >= start - 1e-9 and w.end <= end + 1e-9]

    def sentences(self) -> list[list[Word]]:
        """Group words into sentences, split on terminal punctuation."""
        out: list[list[Word]] = []
        cur: list[Word] = []
        for w in self.words:
            cur.append(w)
            if SENTENCE_END.search(w.text):
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    def sentence_spans(self) -> list[tuple[float, float, str]]:
        """(start, end, text) per sentence — the unit candidates snap to."""
        return [(s[0].start, s[-1].end, " ".join(w.text for w in s))
                for s in self.sentences()]


def _parse_time(stamp: str) -> float:
    stamp = stamp.strip().replace(",", ".")
    parts = stamp.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:  # pragma: no cover - defensive
        raise ValueError(f"bad timestamp: {stamp!r}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _fmt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d},{ms % 1000:03d}"


def parse_srt(text: str, title: str = "") -> Transcript:
    """Parse SRT subtitles into word timestamps (one word per token).

    SRT gives line-level timing, so words inside a cue share its span,
    split evenly — the same approximation every SRT-based cutter uses.
    Word-level SRT/VTT from a real ASR run keeps its own timings.
    """
    words: list[Word] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        timing = lines[1] if "-->" in lines[1] else (lines[0] if "-->" in lines[0] else None)
        body = lines[2:] if "-->" in lines[1] else lines[1:]
        if timing is None:
            continue
        start_s, end_s = timing.split("-->")
        start, end = _parse_time(start_s), _parse_time(end_s)
        tokens = " ".join(body).split()
        if not tokens:
            continue
        span = (end - start) / len(tokens)
        for i, tok in enumerate(tokens):
            words.append(Word(tok, start + i * span, start + (i + 1) * span))
    words.sort(key=lambda w: w.start)
    return Transcript(words, title=title, source="srt")


def parse_vtt(text: str, title: str = "") -> Transcript:
    """Parse WebVTT the same way as SRT (headers and NOTE blocks skipped)."""
    cleaned = "\n".join(
        ln for ln in text.splitlines()
        if not ln.strip().upper().startswith(("WEBVTT", "NOTE", "KIND:", "LANGUAGE:"))
    )
    return parse_srt(cleaned, title=title)


def words_to_srt(words: list[Word]) -> str:
    """Render words as one SRT cue per sentence (used for clip captions)."""
    tr = Transcript(list(words))
    cues = []
    for i, (s, e, txt) in enumerate(tr.sentence_spans(), 1):
        cues.append(f"{i}\n{_fmt_time(s)} --> {_fmt_time(e)}\n{txt}\n")
    return "\n".join(cues)
