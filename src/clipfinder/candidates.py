"""Candidate segments: sliding windows snapped to sentence boundaries.

Blueprint §1.4: clips are 15–60 s (sweet spot 25–45 s), start and end at a
sentence boundary, never mid-word. We slide a window over sentences and
keep every window in range, then score and dedupe downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from .transcript import Transcript, Word

MIN_LEN = 15.0
MAX_LEN = 60.0
SWEET_MIN = 25.0
SWEET_MAX = 45.0


@dataclass(frozen=True)
class Candidate:
    start: float
    end: float
    words: tuple[Word, ...]
    first_sentence: str
    last_sentence: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def length_fit(self) -> float:
        """1.0 inside the 25–45 s sweet spot, fading to 0 at the limits."""
        d = self.duration
        if SWEET_MIN <= d <= SWEET_MAX:
            return 1.0
        if d < SWEET_MIN:
            return max(0.0, (d - MIN_LEN) / (SWEET_MIN - MIN_LEN))
        return max(0.0, (MAX_LEN - d) / (MAX_LEN - SWEET_MAX))


def build_candidates(transcript: Transcript,
                     min_len: float = MIN_LEN,
                     max_len: float = MAX_LEN) -> list[Candidate]:
    sentences = transcript.sentences()
    spans = transcript.sentence_spans()
    out: list[Candidate] = []
    for i in range(len(sentences)):
        words: list[Word] = []
        for j in range(i, len(sentences)):
            words.extend(sentences[j])
            start, end = spans[i][0], spans[j][1]
            dur = end - start
            if dur > max_len:
                break
            if dur >= min_len:
                out.append(Candidate(start, end, tuple(words),
                                     spans[i][2], spans[j][2]))
    return out
