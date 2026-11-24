"""Hook lab — the blueprint's Part 3 upgrade #2.

Generates three A/B opening variants for a clip and scores each one's
hook with the same rubric the ranker uses, so the editor can pick the
opening that retains best instead of guessing:

* ``original`` — as cut.
* ``punchline_first`` — the highest-hook sentence in the clip moves to the
  front (cold open), the rest follows in order.
* ``question_first`` — the opening claim is reframed as a question.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import Candidate
from .scoring import score_hook
from .transcript import Word


@dataclass
class HookVariant:
    name: str
    opening: str
    hook_score: float
    delta_vs_original: float


def _sentences_of(cand: Candidate) -> list[list[Word]]:
    from .transcript import Transcript
    return Transcript(list(cand.words)).sentences()


def _text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def _hook_of_words(words: list[Word], start: float) -> float:
    probe = Candidate(start, words[-1].end if words else start,
                      tuple(words), _text(words[:1]), _text(words[-1:]))
    return score_hook(probe)[0]


def hook_lab(cand: Candidate) -> list[HookVariant]:
    sentences = _sentences_of(cand)
    flat = [w for s in sentences for w in s]
    original_hook = score_hook(cand)[0]
    variants = [HookVariant("original", cand.first_sentence,
                            original_hook, 0.0)]
    if len(sentences) >= 2:
        # Punchline-first: best-hook sentence first, rest in order.
        scored = [(score_hook(Candidate(
            s[0].start, s[-1].end, tuple(s), _text(s), _text(s)))[0], i)
            for i, s in enumerate(sentences)]
        best_i = max(scored)[1]
        if best_i != 0:
            reordered = [sentences[best_i]] + [s for i, s in enumerate(sentences)
                                               if i != best_i]
            words = [w for s in reordered for w in s]
            h = _hook_of_words(words, cand.start)
            variants.append(HookVariant(
                "punchline_first", _text(sentences[best_i]), h, h - original_hook))
    # Question-first: reframe the opening claim as a question.
    first = cand.first_sentence.rstrip(".!")
    if not first.endswith("?"):
        q_words = [Word(t, cand.start, cand.start + 0.4) for t in
                   f"What if {first[0].lower()}{first[1:]}?".split()]
        h = _hook_of_words(q_words + flat, cand.start)
        variants.append(HookVariant("question_first",
                                    f"What if {first[0].lower()}{first[1:]}?",
                                    h, h - original_hook))
    return variants
