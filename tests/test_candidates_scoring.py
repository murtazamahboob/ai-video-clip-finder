"""Tests: candidates snap to sentences, scoring follows blueprint weights."""

from clipfinder.candidates import build_candidates
from clipfinder.data import make_demo_features, make_demo_transcript
from clipfinder.features import neutral_track
from clipfinder.scoring import WEIGHTS, score_candidate


def _demo():
    tr = make_demo_transcript()
    return tr, make_demo_features(tr)


def test_candidates_in_length_range():
    tr, _ = _demo()
    cands = build_candidates(tr)
    assert cands
    assert all(15.0 <= c.duration <= 60.0 for c in cands)
    # Every candidate starts at a sentence start and ends at a sentence end.
    starts = {s for s, _, _ in tr.sentence_spans()}
    ends = {e for _, e, _ in tr.sentence_spans()}
    assert all(c.start in starts and c.end in ends for c in cands)


def test_candidates_cover_sweet_spot():
    tr, _ = _demo()
    assert any(25.0 <= c.duration <= 45.0 for c in build_candidates(tr))


def test_length_fit_is_one_in_sweet_spot():
    tr, _ = _demo()
    sweet = [c for c in build_candidates(tr) if 25.0 <= c.duration <= 45.0][0]
    assert sweet.length_fit == 1.0


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    assert WEIGHTS["hook"] == 0.25 and WEIGHTS["self_contained"] == 0.20


def test_hook_question_scores_high():
    tr, feats = _demo()
    cands = build_candidates(tr)
    # The "So what changed?" region is a question-led clip.
    q = [c for c in cands if c.first_sentence.startswith("So what changed?")]
    assert q and score_candidate(q[0], feats).parts["hook"] >= 5.0


def test_scoring_is_deterministic():
    tr, feats = _demo()
    c = build_candidates(tr)[0]
    assert score_candidate(c, feats).total == score_candidate(c, feats).total


def test_score_total_in_range():
    tr, feats = _demo()
    for c in build_candidates(tr)[:20]:
        s = score_candidate(c, feats)
        assert 0.0 <= s.total <= 10.0
        assert s.reason and s.evidence


def test_needs_context_flagged_as_risk():
    from clipfinder.candidates import Candidate
    from clipfinder.transcript import Word
    words = tuple(Word(t, i * 0.4, (i + 1) * 0.4)
                  for i, t in enumerate(
                      "As I said earlier that thing was confusing and "
                      "remember earlier we talked about it all day long "
                      "and nobody knew what would happen next at all".split()))
    cand = Candidate(0.0, words[-1].end, words, "As I said earlier.", "next at all.")
    s = score_candidate(cand, neutral_track(cand.duration))
    assert s.parts["self_contained"] < 8.0
    assert any("needs-context" in r for r in s.risks)


def test_laughter_boosts_humor():
    tr, feats = _demo()
    cands = build_candidates(tr)
    joke = [c for c in cands if "cache" in c.text.lower()]
    assert joke
    assert score_candidate(joke[0], feats).parts["humor"] >= 5.0
