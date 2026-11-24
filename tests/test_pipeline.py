"""Tests: pipeline dedupe/top-K, campaign, captions, review queue."""

from clipfinder.campaign import DEMO_BRIEF, CampaignBrief
from clipfinder.candidates import build_candidates
from clipfinder.captions import (build_captions, captions_to_ass,
                                 captions_to_srt, translate_block,
                                 translate_word)
from clipfinder.data import (make_demo_features, make_demo_transcript,
                            make_retention_training)
from clipfinder.pipeline import ReviewQueue, dedupe, find_clips
from clipfinder.retention import train_retention
from clipfinder.scoring import score_candidate


def _clips():
    tr = make_demo_transcript()
    x, y, _ = make_retention_training()
    return find_clips(tr, make_demo_features(tr), train_retention(x, y),
                      DEMO_BRIEF), tr


def test_pipeline_returns_top_clips():
    clips, tr = _clips()
    assert 1 <= len(clips) <= 10  # top 10 per hour, source is ~4 min
    assert clips[0].score.total >= clips[-1].score.total


def test_pipeline_dedupe_overlap():
    clips, _ = _clips()
    for i, a in enumerate(clips):
        for b in clips[i + 1:]:
            inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
            assert inter / min(a.duration, b.duration) <= 0.30 + 1e-9


def test_sidecar_has_required_fields():
    clips, _ = _clips()
    side = clips[0].sidecar()
    for key in ("title", "start", "end", "score", "retention_proba",
                "reason", "hook_variants", "transcript", "output"):
        assert key in side
    assert side["output"].startswith("1080x1920")


def test_campaign_banned_word_flagged():
    brief = CampaignBrief("t", banned_words=("scam",))
    assert brief.check("this is a scam offer", 30.0)
    assert not brief.check("a clean honest clip", 30.0)


def test_campaign_length_bounds():
    brief = CampaignBrief("t", min_length=15.0, max_length=60.0)
    assert brief.check("ok", 10.0)
    assert brief.check("ok", 70.0)
    assert not brief.check("ok", 30.0)


def test_campaign_title_hashtags():
    assert DEMO_BRIEF.title_with_hashtags("My clip").endswith("#indiehackers")


def test_captions_max_words_per_line():
    clips, _ = _clips()
    blocks = build_captions(list(clips[0].candidate.words))
    assert blocks
    assert all(len(line.split()) <= 5 for b in blocks for line in b.lines)
    assert all(len(b.lines) <= 2 for b in blocks)


def test_captions_srt_and_ass():
    clips, _ = _clips()
    blocks = build_captions(list(clips[0].candidate.words))
    assert "-->" in captions_to_srt(blocks)
    ass = captions_to_ass(blocks)
    assert "PlayResX: 1080" in ass and "Dialogue:" in ass


def test_translation_keeps_unknown_words():
    assert translate_word("hello", "es") == "hello"
    assert translate_word("money", "es") == "dinero"
    assert translate_word("money", "zh") == "钱"
    assert translate_word("money", "fr") == "money"


def test_review_queue_approve_reject():
    clips, _ = _clips()
    q = ReviewQueue(clips)
    q.decide(0, "approve")
    q.decide(1, "reject")
    out = q.apply()
    assert len(out) == len(clips) - 1
    assert out[0] is clips[0]
    assert clips[1] not in out


def test_review_queue_nudge_shifts_cut():
    clips, _ = _clips()
    q = ReviewQueue(clips)
    q.decide(0, "nudge", nudge_seconds=2.0)
    out = q.apply()
    assert abs(out[0].start - (clips[0].start + 2.0)) < 1e-9


def test_review_queue_rejects_bad_action():
    clips, _ = _clips()
    q = ReviewQueue(clips)
    try:
        q.decide(0, "delete")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_dedupe_keeps_best():
    tr = make_demo_transcript()
    feats = make_demo_features(tr)
    scored = [(c, score_candidate(c, feats)) for c in build_candidates(tr)]
    kept = dedupe(scored)
    assert len(kept) < len(scored)
    assert kept[0][1].total == max(s[1].total for s in scored)
