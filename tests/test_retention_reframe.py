"""Tests: retention predictor, hook lab, 9:16 reframe."""

import numpy as np

from clipfinder.candidates import build_candidates
from clipfinder.data import (make_demo_features, make_demo_transcript,
                            make_retention_training)
from clipfinder.hooklab import hook_lab
from clipfinder.reframe import OUT_H, OUT_W, plan_reframe
from clipfinder.retention import benchmark, train_retention


def test_retention_training_accuracy():
    x, y, _ = make_retention_training()
    model = train_retention(x, y)
    assert model.train_accuracy >= 0.80


def test_retention_held_out_accuracy():
    x, y, true_w = make_retention_training()
    bench = benchmark(x, y, true_w)
    assert bench["held_out_accuracy"] >= 0.75
    assert bench["weight_error_vs_true"] <= 0.25


def test_retention_weights_recover_true_rule():
    x, y, true_w = make_retention_training()
    model = train_retention(x, y)
    # Hook is the strongest true driver; the fit should agree.
    assert model.feature_importance()[0][0] == "hook"


def test_retention_proba_in_range():
    x, y, _ = make_retention_training()
    model = train_retention(x, y)
    p = model.predict_proba(np.array([[0.9, 0.9, 0.9, 0.8, 1.0]]))
    assert 0.0 <= p[0] <= 1.0
    assert p[0] > model.predict_proba(
        np.array([[0.1, 0.1, 0.1, 0.1, 0.0]]))[0]


def test_hook_lab_has_three_variants():
    tr = make_demo_transcript()
    cand = build_candidates(tr)[5]
    variants = hook_lab(cand)
    names = {v.name for v in variants}
    assert "original" in names and "question_first" in names
    assert all(0.0 <= v.hook_score <= 10.0 for v in variants)


def test_hook_lab_deltas_consistent():
    tr = make_demo_transcript()
    cand = build_candidates(tr)[5]
    for v in hook_lab(cand):
        assert abs(v.delta_vs_original - (v.hook_score - hook_lab(cand)[0].hook_score)) < 1e-9


def test_reframe_crop_is_9_by_16():
    tr = make_demo_transcript()
    feats = make_demo_features(tr)
    plan = plan_reframe(feats.face_x, 0.0, 30.0, feats.energy)
    x, y, w, h = plan.crop_pixels(1920, 1080, 5.0)
    assert abs(w / h - 9 / 16) < 0.02
    assert 0 <= x and x + w <= 1920


def test_reframe_smooth_no_jumps():
    tr = make_demo_transcript()
    feats = make_demo_features(tr)
    plan = plan_reframe(feats.face_x, 0.0, 60.0, feats.energy)
    jumps = [abs(b - a) for a, b in zip(plan.centre_x, plan.centre_x[1:])]
    assert max(jumps) <= 0.61  # 30 frames * 0.02 max shift per frame


def test_reframe_fallback_when_no_face():
    face = np.full(60, np.nan)
    plan = plan_reframe(face, 0.0, 10.0)
    assert all(plan.fallback_blur)
    assert all(c == 0.5 for c in plan.centre_x)


def test_output_resolution_constants():
    assert (OUT_W, OUT_H) == (1080, 1920)
