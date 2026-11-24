"""CLI: ``clipfinder demo | rank | benchmark | review``.

* ``demo`` — run the bundled synthetic ~4-minute podcast through the full
  pipeline and write ``outputs/clips.json`` + ``outputs/report.md``.
* ``rank`` — rank clips from an SRT/VTT transcript file (neutral features
  unless a features JSON is supplied).
* ``benchmark`` — retention predictor: 80/20 split accuracy on the bundled
  labelled set, plus the true-weight error.
* ``review`` — print the review queue with keyboard shortcuts (a/r/n).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .campaign import DEMO_BRIEF
from .data import (make_demo_features, make_demo_transcript,
                   make_retention_training)
from .features import neutral_track
from .pipeline import ReviewQueue, find_clips
from .retention import benchmark, train_retention
from .transcript import parse_srt, parse_vtt


def _demo(output_dir: Path) -> dict:
    transcript = make_demo_transcript()
    features = make_demo_features(transcript)
    x, y, true_w = make_retention_training()
    model = train_retention(x, y)
    bench = benchmark(x, y, true_w)
    clips = find_clips(transcript, features, model, DEMO_BRIEF)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": transcript.title,
        "source_duration_s": round(transcript.duration, 1),
        "n_words": len(transcript),
        "retention_benchmark": bench,
        "top_clips": [c.sidecar() for c in clips],
    }
    (output_dir / "clips.json").write_text(json.dumps(report, indent=2))
    lines = [f"# AI Video Clip Finder — demo report",
             f"\nSource: {transcript.title} ({transcript.duration:.0f}s, "
             f"{len(transcript)} words, synthetic — no real video used)",
             f"\nRetention predictor: held-out accuracy "
             f"{bench['held_out_accuracy']:.3f} "
             f"(train {bench['train_accuracy']:.3f}), "
             f"weight error vs true rule {bench['weight_error_vs_true']:.3f}",
             f"\n## Top {len(clips)} clips\n",
             "| # | Clip | Score | Retention | Campaign |",
             "|---|---|---|---|---|"]
    for i, c in enumerate(clips, 1):
        lines.append(f"| {i} | {c.start:.0f}–{c.end:.0f}s ({c.duration:.0f}s) | "
                     f"{c.score.total:.2f} | {c.retention_proba:.2f} | "
                     f"{'OK' if c.campaign_ok else '; '.join(c.campaign_violations)} |")
    for c in clips:
        lines.append(f"\n### {c.title}\n\n{c.score.reason}\n")
        lines.append(f"Hook variants: " + "; ".join(
            f"{v.name} {v.hook_score:.1f} ({v.delta_vs_original:+.1f})"
            for v in c.hook_variants))
        if c.score.risks:
            lines.append("Risks: " + "; ".join(c.score.risks))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    for c in clips:
        print(f"{c.start:6.1f}s–{c.end:6.1f}s  score {c.score.total:4.2f}  "
              f"retention {c.retention_proba:.2f}  {c.title}")
    print(f"\n{len(clips)} clips from {transcript.duration:.0f}s of source; "
          f"retention held-out accuracy {bench['held_out_accuracy']:.3f}")
    return report


def _rank(args: argparse.Namespace) -> int:
    path = Path(args.transcript)
    text = path.read_text()
    transcript = parse_vtt(text) if path.suffix.lower() == ".vtt" else parse_srt(text)
    features = neutral_track(transcript.duration)
    clips = find_clips(transcript, features, None, None)
    for c in clips:
        print(f"{c.start:6.1f}s–{c.end:6.1f}s  score {c.score.total:4.2f}  {c.title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipfinder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--output-dir", default="outputs")
    p = sub.add_parser("rank")
    p.add_argument("transcript", help="SRT or VTT transcript file")
    sub.add_parser("benchmark")
    sub.add_parser("review")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        _demo(Path(args.output_dir))
        return 0
    if args.cmd == "rank":
        return _rank(args)
    if args.cmd == "benchmark":
        x, y, true_w = make_retention_training()
        print(json.dumps(benchmark(x, y, true_w), indent=2))
        return 0
    if args.cmd == "review":
        transcript = make_demo_transcript()
        clips = find_clips(transcript, make_demo_features(transcript),
                           train_retention(*make_retention_training()[:2]),
                           DEMO_BRIEF)
        queue = ReviewQueue(clips)
        print("Review queue — press a (approve), r (reject), n (nudge) per clip:")
        for i, c in enumerate(clips):
            print(f"[{i}] {c.start:.0f}–{c.end:.0f}s score {c.score.total:.2f} "
                  f"— {c.title}")
        print(f"\n{len(clips)} clips queued. Use ReviewQueue.decide(i, 'approve'|'reject'|'nudge') in code, or the demo script.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
