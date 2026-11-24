"""Run the bundled demo: rank clips from the synthetic ~4-minute podcast
and print the top clips with scores and retention predictions.
Writes outputs/clips.json and outputs/report.md.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipfinder.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--output-dir",
                           str(Path(__file__).resolve().parents[1] / "outputs")]))
