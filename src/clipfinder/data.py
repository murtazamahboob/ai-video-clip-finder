"""Bundled demo material — a synthetic ~4-minute podcast, written for
this project. No real video, audio or transcript is downloaded or used:
every word here is invented, so the demo runs anywhere, offline, with no
rights question at all. Real videos must be ones the user owns or is
licensed to clip — the pipeline refuses nothing, but the README and the
campaign check both say so, and the demo never pretends otherwise.
"""

from __future__ import annotations

import numpy as np

from .features import FeatureTrack, make_synthetic_track
from .transcript import Transcript, Word

# (sentence, speaker, hot?) — hot sentences get an energy/laughter spike.
SENTENCES: list[tuple[str, str, bool]] = [
    ("Welcome back to Build In Public, the show where we build software live.", "A", False),
    ("Today I am going to tell you the one mistake that cost me three months.", "A", True),
    ("I built a beautiful product that nobody wanted, and I can prove it.", "A", True),
    ("We had twelve hundred signups and zero paying customers.", "A", False),
    ("So what changed?", "B", False),
    ("I stopped building features and started calling users instead.", "A", False),
    ("The first call lasted nine minutes and changed everything.", "A", True),
    ("She said your onboarding is confusing, and she was right.", "B", False),
    ("We deleted half the dashboard that same afternoon.", "A", False),
    ("Here is the framework I use now before I write a single line of code.", "A", True),
    ("First, find ten people with the problem and talk to them.", "A", False),
    ("Second, sell the solution before you build it.", "A", False),
    ("Third, charge from day one, even if it feels scary.", "A", False),
    ("If nobody will pay, you have learned something cheap.", "A", True),
    ("That sounds harsh, but it saved my startup.", "B", False),
    ("Let me show you the numbers, because the numbers do not lie.", "A", False),
    ("Our conversion rate went from zero point four percent to six percent.", "A", True),
    ("Same product, same traffic, just a clearer promise.", "A", False),
    ("The secret was one sentence on the landing page.", "A", True),
    ("We do your invoicing in ninety seconds, or your money back.", "A", False),
    ("Nobody cares about your features, they care about their Friday afternoon.", "B", True),
    ("That line took me three weeks to write, by the way.", "A", False),
    ("Okay, quick joke break, why did the developer go broke?", "B", True),
    ("Because he used up all his cache.", "B", True),
    ("Right, back to the serious part, pricing.", "A", False),
    ("Most founders price by guessing, and guessing is expensive.", "A", False),
    ("We tested three prices in three weeks, here is what happened.", "A", True),
    ("Nineteen dollars brought volume but the wrong customers.", "A", False),
    ("Forty nine dollars brought fewer signups and triple the revenue.", "A", True),
    ("Ninety nine dollars killed the trial completely, do not do that.", "A", False),
    ("The lesson is simple, price is a filter, not just a number.", "A", False),
    ("Now the part nobody tells you about launching on a Monday.", "A", True),
    ("Do not launch on a Monday, your users are drowning in email.", "A", False),
    ("Tuesday morning, ten o clock, that is the sweet spot.", "A", False),
    ("We got four hundred signups in the first six hours.", "A", True),
    ("And here is the spreadsheet I used to track every single one.", "A", False),
    ("Every signup got a personal email, all four hundred of them.", "B", False),
    ("That is insane, did you sleep at all that week?", "B", True),
    ("About four hours a night, would not recommend it.", "A", True),
    ("But thirty of those people are still paying customers today.", "A", False),
    ("Which brings me to retention, the metric that actually matters.", "A", True),
    ("Acquisition is vanity, retention is sanity, I learned that the hard way.", "A", False),
    ("Our churn dropped from eleven percent to three percent in two months.", "A", True),
    ("The trick was a five minute onboarding call for every new team.", "A", False),
    ("It does not scale, and that is exactly why it works.", "A", True),
    ("Big companies cannot do this, you can, use that advantage.", "B", False),
    ("So, to recap, talk to users, sell first, charge early.", "A", False),
    ("Fix the one sentence that explains your product.", "A", False),
    ("Price like a filter, launch on a Tuesday, and call every new customer.", "A", True),
    ("If you do just one of those things this week, pick the user calls.", "A", False),
    ("Next week we rebuild the landing page live, bring your worst copy.", "A", True),
    ("Thanks for listening, and remember, ship it before it feels ready.", "A", False),
    ("See you next Tuesday, same time, same slightly broken microphone.", "A", True),
]

WPS = 2.6          # words per second, natural speech pace
SENTENCE_GAP = 0.4  # seconds between sentences
HOT_PAUSE = 1.2     # extra pause after a hot (laugh / punchline) sentence


def make_demo_transcript() -> Transcript:
    words: list[Word] = []
    t = 0.5
    for text, speaker, hot in SENTENCES:
        tokens = text.split()
        dur = len(tokens) / WPS
        for i, tok in enumerate(tokens):
            words.append(Word(tok, t + i / WPS, t + (i + 1) / WPS, speaker))
        t += dur + SENTENCE_GAP + (HOT_PAUSE if hot else 0.0)
    return Transcript(words, title="Build In Public — the mistake that cost me three months",
                      source="synthetic demo (written for this project)")


def make_demo_features(transcript: Transcript) -> FeatureTrack:
    hot_seconds: list[int] = []
    for (start, end, _), (_, _, hot) in zip(transcript.sentence_spans(), SENTENCES):
        if hot:
            hot_seconds.append(int(start))
    return make_synthetic_track(transcript.duration, seed=42,
                                hot_seconds=tuple(hot_seconds))


def make_retention_training(n: int = 160, seed: int = 7):
    """Labelled clip set for the retention predictor.

    Features per clip: [hook, emotion, insight, audio_energy, length_fit]
    (each 0..1). Label: a viewer stayed past 30 s. The ground truth is a
    known weighted rule plus noise, so the fitted model can be checked
    against the true weights — an honest, verifiable benchmark.
    """
    rng = np.random.default_rng(seed)
    true_w = np.array([0.45, 0.20, 0.20, 0.10, 0.05])
    x = rng.random((n, 5))
    # Length fit peaks near 30-40 s; encode it directly as a feature.
    logits = x @ true_w - 0.42 + rng.normal(0, 0.08, n)
    y = (logits > 0).astype(int)
    return x, y, true_w
