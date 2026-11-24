# AI Video Clip Finder

Turn a long video into ranked, captioned, vertical short clips — automatically.
Feed it a transcript with word timestamps and a per-second feature track
(audio energy, laughter, face position, speaker), and it returns the top
clips per hour of source: 15–60 second segments, each with a score, a
retention prediction, hook A/B variants, a 9:16 face-tracked reframe plan,
karaoke captions and a campaign check.

**Offline, deterministic, stdlib + NumPy only.** No Whisper, no OpenCV, no
GPU, no LLM, no API key. The bundled demo runs on a synthetic podcast
transcript written for this project — no real video is downloaded or used.

> **Rights first:** only clip videos you own or are licensed to clip.
> Clipping campaigns usually supply approved source material; downloading
> other people's videos to repost can violate copyright and platform rules.
> The pipeline never downloads a video — ingest is your step, from your own
> material.

## Demo results (verified)

Bundled synthetic podcast: *"Build In Public — the mistake that cost me
three months"*, 258 s, 53 sentences, 4 speakers moments, laughter spikes.

| Metric | Result |
|---|---|
| Tests | **40/40 pass** (re-verified in a fresh venv, demo identical) |
| Clips found | **8** from 258 s of source |
| Top clip | 29.1–79.8 s, score **6.70/10**, retention **0.83** — *"The first call lasted nine minutes and changed everything"* |
| Campaign check | **8/8 clips pass** the demo campaign (no banned words, 15–60 s) |
| Retention predictor | train accuracy **0.906**, held-out **0.781** on 160 labelled clips |
| Retention weights vs true rule | cosine similarity **0.967**, normalised weight error **0.099** |

```
clipfinder demo
  29.1s–  79.8s  score 6.70  retention 0.83  The first call lasted nine minutes...
 176.3s– 226.7s  score 6.38  retention 0.77  That is insane, did you sleep at all...
 131.1s– 180.2s  score 6.26  retention 0.69  Forty nine dollars brought fewer...
  94.6s– 145.7s  score 5.89  retention 0.71  Nobody cares about your features...
  70.2s– 104.7s  score 5.51  retention 0.64  Let me show you the numbers...
 222.9s– 253.4s  score 5.12  retention 0.49  So, to recap, talk to users...
  13.3s–  32.6s  score 4.92  retention 0.55  I built a beautiful product...
   0.5s–  18.3s  score 4.46  retention 0.31  Welcome back to Build In Public...
```

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Bundled demo: rank clips from the synthetic podcast
clipfinder demo

# Retention predictor benchmark (80/20 split, true-weight check)
clipfinder benchmark

# Rank clips from your own SRT/VTT transcript (your own video only)
clipfinder rank my-video.srt

# Review queue (approve / reject / nudge cut points)
clipfinder review

# Or run the demo script directly
python examples/demo.py
```

## Python API

```python
from clipfinder import (DEMO_BRIEF, find_clips, make_demo_features,
                        make_demo_transcript, make_retention_training,
                        train_retention)

transcript = make_demo_transcript()          # your SRT/VTT in real use
features = make_demo_features(transcript)    # per-second energy/face/speaker
x, y, _ = make_retention_training()
model = train_retention(x, y)

clips = find_clips(transcript, features, model, DEMO_BRIEF)
top = clips[0]
print(top.title, top.score.total, top.retention_proba)
print(top.score.reason)                      # one-line explainable ranking
print(top.sidecar())                         # JSON sidecar: score, hook, risks
print(top.captions_srt())                    # karaoke captions for the clip
```

## How it works

1. **Transcript** — word-level timestamps from SRT/VTT (or a real ASR run).
   Sentences are the unit everything snaps to.
2. **Candidates** — sliding windows over sentences, 15–60 s (sweet spot
   25–45 s), start and end at a sentence boundary, never mid-word.
3. **Score** — the blueprint's weights, as a transparent rubric instead of
   an LLM judge (so it is reproducible, free and offline): hook 0.25,
   self-contained 0.20, emotion 0.15, insight 0.15, humor 0.10, audio
   energy 0.05, speaker dynamics 0.05, campaign relevance 0.05. A payoff
   adjustment (±0.5) rewards clips that land their ending. Risks are
   flagged, never hidden: profanity/sensitive language, needs-context cuts,
   claims to verify.
4. **Retention predictor** — logistic regression trained from scratch on
   your own labelled clips (hook, emotion, insight, audio energy, length
   fit → stayed past 30 s?). The bundled set has a known ground-truth rule,
   so the fit is checked against the true weights.
5. **Hook lab** — three A/B openings per clip (original, punchline-first
   cold open, question-first), each re-scored with the same hook rubric.
6. **Dedupe** — reject clips with time overlap above 30% or transcript
   similarity above 0.85; return the top 10 per hour of source.
7. **Reframe** — 9:16 crop (1080×1920) tracking the active speaker's face,
   EMA smoothing (alpha 0.20), max shift limited per frame, crop locked to
   the dominant speaker for at least 1.5 s, blurred-background fit when no
   face is visible, 15% punch-in zoom on slow moments.
8. **Captions** — karaoke word timings, 3–5 words per line, max 2 lines,
   SRT + ASS export, deterministic Spanish/Chinese word-map translation
   (words it does not know are kept, never invented).
9. **Campaign mode** — the brief (required hashtags, banned words, allowed
   length, must-include handle) is checked against every clip; failures are
   listed as violations, never silently fixed.
10. **Review queue** — approve (`a`), reject (`r`) or nudge (`n`) cut points
    in seconds; the human stays in charge of what ships.

## Project layout

```
src/clipfinder/
  transcript.py   SRT/VTT parsing, sentences, word timestamps
  features.py     per-second feature track (energy, laughter, face, speaker)
  candidates.py   sentence-boundary windows, 15–60 s
  scoring.py      blueprint-weighted rubric + explainable reasons + risks
  retention.py    logistic regression from scratch + benchmark
  hooklab.py      3 A/B opening variants, re-scored
  reframe.py      face-tracked 9:16 crop, EMA smoothing, smart zoom
  captions.py     karaoke captions, SRT/ASS, ES/ZH word-map translation
  campaign.py     campaign brief validation
  pipeline.py     candidates → score → dedupe → top-K, review queue
  data.py         synthetic demo podcast + labelled retention set
  cli.py          demo | rank | benchmark | review
tests/            40 tests across 4 files
examples/demo.py  one-command demo
outputs/          clips.json + report.md from the demo run
```

## Honest limits

- **No ASR, no ffmpeg, no face detection in this package.** A real
  deployment ingests with ffmpeg + faster-whisper (word timestamps, VAD)
  + a face detector and feeds the transcript + feature track in — those
  are adapters, not hidden dependencies. What is here is the ranking,
  reframing, captioning and campaign brain, fully testable offline.
- **The scorer is a rubric, not an LLM judge.** The blueprint calls for an
  LLM to judge hook / insight / emotion; this package implements those
  criteria as a deterministic lexicon + signal rubric so anyone can rerun
  it for free. It will miss subtle jokes and brilliant-but-quiet insights
  an LLM would catch — the retention predictor is the part that learns
  from your real results.
- **Translation is a word map, not machine translation.** Full MT needs an
  API; the map covers the demo vocabulary and keeps unknown words.
- **Scores are relative, not view guarantees.** Retrain the retention
  predictor on your own clips' real view-duration data before trusting it
  on a new channel — that is the point of the upgrade.

## Blueprint

Built from the ML Projects Master Blueprint, Project 1: AI Video Clip
Finder — including its Part 3 upgrades: retention predictor trained on
your own results, hook lab with A/B variants, campaign mode,
multi-language captions, smart B-roll/zooms, explainable ranking, and a
review queue with keyboard shortcuts.
