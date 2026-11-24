"""AI Video Clip Finder — turn a long video's transcript into ranked,
captioned, vertical short clips. Offline, deterministic, stdlib + NumPy.
"""

from .campaign import DEMO_BRIEF, CampaignBrief
from .candidates import build_candidates
from .data import make_demo_features, make_demo_transcript, make_retention_training
from .pipeline import ReviewQueue, find_clips
from .retention import benchmark, train_retention

__all__ = ["DEMO_BRIEF", "CampaignBrief", "ReviewQueue", "benchmark",
           "build_candidates", "find_clips", "make_demo_features",
           "make_demo_transcript", "make_retention_training", "train_retention"]
__version__ = "1.0.0"
