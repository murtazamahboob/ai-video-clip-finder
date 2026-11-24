"""Campaign mode — the blueprint's Part 3 upgrade #3.

A campaign brief (required hashtags, banned words, allowed length,
must-include handle) is checked against every clip automatically, so a
pay-per-view campaign never gets a clip it has to reject. A clip that
fails is flagged, never silently fixed: the editor sees the violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CampaignBrief:
    name: str
    required_hashtags: tuple[str, ...] = ()
    banned_words: tuple[str, ...] = ()
    must_include_handle: str = ""
    min_length: float = 15.0
    max_length: float = 60.0

    def check(self, text: str, duration: float,
              title: str = "") -> list[str]:
        violations: list[str] = []
        low = f"{text} {title}".lower()
        for word in self.banned_words:
            if word.lower() in low:
                violations.append(f"banned word: {word!r}")
        if self.must_include_handle and self.must_include_handle.lower() not in low:
            violations.append(f"missing required handle: {self.must_include_handle!r}")
        if duration < self.min_length:
            violations.append(f"too short: {duration:.1f}s < {self.min_length:.0f}s")
        if duration > self.max_length:
            violations.append(f"too long: {duration:.1f}s > {self.max_length:.0f}s")
        return violations

    def title_with_hashtags(self, title: str) -> str:
        tags = " ".join(f"#{t.lstrip('#')}" for t in self.required_hashtags)
        return f"{title} {tags}".strip()


# The bundled demo campaign: the podcast's own clipping campaign.
# Hashtags/handle are appended at export (title_with_hashtags), so the
# brief does not demand them inside the spoken transcript itself.
DEMO_BRIEF = CampaignBrief(
    name="Build In Public — Shorts campaign",
    required_hashtags=("buildinpublic", "indiehackers"),
    banned_words=("scam", "guarantee"),
    must_include_handle="",
    min_length=15.0, max_length=60.0,
)
