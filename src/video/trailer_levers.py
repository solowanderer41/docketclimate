"""
Editorial levers for the trailer pipeline.

The critic scores storyboards against these dials. They are TARGETS, not
generator parameters — the generator's system prompt encodes the structural
rules; levers say which way to lean within them.

Iterate by editing defaults here, not the generator's prompt. Once you have
ERR data across shipped Reels, levers become the unit of experimentation:
ship three Reels with hook_style=held_empty and three with hook_style=
metaphor_chain, compare. The lever values shipped with each Reel should be
logged so the comparison is mechanical, not memory-based.

Five levers, deliberately. Four others (closing_rhythm, political_shot_
directness, held_empty_count, shot_count_target) felt like overengineering
without performance data to ground them. Add them when we have signal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


HookStyle = Literal["held_empty", "metaphor_chain", "declaration"]
TitlePlacement = Literal["late_punchline", "tease_then_punchline"]
MetaphorDensity = Literal["pure", "one_literal_per_act"]
Pace = Literal["ma_heavy", "balanced", "cut_driven"]


@dataclass
class LeverConfig:
    """Editorial dials a storyboard should honor."""

    vo_word_target: int = 50
    """Total VO words across the piece. Range 30-80. Lower forces silence;
    higher allows more evidence per shot."""

    hook_style: HookStyle = "held_empty"
    """How the first 1-2 shots establish stakes.
    - held_empty: locked composition with no subject, music + image only
    - metaphor_chain: 2-3 oblique objects, no VO, no people
    - declaration: a single object or location that names the texture
      bluntly (e.g. close on a flood map being redrawn). Faster, less ma."""

    title_placement: TitlePlacement = "late_punchline"
    """When the thesis appears.
    - late_punchline: title cuts in after the final body shot (current Eco-
      Friendly and Manns pattern)
    - tease_then_punchline: a fragment of the title appears briefly mid-
      piece, then resolves at the end. Use when the article has a literal
      hook the viewer would recognize."""

    metaphor_density: MetaphorDensity = "pure"
    """How literal the imagery can get.
    - pure: every body shot is metaphor; only the political shot names the
      room
    - one_literal_per_act: one literal anchoring shot per act (a person, a
      map, a building) to ground viewers who need it. Trades cinematic
      conviction for legibility."""

    pace: Pace = "ma_heavy"
    """Overall rhythm.
    - ma_heavy: 7-8 shots, average 8-9s each, deep holds (current default
      for waiting/slowness stories)
    - balanced: 9-10 shots, average 6-7s
    - cut_driven: 10 shots, average 5-6s, tighter holds. For acceleration
      stories (storms, market panics, breakthroughs)."""

    def describe(self) -> str:
        """Human-readable brief — drops into the critic's system prompt."""
        return (
            f"- vo_word_target: {self.vo_word_target}\n"
            f"  (total VO words across the body; the storyboard's total_vo_word_count "
            f"should land within ±15% of this)\n"
            f"- hook_style: {self.hook_style}\n"
            f"  (how the opening shots establish texture and withhold topic)\n"
            f"- title_placement: {self.title_placement}\n"
            f"  (when the thesis appears in the piece)\n"
            f"- metaphor_density: {self.metaphor_density}\n"
            f"  (how literal the body imagery is permitted to get)\n"
            f"- pace: {self.pace}\n"
            f"  (shot count, average duration, hold depth)\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "standard": LeverConfig(),
    "urgent": LeverConfig(
        vo_word_target=40,
        hook_style="declaration",
        title_placement="tease_then_punchline",
        metaphor_density="pure",
        pace="cut_driven",
    ),
    "contemplative": LeverConfig(
        vo_word_target=60,
        hook_style="held_empty",
        title_placement="late_punchline",
        metaphor_density="pure",
        pace="ma_heavy",
    ),
}
"""Named bundles. Use 'urgent' for breaking/breakthrough stories and
'contemplative' for waiting/loss stories. 'standard' is the current
default and matches what the generator's prompt is tuned for."""


def get_preset(name: str) -> LeverConfig:
    """Look up a preset by name. Raises if unknown."""
    name = name.lower().strip()
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {name!r}. Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]


def parse_overrides(overrides: str) -> dict:
    """Parse a CLI string like 'vo_word_target=45,pace=balanced' into a dict
    of field overrides. Used to tweak a preset without defining a new one."""
    if not overrides:
        return {}
    out = {}
    for kv in overrides.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Override must be key=value, got: {kv!r}")
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "vo_word_target":
            out[k] = int(v)
        else:
            out[k] = v
    return out


def build_config(preset_name: str = "standard", overrides: str = "") -> LeverConfig:
    """Build a config from a preset name plus optional --levers overrides."""
    base = get_preset(preset_name)
    o = parse_overrides(overrides)
    if not o:
        return base
    merged = {**base.to_dict(), **o}
    return LeverConfig(**merged)
