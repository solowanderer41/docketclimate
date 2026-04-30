"""
Trailer-tier storyboard generation for Docket Climate flagship Reels.

Produces, in one Claude API call:
- 1-2 sentence story summary and 1 sentence acid-crisp thesis
- 6-8 punchline title-card candidates
- A Suno music prompt sized for a 60-75s build with a cold drop at the title
- A 3-4 act color script that does dramatic work (not section taxonomy)
- 12-16 shots with Midjourney prompts, Kling motion notes, optional VO

Design principles encoded in the system prompt:
- Metaphor over literal illustration
- Thesis withheld until shot ~12+; title lands as punchline
- Big Short escalation: small dopamine hits while withholding the topic
- Music + image carry first 2-3s; VO joins late
- 16:9 letterbox; Kling watermark stays in lower-right (compose around it)

Slots into the existing src/video/ pipeline. Companion modules to be written:
- midjourney_client.py : auto or manual handoff for stills
- kling_client.py      : i2v on selected MJ stills
- suno_client.py       : music generation from music_prompt
- trailer.py           : final FFmpeg assembly with VO + music ducking + title beat
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Seed exemplar: "When Eco-Friendly Became Illegal" (Apr 2026 flagship)
# This Reel set the format. Including it as a worked example in-prompt is the
# single biggest lever for keeping output on-voice.
# ---------------------------------------------------------------------------

SEED_EXEMPLAR = """\
TITLE: When Eco-Friendly Became Illegal
RUNTIME: 73s   ASPECT: 16:9 letterboxed in 9:16   GENERATOR: MJ -> Kling 3.0 i2v
THESIS (lands at the title, ~60s in): EU regulation ostensibly meant to fight
greenwashing ended up making it illegal to label products "eco-friendly,"
chilling honest claims without addressing underlying harms. Of course
bureaucracy was ineffective.

SHOT FLOW (paraphrased — note metaphor, not illustration):
  Act 1 — industrial blue, cool fluorescent (the system, sterile)
    1. Warehouse aisle, slow dolly forward, ceiling lights         (music only)
    2. Warehouse low angle, ceiling lights receding overhead       (music only)
    3. Macro green recycle stamp on cardboard                      (VO begins)
    4. Same stamp, slight push-in, focus shift
    5. Hand pulling green-labeled product off retail shelf
    6. Tighter on hand and "Eco-" label
  Act 2 — golden hour amber (the institution)
    7. EU Parliament exterior at sunset, flags on roof
    8. Suited silhouette walking warehouse, backlit
    9. EU Parliament chamber, full assembly wide
    10. Same chamber, alternate angle
    11. Gavel slamming in front of EU flag, hand entering frame
    12. News-anchor-style figure, world map background
  Act 3 — dusty orange and mahogany (consequence and complicity)
    13. Dust storm rolling over farmland
    14. Man hiding face behind crumpled document
    15. Formal dinner, men toasting in wood-paneled room
    16. Same dinner, closer on a single toaster
  Coda — washed teal (the actual problem, unspoken)
    17. Blurred eco product on shelf, defocused
    18. Flooded fenceline, gray sky
    19. Bleached white coral underwater
  Title beat (music drops, no VO):
    20. Black. Title: "When Eco-Friendly Became Illegal"
    21. Black. CTA: "More stories like this at the Docket / @docketclimate"

KEY MOVES TO PRESERVE:
- 60 of 73 seconds elapse before the title reveals the topic
- No on-screen text or subtitles during the body
- 4 distinct palettes mark 4 narrative beats; color is doing dramatic work
- Causal chain told only in metaphor: regulation -> market -> industry -> climate
- Closing shots (flood, bleached coral) are the unspoken "meanwhile, the real problem"
- Kling watermark sits in lower-right of every shot; subjects compose upper-left/center
"""


SYSTEM_PROMPT = f"""You are storyboarding a flagship climate-journalism Reel for The Docket. This is NOT an explainer and NOT a balanced summary. It is a 60-75 second cinematic short whose thesis lands as a punchline at the end.

EDITORIAL VOICE
- Acid-crisp and dryly ironic. The implicit framing is "of course X happened — what did anyone expect?"
- Big Short logic: keep small dopamine hits arriving (cuts, tonal shifts, unexpected juxtapositions) while the actual topic is withheld
- A first-time viewer should not know the article's topic until shot 12 or later. They should know the texture (industrial? bureaucratic? domestic? climatic?) and feel dread building

VISUAL LOGIC
- Metaphor, not illustration. NEVER depict the article's claim literally. If the article is about agricultural runoff, do NOT show "a farmer with a chemical sprayer." Show a glass of water on a kitchen counter, a fish market at dawn, a mother watching a child drink from a tap. The viewer assembles meaning.
- Each shot must escalate curiosity. From shot 1 to shot ~10 the viewer is silently asking "what is this even about?" From shot 11 onward the answer is arriving.
- Three or four act color script. Each act has one palette. Color is doing DRAMATIC work, not categorical work. Ignore the publication's section colors entirely — they are taxonomy, not drama.
- Closing 1-2 shots before the title beat should imply the actual climate consequence the article is really about. Wordless.

TECHNICAL CONSTRAINTS
- Output is 16:9 horizontal, letterboxed inside a 9:16 Reel
- Generated via Midjourney stills -> Kling 3.0 image-to-video. Every Kling shot has a watermark in the lower-right. DO NOT compose critical subject matter in the lower-right corner.
- 12-16 shots. Each shot 3-6 seconds. Total body 50-65s. Title 3s. CTA 6s. Total runtime 60-75s.
- Music begins at t=0. VO joins at shot 3 (~6-8s in) and runs through shot N-2 (the last visual before title). Music carries the title and CTA alone.
- Title card is black background, white sans-serif, single line. It IS the thesis. No narration over it.
- No on-screen text during the body. No subtitles. No chyrons. No kinetic typography. The Kling watermark is the only text in shots 1 through N.

VOICEOVER STYLE
- Spare. The body's VO should be 60-100 total words across all shots that have any. Many shots have none.
- Each VO line is one sentence, often a fragment. Declarative, not explanatory.
- Avoid "this means" or "what's happening is" — the VO should not interpret the visuals, it should run alongside them
- The final pre-title VO line should land just before the cut to black, like a closing argument

OUTPUT FORMAT
Return one JSON object matching this schema exactly. No prose outside the JSON. No code fences.

{{
  "story_summary": "1-2 plain-language sentences: what the article is actually about",
  "thesis": "1 sentence: the acid-crisp punchline take. This becomes the title.",
  "title_candidates": ["6-8 short title-card candidates, each <= 8 words, dryly ironic"],
  "music_prompt": "A Suno prompt: genre, instrumentation, mood, tempo, arc. Must support a 60-75s build with a quiet open and a cold drop-out at the title beat. ~40-60 words.",
  "color_script": [
    {{"act": 1, "palette": "single color-script word or phrase", "rationale": "what mood and what narrative beat"}},
    {{"act": 2, "palette": "...", "rationale": "..."}},
    {{"act": 3, "palette": "...", "rationale": "..."}}
  ],
  "shots": [
    {{
      "index": 1,
      "act": 1,
      "duration_s": 4.0,
      "midjourney_prompt": "the MJ prompt: style, subject, framing, lighting, lens, mood. End with --ar 16:9 and any stylization tag. Compose subject UPPER-LEFT or CENTER, never lower-right.",
      "kling_motion": "what moves in the i2v output: camera language (slow dolly forward, rack focus, pull back) and any subject motion. One sentence.",
      "voiceover": null,
      "rationale": "what beat this shot carries and why it is metaphor, not illustration"
    }}
  ],
  "title_card": {{
    "appears_after_shot": 14,
    "selected_title": "your top pick from title_candidates",
    "duration_s": 3.0
  }},
  "cta": {{
    "text": "More stories like this at the Docket\\n@docketclimate",
    "duration_s": 6.0
  }}
}}

EXEMPLAR — the Reel that defined this format:
{SEED_EXEMPLAR}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    index: int
    act: int
    duration_s: float
    midjourney_prompt: str
    kling_motion: str
    voiceover: Optional[str]
    rationale: str


@dataclass
class ColorBeat:
    act: int
    palette: str
    rationale: str


@dataclass
class TitleCard:
    appears_after_shot: int
    selected_title: str
    duration_s: float


@dataclass
class CTA:
    text: str
    duration_s: float


@dataclass
class TrailerStoryboard:
    story_summary: str
    thesis: str
    title_candidates: list[str]
    music_prompt: str
    color_script: list[ColorBeat]
    shots: list[Shot]
    title_card: TitleCard
    cta: CTA

    @classmethod
    def from_json(cls, data: dict) -> "TrailerStoryboard":
        return cls(
            story_summary=data["story_summary"],
            thesis=data["thesis"],
            title_candidates=data["title_candidates"],
            music_prompt=data["music_prompt"],
            color_script=[ColorBeat(**c) for c in data["color_script"]],
            shots=[Shot(**s) for s in data["shots"]],
            title_card=TitleCard(**data["title_card"]),
            cta=CTA(**data["cta"]),
        )

    def total_runtime_s(self) -> float:
        body = sum(s.duration_s for s in self.shots)
        return body + self.title_card.duration_s + self.cta.duration_s

    def shots_with_vo(self) -> list[Shot]:
        return [s for s in self.shots if s.voiceover]

    def total_vo_word_count(self) -> int:
        return sum(len(s.voiceover.split()) for s in self.shots if s.voiceover)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        """Return a list of human-readable warnings. Empty list = clean."""
        warnings = []
        rt = self.total_runtime_s()
        if rt < 55 or rt > 80:
            warnings.append(f"Runtime {rt:.1f}s outside target 60-75s window")
        if len(self.shots) < 12 or len(self.shots) > 16:
            warnings.append(f"{len(self.shots)} shots outside target 12-16")
        if any(s.duration_s < 2.5 or s.duration_s > 7 for s in self.shots):
            bad = [s.index for s in self.shots if s.duration_s < 2.5 or s.duration_s > 7]
            warnings.append(f"Shots {bad} have durations outside 2.5-7s")
        if self.title_card.appears_after_shot < len(self.shots) - 2:
            warnings.append(
                f"Title appears after shot {self.title_card.appears_after_shot} "
                f"of {len(self.shots)} — should appear near the end (last 1-2 shots)"
            )
        wc = self.total_vo_word_count()
        if wc < 40 or wc > 130:
            warnings.append(f"VO total {wc} words outside target 60-100")
        # First two shots should have no VO (music + image hook)
        if any(s.voiceover for s in self.shots[:2]):
            warnings.append("Shots 1-2 should have no VO (music + image hook)")
        if self.title_card.selected_title not in self.title_candidates:
            warnings.append("selected_title not in title_candidates")
        return warnings


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_trailer_storyboard(
    article_title: str,
    article_body: str,
    article_section: str,
    *,
    model: str = "claude-opus-4-7",
    client: Optional[Anthropic] = None,
) -> TrailerStoryboard:
    """Generate a full storyboard, music prompt, and title candidates for one flagship Reel.

    article_section is passed for context only. The trailer tier deliberately
    ignores section-color theming — color does dramatic work here.
    """
    client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_prompt = (
        f"ARTICLE SECTION: {article_section}\n"
        f"ARTICLE TITLE: {article_title}\n\n"
        f"ARTICLE BODY:\n{article_body}\n\n"
        "Generate the storyboard now. Return JSON only, no prose outside the JSON, no code fences."
    )

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    # Defensive: strip code fences if the model included them anyway
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    data = json.loads(raw)
    return TrailerStoryboard.from_json(data)


# ---------------------------------------------------------------------------
# Render to a markdown brief for human curation
# ---------------------------------------------------------------------------

def render_storyboard_markdown(sb: TrailerStoryboard) -> str:
    """Render the storyboard as a markdown brief — the artifact you read on
    Sunday afternoon while generating MJ stills, and again on Monday morning
    while curating Kling outputs."""
    out = []
    out.append("# Trailer storyboard\n")
    out.append(f"**Story.** {sb.story_summary}\n")
    out.append(f"**Thesis.** {sb.thesis}\n")
    out.append(
        f"**Runtime.** {sb.total_runtime_s():.1f}s "
        f"({len(sb.shots)} shots, {sb.total_vo_word_count()} VO words)\n"
    )

    warnings = sb.validate()
    if warnings:
        out.append("\n**⚠ Warnings**")
        for w in warnings:
            out.append(f"- {w}")

    out.append("\n## Title candidates")
    for i, t in enumerate(sb.title_candidates, 1):
        marker = "★" if t == sb.title_card.selected_title else " "
        out.append(f"{marker} {i}. {t}")

    out.append("\n## Music (Suno)")
    out.append(f"> {sb.music_prompt}")

    out.append("\n## Color script")
    for c in sb.color_script:
        out.append(f"- **Act {c.act} — {c.palette}.** {c.rationale}")

    out.append("\n## Shots")
    current_act = 0
    for s in sb.shots:
        if s.act != current_act:
            current_act = s.act
            out.append(f"\n### Act {current_act}")
        out.append(f"\n**Shot {s.index}** ({s.duration_s}s)")
        out.append(f"- *MJ:* `{s.midjourney_prompt}`")
        out.append(f"- *Motion:* {s.kling_motion}")
        out.append(f"- *VO:* {s.voiceover if s.voiceover else '—'}")
        out.append(f"- *Why:* {s.rationale}")

    out.append(f"\n## Title card")
    out.append(
        f"After shot {sb.title_card.appears_after_shot}, "
        f"hold {sb.title_card.duration_s}s on black."
    )
    out.append(f"> **{sb.title_card.selected_title}**")

    out.append(f"\n## CTA ({sb.cta.duration_s}s)")
    out.append(f"> {sb.cta.text}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI for quick local testing: pipe an article in, get markdown out
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Generate a flagship-tier trailer storyboard.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--section", required=True, help="lived | systems | science | futures | archive | lab")
    parser.add_argument("--body-file", required=True, help="path to article body text")
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of markdown")
    args = parser.parse_args()

    with open(args.body_file) as f:
        body = f.read()

    sb = generate_trailer_storyboard(
        article_title=args.title,
        article_body=body,
        article_section=args.section,
        model=args.model,
    )

    if args.json:
        json.dump(sb.to_dict(), sys.stdout, indent=2)
    else:
        sys.stdout.write(render_storyboard_markdown(sb))
    sys.stdout.write("\n")
