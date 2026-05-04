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
from dataclasses import dataclass, asdict, field
from typing import List, Optional
 
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
- Big Short logic: keep the topic withheld while small juxtapositions and tonal shifts arrive. Dread builds before the subject is named.
- A first-time viewer should not know the article's topic until the final 1-2 shots. They should know the texture (industrial? bureaucratic? domestic? climatic?) and feel an emotion accumulating.
 
THE GOVERNING PRINCIPLE — MA (間)
This is the most important rule. Kurosawa's concept of negative space, the held interval that gives the surrounding elements meaning. The piece is not a montage. It is a sequence of held images with deliberate silence and stillness between events.
 
- Each shot has TWO durations: an "establish" phase where the viewer reads the frame (~2-3s), and a "held empty" phase where nothing new arrives and the held duration itself becomes the experience. The held empty is where the feeling lands.
- Cuts come AFTER the viewer has finished reading the frame, not on the read. If a shot is 7s, the viewer "got it" at 2s; the next 5s are the work.
- Some shots have NO voiceover at all. The held silence is part of the editorial argument — especially when the article's topic is itself about waiting, opacity, slowness, or absence.
- Honor the article's own grammar. If the article is about waiting (FEMA), shots hold longer. If it is about acceleration (storms, market panics), shots can be tighter — but ma is still present in the title beat and at least one mid-piece pause.
 
VISUAL LOGIC
- Metaphor, not illustration. NEVER depict the article's claim literally. Show evidence of a person, never the person. Show consequence, never the announcement of consequence. The viewer assembles meaning from the held image.
- 3-4 act color script. Each act has one palette. Color does DRAMATIC work, not categorical work. Ignore any publication section-color taxonomy — it is irrelevant here.
- The closing shot before the title is the one place the editorial stance can name the room — a single political/specific image (an unsigned signature line, a postponed-meeting notice, an empty chair at a long table). Everything else is oblique. This shot earns its directness by being the only one.
 
THREE STRUCTURAL REQUIREMENTS — these are non-negotiable
 
1. ONE HELD EMPTY shot. A composition with no human subject and no narrative information that progresses, held for 5+ seconds of held_empty_s. Locked camera. The viewer's eye searches the frame for a subject and does not find one. This is where ma lives most explicitly. Examples: an empty fluorescent corridor with no figure walking through it; a doorframe with no one in it; a single object on a surface with the rest of the frame negative space; a horizon with nothing crossing it. Mark this shot with is_held_empty: true.
 
2. ONE REPEATED COMPOSITION. A shot whose composition matches an earlier shot in the piece — same room, same framing, same camera position — but with subtraction. Something present in the first instance is gone in the second: light, an object, a person's evidence, warmth. The viewer registers the loss without being told. The MJ prompts for the paired shots should be near-identical except for the subtracted elements; in production they will be generated from the same MJ seed. Record the indices of the paired shots in repeated_composition_shot_indices at the top level (e.g. [1, 7] means shot 1 and shot 7 are the paired composition).
 
3. ONE POLITICAL SHOT, positioned as the final body shot or second-to-last. The single literal image in the piece. It names the room — bureaucratic, institutional, specific — but still without on-screen text. Mark it with is_political_shot: true.
 
VOICEOVER STYLE
- Spare. The body's VO is 40-60 total words across the entire piece. Many shots have NO VO at all (voiceover: null).
- VO is EVIDENCE, not narration. Declarative facts from the article, often as fragments. Never interpretive. "She bought the house ten years ago" is allowed. "She was hopeful" is not. "Nine hundred for the mortgage" is allowed. "Her savings dwindled" is not — let the visual carry it.
- Use the article's own strongest sentences when possible, lightly edited for spoken cadence.
- The first 1-2 shots have NO VO. Music + image carries the opening. VO joins on shot 2 or 3 at the earliest.
- The held empty shot has NO VO.
- The final pre-title VO line is short, lands early in its shot, and leaves held silence after it before the cut to black.
 
TECHNICAL CONSTRAINTS
- Output is 16:9 horizontal, letterboxed inside a 9:16 Reel.
- Generated via Midjourney stills -> Kling 3.0 image-to-video. Every Kling shot has a watermark in the lower-right. DO NOT compose critical subject matter in the lower-right corner.
- 7-10 shots. Each shot 5-10 seconds. Total body 50-65s. Title 3s. CTA 6s. Total runtime 60-75s.
- For each shot, establish_s + held_empty_s must equal duration_s (within 0.1s). The split tells the editor where the read ends and the hold begins. Held_empty_s is at least 2s on every shot, and at least 5s on the held-empty shot.
- Title card is black background, white sans-serif, single line. It IS the thesis. No narration over it. Music drops to silence or a single low note.
- No on-screen text during the body. No subtitles. No chyrons. No kinetic typography. The Kling watermark is the only text in shots 1 through N.
 
OUTPUT FORMAT
Return one JSON object matching this schema exactly. No prose outside the JSON. No code fences.
 
{{
  "story_summary": "1-2 plain-language sentences: what the article is actually about",
  "thesis": "1 sentence: the acid-crisp punchline take. Becomes the title.",
  "title_candidates": ["6-8 candidates, each <= 8 words, dryly ironic, prefer titles with a verb"],
  "music_prompt": "A Suno prompt: genre, instrumentation, mood, tempo, arc. Must support a 60-75s piece with deliberate pauses inside the body (not just at the title), and a cold drop at the title beat. ~40-60 words.",
  "color_script": [
    {{"act": 1, "palette": "single color-script word or phrase", "rationale": "what mood and what narrative beat"}}
  ],
  "repeated_composition_shot_indices": [1, 7],
  "shots": [
    {{
      "index": 1,
      "act": 1,
      "duration_s": 8.0,
      "establish_s": 3.0,
      "held_empty_s": 5.0,
      "midjourney_prompt": "MJ prompt: style, subject, framing, lighting, lens, mood. End with --ar 16:9 --style raw. Compose subject UPPER-LEFT or CENTER, never lower-right.",
      "kling_motion": "what moves: usually locked camera with subtle environmental motion (light shifting, dust, faint breeze). Avoid showy camera moves; ma rewards stillness. One sentence.",
      "voiceover": null,
      "feeling": "single word or short phrase: recognition | dread | claustrophobia | vertigo | resignation | anger | grief | unease",
      "is_held_empty": false,
      "is_political_shot": false,
      "rationale": "what feeling this shot produces and how the held empty earns it"
    }}
  ],
  "title_card": {{
    "appears_after_shot": 8,
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
    establish_s: float
    held_empty_s: float
    midjourney_prompt: str
    kling_motion: str
    voiceover: Optional[str]
    feeling: str
    is_held_empty: bool
    is_political_shot: bool
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
    repeated_composition_shot_indices: list[int]
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
            repeated_composition_shot_indices=data.get("repeated_composition_shot_indices", []),
            shots=[Shot(**s) for s in data["shots"]],
            title_card=TitleCard(**data["title_card"]),
            cta=CTA(**data["cta"]),
        )
 
    def total_runtime_s(self) -> float:
        body = sum(s.duration_s for s in self.shots)
        return body + self.title_card.duration_s + self.cta.duration_s
 
    def total_held_empty_s(self) -> float:
        return sum(s.held_empty_s for s in self.shots)
 
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
 
        if len(self.shots) < 7 or len(self.shots) > 10:
            warnings.append(f"{len(self.shots)} shots outside target 7-10 (ma demands fewer, longer)")
 
        # Per-shot duration: 5-10s
        bad_dur = [s.index for s in self.shots if s.duration_s < 5.0 or s.duration_s > 10.0]
        if bad_dur:
            warnings.append(f"Shots {bad_dur} have durations outside 5-10s")
 
        # establish_s + held_empty_s must equal duration_s
        for s in self.shots:
            if abs((s.establish_s + s.held_empty_s) - s.duration_s) > 0.1:
                warnings.append(
                    f"Shot {s.index}: establish_s ({s.establish_s}) + held_empty_s ({s.held_empty_s}) "
                    f"!= duration_s ({s.duration_s})"
                )
 
        # Every shot needs at least 2s of held empty
        thin = [s.index for s in self.shots if s.held_empty_s < 2.0]
        if thin:
            warnings.append(f"Shots {thin} have held_empty_s < 2.0s — too tight; ma requires the hold")
 
        # Title card timing
        if self.title_card.appears_after_shot < len(self.shots) - 1:
            warnings.append(
                f"Title appears after shot {self.title_card.appears_after_shot} of {len(self.shots)} — "
                f"should appear after the last shot"
            )
 
        # VO budget: 40-60 words
        wc = self.total_vo_word_count()
        if wc < 30 or wc > 70:
            warnings.append(f"VO total {wc} words outside target 40-60")
 
        # First shot has no VO
        if self.shots and self.shots[0].voiceover:
            warnings.append("Shot 1 should have no VO (music + image opens the piece)")
 
        # Mandatory: exactly one held-empty shot, with held_empty_s >= 5
        held_empties = [s for s in self.shots if s.is_held_empty]
        if len(held_empties) == 0:
            warnings.append("No is_held_empty shot present — ma requires one held-empty shot")
        elif len(held_empties) > 1:
            warnings.append(
                f"{len(held_empties)} held-empty shots ({[s.index for s in held_empties]}); "
                f"the held empty has more weight when there is exactly one"
            )
        for s in held_empties:
            if s.held_empty_s < 5.0:
                warnings.append(f"Held-empty shot {s.index} has held_empty_s={s.held_empty_s}s — must be >= 5s")
            if s.voiceover:
                warnings.append(f"Held-empty shot {s.index} has VO — held empties carry no narration")
 
        # Mandatory: at least one political shot, near the end
        political = [s for s in self.shots if s.is_political_shot]
        if not political:
            warnings.append("No is_political_shot present — ma piece needs one specific naming-the-room shot")
        else:
            for s in political:
                # Political shot should sit in the final 1-2 positions before the title
                if s.index < len(self.shots) - 1:
                    warnings.append(
                        f"Political shot {s.index} sits earlier than shot {len(self.shots) - 1} — "
                        f"the political shot lands as the last or second-to-last body shot"
                    )
 
        # Mandatory: a repeated composition pair, both indices valid
        rc = self.repeated_composition_shot_indices
        if len(rc) != 2:
            warnings.append(
                f"repeated_composition_shot_indices must list exactly 2 indices, got {rc}"
            )
        else:
            valid_indices = {s.index for s in self.shots}
            if not all(i in valid_indices for i in rc):
                warnings.append(f"repeated_composition_shot_indices {rc} reference shots that don't exist")
            elif rc[0] >= rc[1]:
                warnings.append(f"repeated_composition_shot_indices {rc} should be in ascending order")
 
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
        f"**Runtime.** {sb.total_runtime_s():.1f}s · "
        f"{len(sb.shots)} shots · {sb.total_vo_word_count()} VO words · "
        f"{sb.total_held_empty_s():.1f}s of held empty\n"
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
 
    if len(sb.repeated_composition_shot_indices) == 2:
        a, b = sb.repeated_composition_shot_indices
        out.append(f"\n**Repeated composition:** shot {a} ↔ shot {b} "
                   f"(generate from same MJ seed; second instance shows subtraction)")
 
    out.append("\n## Shots")
    current_act = 0
    for s in sb.shots:
        if s.act != current_act:
            current_act = s.act
            out.append(f"\n### Act {current_act}")
        markers = []
        if s.is_held_empty:
            markers.append("**HELD EMPTY**")
        if s.is_political_shot:
            markers.append("**POLITICAL**")
        if s.index in sb.repeated_composition_shot_indices:
            markers.append("**ECHO**")
        marker_str = f" — {' · '.join(markers)}" if markers else ""
 
        out.append(f"\n**Shot {s.index}** ({s.duration_s}s "
                   f"= {s.establish_s}s establish + {s.held_empty_s}s held){marker_str}")
        out.append(f"- *Feeling:* {s.feeling}")
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
