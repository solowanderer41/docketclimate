"""
Critic agent for trailer storyboards — the gate between generator and production.

WORKFLOW
The generator (trailer_storyboard.py) produces storyboard v1. This agent acts
as a gate. It reads v1, the article body, and a LeverConfig, and either
approves the storyboard as-is or revises it ONCE. The user receives a single
final storyboard to act on, plus optional critique artifacts for the audit
trail.

OUTPUTS
- <input>_final.json   — the approved or revised storyboard
- <input>_final.md     — markdown brief with a 1-3 line critic summary at top
- <input>_critique.md  — full scorecard (only with --full-critique)

If the revised storyboard fails the generator's structural validator, the
critic's output is rejected and the original v1 is preserved with a warning.
Better to ship a v1 we trust than a v2 that violates ma rules.

DESIGN NOTES
- Single revision pass. If the user wants another, they re-run.
- The critic is permitted to score "approved" and copy v1 verbatim. It is
  NOT permitted to score "revised" without producing a valid v2.
- max_tokens is large (16000) because the response must include a full
  revised storyboard inline.
- Uses the SAME structural rules as the generator. If the generator's rules
  change, this module needs no edit — it imports the dataclass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

from .trailer_storyboard import TrailerStoryboard, render_storyboard_markdown
from .trailer_levers import LeverConfig, PRESETS, build_config


CRITIC_SYSTEM_PROMPT = """You are an editorial critic for a flagship climate-journalism Reel from The Docket. You receive a storyboard JSON, the article body it was generated from, and the editorial levers it should honor. Your job is to score the storyboard against seven dimensions and either approve it as-is or produce ONE revised version.

CRITIQUE DIMENSIONS — score each 1-5

1. THESIS BITE
   Does the thesis line have editorial stance, or is it a balanced summary?
   5 = punchy, opinionated, "of course" framing. "Miami forgot to budget for the wind."
   3 = clear but soft. "Miami's sea breeze is weakening."
   1 = balanced explainer. "A new study shows sea breezes are changing."

2. TITLE CUT
   Does the selected title work as a punchline?
   5 = active verb, ≤8 words, rhythmic, dryly ironic. Lands like a closing line.
   3 = clear and short but descriptive.
   1 = a label, not a punchline. "Sea Breeze Loss in Coastal Cities."

3. METAPHOR ESCALATION
   Does each shot deepen curiosity, or is it a flat list of vignettes?
   5 = chain that builds dread or recognition toward the title. Withholds topic until the late shots.
   3 = competent metaphor but no escalation; viewer "gets it" by shot 4.
   1 = decorative imagery with no through-line.

4. VO DISCIPLINE
   Is the VO evidence-fragmentary, or interpretive narration?
   5 = sparse declarative facts. "Nine hundred for the mortgage." "Three hundred thousand outdoor workers."
   3 = factual but slightly explanatory. "She has been paying nine hundred for her mortgage."
   1 = explainer voice. "What this means is that she's been paying for a house she can't live in."

5. ARTICLE FIDELITY  [JOURNALISM CONSTRAINT — not stylistic]
   Every number, percentage, dollar figure, date, name, and place in the VO appears in the article body.
   5 = all factual claims verbatim from article. Editorial compression allowed.
   3 = one or two compressed claims that are accurate but not verbatim.
   1 = invented statistics, rounded numbers, or fabricated names. Cannot ship.

6. HELD-EMPTY CONVICTION
   Does the is_held_empty shot earn its length, or is it just long?
   5 = the absence is the subject. The viewer searches for movement and finds none, and the searching is the experience.
   3 = a quiet shot that holds, but the negative space isn't doing dramatic work.
   1 = a generic establishing shot tagged as held empty.

7. CLOSING LANDING
   Does the final pre-title shot set up the title's punchline?
   5 = the title feels inevitable. The closing shot creates the question the title answers.
   3 = closing shot is strong but disconnected from the title.
   1 = title arrives out of context with the body's emotional state.

LEVERS THE STORYBOARD WAS TOLD TO HONOR
The storyboard was generated against these dials. Score each dimension partly by whether the storyboard delivered AGAINST these targets:

{LEVERS_BLOCK}

DECISION RULE
- If ALL seven dimensions score 4 or 5 → verdict: "approved". Copy storyboard verbatim into revised_storyboard. summary = "No changes; storyboard ships as-is."
- If ANY dimension scores 3 or below → verdict: "revised". Produce a revised storyboard that addresses the weak dimensions. The revision must:
  * Pass the same structural rules as the generator (7-10 shots; one is_held_empty shot with held_empty_s ≥ 5.0s; one is_political_shot near the end; exactly two repeated_composition_shot_indices; first shot has no VO; total VO 30-70 words; runtime 55-80s).
  * Include EVERY field the generator produces (index, act, duration_s, establish_s, held_empty_s, midjourney_prompt, kling_motion, voiceover, feeling, is_held_empty, is_political_shot, rationale per shot; plus story_summary, thesis, title_candidates, music_prompt, color_script, repeated_composition_shot_indices, title_card, cta at the top level).
  * Keep the article's factual claims verbatim in VO. Do not invent numbers when revising.

REVISION DISCIPLINE
- Revise only what's weak. If shot 3 is fine and shot 7 is the problem, leave shot 3 alone.
- The title_candidates list can be regenerated entirely if title_cut scored low; otherwise leave it.
- Don't change the story or thesis just to make it punchier. The article's actual argument is the constraint.
- If you change the political shot, it must still name the room with one specific image, not become a second metaphor.

OUTPUT
Return one JSON object. No prose outside the JSON. No code fences.

{
  "verdict": "approved" | "revised",
  "summary": "1-3 sentences describing what changed and why. If approved: 'No changes; storyboard ships as-is.'",
  "scorecard": {
    "thesis_bite":            {"score": 4, "note": "one sentence justification"},
    "title_cut":              {"score": 5, "note": "..."},
    "metaphor_escalation":    {"score": 4, "note": "..."},
    "vo_discipline":          {"score": 4, "note": "..."},
    "article_fidelity":       {"score": 5, "note": "..."},
    "held_empty_conviction":  {"score": 4, "note": "..."},
    "closing_landing":        {"score": 4, "note": "..."}
  },
  "weaknesses": [
    {"shot_index": 3, "issue": "what is wrong", "suggestion": "what to do"}
  ],
  "revised_storyboard": { /* full storyboard JSON in the schema from the generator. Required even when verdict is 'approved' — in that case copy v1 verbatim. */ }
}
"""


def critique_and_revise(
    storyboard: dict,
    article_body: str,
    levers: LeverConfig,
    *,
    model: str = "claude-opus-4-7",
    client: Optional[Anthropic] = None,
) -> dict:
    """Run the critic. Returns the parsed JSON response including verdict,
    scorecard, weaknesses, and revised_storyboard."""
    client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    system = CRITIC_SYSTEM_PROMPT.replace("{LEVERS_BLOCK}", levers.describe())

    user_prompt = (
        f"ARTICLE BODY:\n{article_body}\n\n"
        f"STORYBOARD V1:\n{json.dumps(storyboard, indent=2)}\n\n"
        "Critique and revise. Return one JSON object only, no prose outside, no code fences."
    )

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_summary_banner(critique: dict, levers: LeverConfig) -> str:
    """Three-line banner shown at the top of the final markdown brief."""
    verdict = critique["verdict"]
    summary = critique["summary"]

    # Mini-scorecard inline: dimensions below 4 get called out
    weak = []
    for dim, entry in critique["scorecard"].items():
        if entry["score"] < 4:
            weak.append(f"{dim.replace('_', ' ')} ({entry['score']}/5)")
    weak_str = f" Weak: {', '.join(weak)}." if weak else ""

    return (
        f"**Critic — {verdict.upper()}.** {summary}{weak_str}\n\n"
        f"*Lever preset reflected: vo_word_target={levers.vo_word_target}, "
        f"hook_style={levers.hook_style}, title_placement={levers.title_placement}, "
        f"metaphor_density={levers.metaphor_density}, pace={levers.pace}.*\n"
    )


def render_critique_markdown(critique: dict, levers: LeverConfig) -> str:
    """Full critique scorecard — only written when --full-critique is set."""
    lines = []
    lines.append("# Storyboard critique\n")
    lines.append(f"**Verdict.** {critique['verdict']}\n")
    lines.append(f"**Summary.** {critique['summary']}\n")

    lines.append("\n## Levers used")
    lines.append("```")
    lines.append(levers.describe().strip())
    lines.append("```")

    lines.append("\n## Scorecard")
    for dim, entry in critique["scorecard"].items():
        score = entry["score"]
        note = entry["note"]
        bar = "★" * score + "·" * (5 - score)
        pretty = dim.replace("_", " ").title()
        lines.append(f"- **{pretty}** `{bar}` ({score}/5) — {note}")

    weaknesses = critique.get("weaknesses") or []
    if weaknesses:
        lines.append("\n## Shot-level notes")
        for w in weaknesses:
            lines.append(f"- **Shot {w['shot_index']}** — {w['issue']}")
            lines.append(f"  - *Suggestion:* {w['suggestion']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def run(
    storyboard_path: Path,
    article_body_path: Path,
    preset: str,
    overrides: str,
    full_critique: bool,
    model: str,
) -> int:
    if not storyboard_path.exists():
        print(f"Storyboard not found: {storyboard_path}", file=sys.stderr)
        return 1
    if not article_body_path.exists():
        print(f"Article body not found: {article_body_path}", file=sys.stderr)
        return 1

    storyboard = json.loads(storyboard_path.read_text())
    article_body = article_body_path.read_text()
    levers = build_config(preset, overrides)

    print(f"Running critic with preset '{preset}'"
          + (f" + overrides '{overrides}'" if overrides else "")
          + f" against {storyboard_path.name}...", file=sys.stderr)

    try:
        critique = critique_and_revise(storyboard, article_body, levers, model=model)
    except json.JSONDecodeError as e:
        print(f"Critic returned malformed JSON: {e}", file=sys.stderr)
        return 2

    # Validate revised storyboard against generator's structural rules
    revised_dict = critique.get("revised_storyboard")
    if not revised_dict:
        print("Critic output missing revised_storyboard. Aborting.", file=sys.stderr)
        return 2

    try:
        revised_sb = TrailerStoryboard.from_json(revised_dict)
    except (KeyError, TypeError) as e:
        # Revision is malformed — fall back to v1
        print(f"⚠ Revised storyboard failed schema validation: {e}", file=sys.stderr)
        print("  Falling back to storyboard v1. Critic output saved for inspection.", file=sys.stderr)
        err_path = storyboard_path.with_name(storyboard_path.stem + "_critic_error.json")
        err_path.write_text(json.dumps(critique, indent=2))
        revised_sb = TrailerStoryboard.from_json(storyboard)
        critique["verdict"] = "fallback_to_v1"
        critique["summary"] = (
            f"Critic produced an invalid revision; shipping v1 unchanged. "
            f"See {err_path.name} for the malformed output."
        )

    validation_warnings = revised_sb.validate()

    # Write outputs next to input
    stem = storyboard_path.stem
    if stem.endswith("_v1"):
        stem = stem[:-3]
    final_json = storyboard_path.with_name(f"{stem}_final.json")
    final_md = storyboard_path.with_name(f"{stem}_final.md")

    final_json.write_text(json.dumps(revised_sb.to_dict(), indent=2))

    banner = render_summary_banner(critique, levers)
    brief = banner + "\n---\n\n" + render_storyboard_markdown(revised_sb)
    final_md.write_text(brief)

    if full_critique:
        critique_md = storyboard_path.with_name(f"{stem}_critique.md")
        critique_md.write_text(render_critique_markdown(critique, levers))
        print(f"  Full critique:  {critique_md}", file=sys.stderr)

    # Console summary
    print(f"\nVerdict: {critique['verdict']}", file=sys.stderr)
    print(f"Summary: {critique['summary']}", file=sys.stderr)
    print("\nScorecard:", file=sys.stderr)
    for dim, entry in critique["scorecard"].items():
        bar = "★" * entry["score"] + "·" * (5 - entry["score"])
        print(f"  {dim:<24} {bar} ({entry['score']}/5)", file=sys.stderr)

    if validation_warnings:
        print("\n⚠ Structural warnings on shipped storyboard:", file=sys.stderr)
        for w in validation_warnings:
            print(f"  - {w}", file=sys.stderr)

    print(f"\nFinal storyboard: {final_json}", file=sys.stderr)
    print(f"Brief:            {final_md}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Critic gate for trailer storyboards. Produces one final storyboard, "
                    "either approved as-is or revised once."
    )
    parser.add_argument("storyboard", type=Path,
                        help="Path to storyboard JSON (output of trailer_storyboard.py --json)")
    parser.add_argument("--article-body", type=Path, required=True,
                        help="Path to the article body text file the storyboard was generated from")
    parser.add_argument("--preset", default="standard", choices=sorted(PRESETS),
                        help="Lever preset (default: standard)")
    parser.add_argument("--levers", default="",
                        help="Comma-separated overrides on top of the preset, e.g. "
                             "'vo_word_target=40,pace=cut_driven'")
    parser.add_argument("--full-critique", action="store_true",
                        help="Also write a <stem>_critique.md with the full scorecard")
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Model to use (default: claude-opus-4-7)")
    args = parser.parse_args()

    sys.exit(run(
        args.storyboard,
        args.article_body,
        args.preset,
        args.levers,
        args.full_critique,
        args.model,
    ))


if __name__ == "__main__":
    main()
