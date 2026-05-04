"""
Manual Midjourney handoff CLI for the trailer pipeline.

The storyboard generator produces 7-10 shots, each with an MJ prompt. This
module walks you through generating those stills by hand in Discord, capturing
the chosen image filename for each shot, and persisting the result so the next
stage (Kling i2v) can run automatically.

USAGE
    python -m src.video.trailer_mj_handoff /path/to/storyboard.json

PRECONDITIONS
- A storyboard JSON saved by trailer_storyboard.py with --json
- A working directory where you'll save MJ stills (default: ./mj_stills/)

WORKFLOW PER SHOT
1. CLI prints the MJ prompt and copies it to your clipboard (macOS: pbcopy)
2. You paste into Midjourney Discord, run /imagine, generate variants
3. Pick the strongest image; download it
4. Drop it into ./mj_stills/ with any filename
5. Type 'd' (done) at the prompt; CLI lists new files in mj_stills/
6. You pick one by number; CLI renames it to shot_<index>.png and records the mapping
7. Continue to next shot

OUTPUT
A storyboard_with_stills.json next to the input file, augmented with a
'mj_still_path' field on each shot. That file is the input to the Kling client.

DESIGN NOTES
- The repeated-composition pair (e.g. shots 1 and 7) reuses the same MJ seed.
  When you reach the second shot of the pair, the CLI reminds you of the
  first shot's seed and prompt so you can regenerate with --seed <N> and
  swap in the subtraction phrase. You're trusted to do the seed math; this
  isn't ApiFrame, it's a notebook.
- Skipping a shot is allowed (mark with 's'); you can finish later. The
  --resume flag picks up where you left off.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard copy. Returns True on success, False if no
    clipboard tool is available (Linux without xclip, etc.)."""
    candidates = []
    if sys.platform == "darwin":
        candidates.append(["pbcopy"])
    elif sys.platform == "win32":
        candidates.append(["clip"])
    else:
        candidates.append(["xclip", "-selection", "clipboard"])
        candidates.append(["wl-copy"])
        candidates.append(["xsel", "--clipboard", "--input"])

    for cmd in candidates:
        try:
            p = subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=2)
            if p.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return False


def list_image_files(d: Path) -> list[Path]:
    """Return image files in d, sorted by mtime descending (newest first)."""
    if not d.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def prompt_user(msg: str) -> str:
    """Prompt with a clear cue. Strips whitespace from response."""
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(130)


# ---------------------------------------------------------------------------
# Per-shot interactive loop
# ---------------------------------------------------------------------------

@dataclass
class HandoffState:
    """State held across the whole interactive session."""
    storyboard: dict
    storyboard_path: Path
    stills_dir: Path
    output_path: Path
    # We track filenames already claimed by a shot so 'list' shows only new ones
    claimed: set[str] = field(default_factory=set)

    def save(self):
        """Write the augmented storyboard atomically."""
        tmp = self.output_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.storyboard, f, indent=2)
        tmp.replace(self.output_path)


def handle_shot(state: HandoffState, shot: dict, total: int) -> str:
    """Walk the user through one shot. Returns one of: 'next', 'skip', 'quit'.

    Mutates shot in-place by setting shot['mj_still_path'] on success.
    """
    idx = shot["index"]
    prompt = shot["midjourney_prompt"]
    motion = shot["kling_motion"]

    # Repeated-composition reminder
    repeat_pair = state.storyboard.get("repeated_composition_shot_indices", [])
    is_second_of_pair = (
        len(repeat_pair) == 2
        and idx == repeat_pair[1]
    )
    paired_first_shot = None
    if is_second_of_pair:
        for s in state.storyboard["shots"]:
            if s["index"] == repeat_pair[0]:
                paired_first_shot = s
                break

    # Markers
    markers = []
    if shot.get("is_held_empty"):
        markers.append("HELD EMPTY")
    if shot.get("is_political_shot"):
        markers.append("POLITICAL")
    if idx in repeat_pair:
        which = "first" if idx == repeat_pair[0] else "second"
        markers.append(f"ECHO ({which} of pair)")

    print()
    print("=" * 70)
    header = f"SHOT {idx} of {total}"
    if markers:
        header += "  —  " + " · ".join(markers)
    print(header)
    print("=" * 70)
    print(f"Duration: {shot['duration_s']}s "
          f"({shot.get('establish_s', '?')}s establish + "
          f"{shot.get('held_empty_s', '?')}s held)")
    print(f"Feeling : {shot.get('feeling', '—')}")
    if shot.get("voiceover"):
        print(f"VO      : \"{shot['voiceover']}\"")
    else:
        print("VO      : —")
    print()
    print("MJ PROMPT (copied to clipboard):")
    print(f"  {prompt}")
    print()
    print(f"Kling motion to remember for i2v: {motion}")

    if is_second_of_pair and paired_first_shot:
        first_still = paired_first_shot.get("mj_still_path")
        print()
        print(f"  ⤷ ECHO PAIR: this shot mirrors shot {paired_first_shot['index']}.")
        if first_still:
            print(f"     First instance was saved as: {first_still}")
            print(f"     In Discord, regenerate using --seed from that image (right-click → Show Seed)")
            print(f"     or use /describe on the saved image to recover MJ params.")
        else:
            print(f"     (Shot {paired_first_shot['index']} hasn't been generated yet — "
                  f"you may want to skip and come back to keep the pair tight.)")

    copied = copy_to_clipboard(prompt)
    if not copied:
        print()
        print("  (Could not copy to clipboard automatically — copy the prompt manually.)")

    print()
    print("Now: paste the prompt into Midjourney, generate, pick your favorite,")
    print(f"     and download the image into:  {state.stills_dir}/")
    print()

    while True:
        choice = prompt_user(
            "  [d] done — list new files     "
            "[s] skip this shot     "
            "[r] re-print prompt     "
            "[q] quit and save\n"
            "  > "
        )

        if choice == "q":
            return "quit"
        if choice == "s":
            print("  Skipped. Will need to revisit before Kling stage.")
            return "skip"
        if choice == "r":
            print()
            print(prompt)
            print()
            copy_to_clipboard(prompt)
            continue
        if choice == "d":
            files = list_image_files(state.stills_dir)
            unclaimed = [f for f in files if f.name not in state.claimed]
            if not unclaimed:
                print(f"  No new images in {state.stills_dir}/. Add one and try again.")
                continue

            print()
            print("  New images (newest first):")
            for i, f in enumerate(unclaimed[:10], 1):
                size_kb = f.stat().st_size // 1024
                print(f"    [{i}] {f.name}  ({size_kb} KB)")
            print()

            pick = prompt_user("  Pick number (or 'b' to go back): ")
            if pick == "b":
                continue
            try:
                n = int(pick)
                if not (1 <= n <= len(unclaimed[:10])):
                    print("  Out of range.")
                    continue
            except ValueError:
                print("  Not a number.")
                continue

            chosen = unclaimed[n - 1]
            target = state.stills_dir / f"shot_{idx:02d}{chosen.suffix.lower()}"
            # If a same-named file already exists from a previous run, archive it
            if target.exists() and target != chosen:
                archive = state.stills_dir / f"shot_{idx:02d}.prev{chosen.suffix.lower()}"
                target.rename(archive)
                print(f"  Existing {target.name} archived as {archive.name}")
            if target != chosen:
                shutil.copy2(chosen, target)
            shot["mj_still_path"] = str(target.resolve())
            state.claimed.add(chosen.name)
            state.claimed.add(target.name)
            state.save()
            print(f"  ✓ Saved as {target.name}; storyboard updated.")
            return "next"

        print("  Unrecognized. Type one letter: d, s, r, or q.")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def run(storyboard_path: Path, stills_dir: Path, resume: bool) -> int:
    if not storyboard_path.exists():
        print(f"Storyboard not found: {storyboard_path}", file=sys.stderr)
        return 1

    with open(storyboard_path) as f:
        sb = json.load(f)

    output_path = storyboard_path.with_name(
        storyboard_path.stem + "_with_stills.json"
    )
    if resume and output_path.exists():
        print(f"Resuming from {output_path.name}")
        with open(output_path) as f:
            sb = json.load(f)

    stills_dir.mkdir(parents=True, exist_ok=True)

    state = HandoffState(
        storyboard=sb,
        storyboard_path=storyboard_path,
        stills_dir=stills_dir,
        output_path=output_path,
    )

    # Pre-populate claimed with any shot_NN.* that already match resumed shots
    for s in sb["shots"]:
        if s.get("mj_still_path"):
            state.claimed.add(Path(s["mj_still_path"]).name)

    shots = sb["shots"]
    total = len(shots)

    print()
    print(f"Trailer MJ handoff — {sb.get('thesis', '(no thesis)')}")
    print(f"  storyboard : {storyboard_path}")
    print(f"  stills dir : {stills_dir}")
    print(f"  output     : {output_path}")
    print(f"  shots      : {total}")
    print()
    repeat = sb.get("repeated_composition_shot_indices", [])
    if len(repeat) == 2:
        print(f"  echo pair  : shots {repeat[0]} ↔ {repeat[1]} "
              f"(use same MJ seed; second shows subtraction)")
        print()

    completed = 0
    skipped = []
    for shot in shots:
        if shot.get("mj_still_path") and resume:
            print(f"Shot {shot['index']}: already done → "
                  f"{Path(shot['mj_still_path']).name}")
            completed += 1
            continue

        result = handle_shot(state, shot, total)
        if result == "quit":
            print()
            print(f"Saved progress to {output_path}. Resume with --resume.")
            return 0
        if result == "skip":
            skipped.append(shot["index"])
        else:
            completed += 1

    print()
    print("=" * 70)
    print(f"Done. {completed}/{total} shots have stills.")
    if skipped:
        print(f"Skipped: {skipped}. Re-run with --resume to finish.")
    print(f"Augmented storyboard saved to: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Walk through manual Midjourney still generation for a trailer storyboard."
    )
    parser.add_argument("storyboard", type=Path,
                        help="Path to storyboard JSON (output of --json from trailer_storyboard.py)")
    parser.add_argument("--stills-dir", type=Path, default=Path("./mj_stills"),
                        help="Directory where you'll drop downloaded MJ images (default: ./mj_stills)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip shots that already have mj_still_path set")
    args = parser.parse_args()

    sys.exit(run(args.storyboard, args.stills_dir, args.resume))


if __name__ == "__main__":
    main()
