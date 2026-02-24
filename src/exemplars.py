"""
Exemplar management for The Docket.

Stores flagged "good work" — text posts with strong hooks and video scripts
with great pacing — as exemplars that accumulate over time. Phase 2 will
inject these into LLM prompts as few-shot examples to steer content quality
in a positive direction.

Two exemplar types:
    - good_copy: Text posts with strong hooks/framing (any platform)
    - good_video: Reels with good slide pacing, hooks, or visual treatment
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_EXEMPLARS_PATH = PROJECT_ROOT / "data" / "exemplars" / "exemplars.json"


@dataclass
class Exemplar:
    """A flagged piece of good work."""
    id: str                         # "ex_{queue_item_id}"
    exemplar_type: str              # "good_copy" or "good_video"
    queue_item_id: str              # links back to original QueueItem
    issue_number: int | None
    platform: str
    article_title: str
    section: str
    is_feature: bool
    text: str                       # full post text
    video_script: dict | None       # full video_script dict (for good_video)
    flagged_at: str                 # ISO datetime
    engagement_score: float = 0.0   # from metrics if available, else 0
    notes: str = ""                 # optional user annotation


def create_manual_exemplar(
    platform: str,
    section: str,
    article_title: str,
    text: str,
    exemplar_type: str = "good_copy",
    notes: str = "",
) -> Exemplar:
    """Create an exemplar from scratch — not linked to any queue item.

    Manual exemplars are gold-standard references written by the user.
    They receive an engagement_score of 100.0 so they rank highest
    when injected into LLM prompts.

    Args:
        platform: Target platform (e.g. "twitter", "bluesky", "threads").
        section: Section ID (e.g. "lived", "systems", "science").
        article_title: Article or topic this exemplar is written for.
        text: Full post text.
        exemplar_type: "good_copy" (default) or "good_video".
        notes: Optional annotation about why this is good.

    Returns:
        A new Exemplar instance with a synthetic ID.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Exemplar(
        id=f"ex_manual_{ts}",
        exemplar_type=exemplar_type,
        queue_item_id=f"manual_{ts}",
        issue_number=None,
        platform=platform,
        article_title=article_title,
        section=section,
        is_feature=True,
        text=text,
        video_script=None,
        flagged_at=datetime.now().isoformat(),
        engagement_score=100.0,
        notes=notes or "Manual exemplar",
    )


def flag_from_queue_item(
    item,
    exemplar_type: str,
    issue_number: int | None = None,
    engagement_score: float = 0.0,
    notes: str = "",
) -> Exemplar:
    """Create an Exemplar from a QueueItem.

    Args:
        item: A QueueItem (from scheduler.py) — duck-typed to avoid circular imports.
        exemplar_type: "good_copy" or "good_video".
        issue_number: Issue number (extracted from queue metadata).
        engagement_score: Engagement score from metrics, if available.
        notes: Optional user annotation.

    Returns:
        A new Exemplar instance.
    """
    return Exemplar(
        id=f"ex_{item.id}",
        exemplar_type=exemplar_type,
        queue_item_id=item.id,
        issue_number=issue_number,
        platform=item.platform,
        article_title=item.article_title,
        section=item.section,
        is_feature=item.is_feature,
        text=item.text,
        video_script=item.video_script if exemplar_type == "good_video" else None,
        flagged_at=datetime.now().isoformat(),
        engagement_score=engagement_score,
        notes=notes,
    )


def save_exemplar(
    exemplar: Exemplar,
    path: Path = DEFAULT_EXEMPLARS_PATH,
) -> None:
    """Append an exemplar to the cumulative JSON store.

    Deduplicates by queue_item_id — flagging the same item twice
    updates the existing entry (preserving the latest notes/score).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_exemplars(path)
    existing_map = {e.queue_item_id: e for e in existing}

    if exemplar.queue_item_id in existing_map:
        # Update existing entry
        existing_map[exemplar.queue_item_id] = exemplar
        all_exemplars = list(existing_map.values())
        console.print(f"[dim]Updated existing exemplar for {exemplar.queue_item_id}[/dim]")
    else:
        all_exemplars = existing + [exemplar]
        console.print(
            f"[green]Saved new exemplar: {exemplar.exemplar_type} — "
            f"{exemplar.article_title[:40]}[/green]"
        )

    with open(path, "w") as f:
        json.dump([asdict(e) for e in all_exemplars], f, indent=2)

    total = len(all_exemplars)
    copies = sum(1 for e in all_exemplars if e.exemplar_type == "good_copy")
    videos = sum(1 for e in all_exemplars if e.exemplar_type == "good_video")
    console.print(f"[dim]Exemplar store: {total} total ({copies} copy, {videos} video)[/dim]")


def load_exemplars(path: Path = DEFAULT_EXEMPLARS_PATH) -> list[Exemplar]:
    """Load all exemplars from the cumulative JSON file."""
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        return [Exemplar(**d) for d in data]
    except (json.JSONDecodeError, TypeError):
        return []


def get_exemplars_by_type(
    exemplar_type: str,
    path: Path = DEFAULT_EXEMPLARS_PATH,
) -> list[Exemplar]:
    """Filter exemplars by type ('good_copy' or 'good_video')."""
    return [e for e in load_exemplars(path) if e.exemplar_type == exemplar_type]


def get_exemplars_for_prompt(
    exemplar_type: str,
    section: str | None = None,
    limit: int = 3,
    path: Path = DEFAULT_EXEMPLARS_PATH,
) -> list[Exemplar]:
    """Get the best exemplars for LLM prompt injection.

    Returns exemplars sorted by engagement_score (highest first),
    optionally filtered by section. Prioritizes section matches
    but fills remaining slots from other sections.

    Args:
        exemplar_type: "good_copy" or "good_video".
        section: Prefer exemplars from this section (e.g. "lived").
        limit: Maximum number of exemplars to return.
        path: Path to exemplars JSON file.

    Returns:
        Up to `limit` Exemplar objects, best first.
    """
    all_typed = get_exemplars_by_type(exemplar_type, path)

    if not all_typed:
        return []

    if section:
        # Prioritize section matches, fill with others
        section_matches = sorted(
            [e for e in all_typed if e.section == section],
            key=lambda e: e.engagement_score,
            reverse=True,
        )
        others = sorted(
            [e for e in all_typed if e.section != section],
            key=lambda e: e.engagement_score,
            reverse=True,
        )
        combined = section_matches + others
        return combined[:limit]

    return sorted(all_typed, key=lambda e: e.engagement_score, reverse=True)[:limit]


def print_exemplars_table(exemplars: list[Exemplar]) -> None:
    """Display a Rich table of exemplars."""
    if not exemplars:
        console.print("[dim]No exemplars flagged yet.[/dim]")
        return

    table = Table(title="Flagged Exemplars", show_lines=False)
    table.add_column("Type", width=8)
    table.add_column("Article", width=40)
    table.add_column("Section", width=10)
    table.add_column("Platform", width=10)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Flagged", width=12)
    table.add_column("Notes", width=30)

    for e in exemplars:
        type_icon = "\U0001f3ac" if e.exemplar_type == "good_video" else "\U0001f4dd"
        score_str = f"{e.engagement_score:.0f}" if e.engagement_score else "-"

        # Format date
        try:
            dt = datetime.fromisoformat(e.flagged_at)
            date_str = dt.strftime("%b %d")
        except (ValueError, TypeError):
            date_str = "-"

        table.add_row(
            type_icon,
            e.article_title[:38] + (".." if len(e.article_title) > 38 else ""),
            e.section,
            e.platform,
            score_str,
            date_str,
            (e.notes[:28] + ".." if len(e.notes) > 28 else e.notes) or "-",
        )

    console.print(table)


def print_exemplar_detail(exemplar: Exemplar) -> None:
    """Display a single exemplar in detail."""
    type_label = "Good Video" if exemplar.exemplar_type == "good_video" else "Good Copy"

    content_lines = [
        f"[bold]{type_label}[/bold]: {exemplar.article_title}",
        f"Section: {exemplar.section} | Platform: {exemplar.platform}",
        f"Issue: {exemplar.issue_number or '-'} | Engagement: {exemplar.engagement_score:.0f}",
    ]

    if exemplar.notes:
        content_lines.append(f"Notes: [italic]{exemplar.notes}[/italic]")

    content_lines.append("")

    if exemplar.exemplar_type == "good_video" and exemplar.video_script:
        vs = exemplar.video_script
        content_lines.append(f"[cyan]Hook:[/cyan] {vs.get('hook', '-')}")
        for i, slide in enumerate(vs.get("body_slides", []), 1):
            content_lines.append(f"[cyan]Slide {i}:[/cyan] {slide}")
        tier = vs.get("video_tier", "narrative")
        score = vs.get("cinematic_score", "-")
        content_lines.append(f"\n[dim]Tier: {tier} | Cinematic: {score}/10[/dim]")
    else:
        # Show full post text
        content_lines.append(exemplar.text)

    console.print(Panel(
        "\n".join(content_lines),
        border_style="green",
        title=f"[bold]Exemplar: {exemplar.id}[/bold]",
    ))
