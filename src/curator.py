"""
Interactive content curator for The Docket.

Presents generated content in a Rich terminal session for human review.
Users can approve, skip, edit copy, and toggle video for feature articles.
News cards get bulk selection.

This creates a feedback loop — the user shapes article selection
and copy quality before the queue gets built.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text

from src.content_generator import (
    TextPost, VideoScript, _score_article, _build_unseen_reality_hook,
    _build_post_text, _generate_body_slides, _build_voiceover,
    SECTION_HASHTAGS, ISSUE_URL, PLATFORM_LIMITS,
)
from src.scraper import Article

console = Console()


@dataclass
class CuratedArticle:
    """An article approved for promotion with (possibly edited) copy."""
    title: str
    section_id: str
    url: str
    is_feature: bool
    hook: str
    score: float
    for_video: bool = False
    # Original hook preserved for diffing
    original_hook: str = ""
    # Video script data (populated when for_video=True)
    body_slides: list[str] = field(default_factory=list)
    voiceover_text: str = ""
    is_external: bool = False
    # AI image generation fields (Fix O)
    cinematic_score: int = 0
    video_tier: str = "narrative"          # "cinematic" or "narrative"
    image_prompts: list[str] = field(default_factory=list)
    background_prompt: str = ""


@dataclass
class CurationResult:
    """Output of the curation process."""
    issue_number: int | None
    selected_features: list[CuratedArticle] = field(default_factory=list)
    selected_news: list[CuratedArticle] = field(default_factory=list)

    @property
    def video_features(self) -> list[CuratedArticle]:
        return [f for f in self.selected_features if f.for_video]

    @property
    def all_selected(self) -> list[CuratedArticle]:
        return self.selected_features + self.selected_news

    def to_dict(self) -> dict:
        return {
            "issue_number": self.issue_number,
            "selected_features": [asdict(a) for a in self.selected_features],
            "selected_news": [asdict(a) for a in self.selected_news],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Curation saved to {path}[/green]")

    @classmethod
    def load(cls, path: Path) -> "CurationResult":
        with open(path) as f:
            data = json.load(f)
        result = cls(issue_number=data.get("issue_number"))
        for a in data.get("selected_features", []):
            result.selected_features.append(CuratedArticle(**a))
        for a in data.get("selected_news", []):
            result.selected_news.append(CuratedArticle(**a))
        return result


def _edit_in_editor(text: str, label: str = "hook") -> str:
    """Open text in $EDITOR for multi-line editing. Falls back to inline input."""
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", ""))
    if not editor:
        # No editor configured — fall back to inline input
        console.print(f"[dim]Enter new {label} (or press Enter to keep current):[/dim]")
        new = input("> ").strip()
        return new if new else text

    try:
        suffix = ".txt"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, prefix=f"docket_{label}_", delete=False
        ) as f:
            f.write(text)
            path = f.name

        console.print(f"[dim]Opening {label} in {editor}... save and close to continue.[/dim]")
        subprocess.call([editor, path])

        with open(path) as f:
            result = f.read().strip()
        os.unlink(path)

        if result and result != text:
            return result
        console.print(f"[dim]No changes detected — kept original {label}.[/dim]")
        return text
    except Exception as e:
        console.print(f"[red]Editor failed ({e}). Falling back to inline input.[/red]")
        new = input("> ").strip()
        return new if new else text


def curate_content(
    issue,
    text_posts: list[TextPost],
    video_scripts: list[VideoScript],
    config: dict,
    auto: bool = False,
) -> CurationResult:
    """
    Main curation entry point.

    If auto=True, selects top-scored articles without interaction.
    Otherwise, launches interactive Rich terminal session.
    """
    features = [a for a in issue.articles if a.is_feature]
    news = [a for a in issue.articles if not a.is_feature and a.url
            and getattr(a, "is_external", False)]

    schedule_config = config.get("schedule", {})
    cold_start = config.get("cold_start", {})
    is_cold_start = cold_start.get("enabled", False)

    if is_cold_start:
        features_per_day = cold_start.get("features_per_day", 1)
        news_per_day = cold_start.get("news_per_day", 1)
        videos_per_week = cold_start.get("videos_per_week", 2)
        console.print("[yellow]Cold-start mode active — reduced volume[/yellow]")
    else:
        features_per_day = schedule_config.get("features_per_day", 3)
        news_per_day = schedule_config.get("news_per_day", 4)
        videos_per_week = schedule_config.get("videos_per_week", 5)
    weekdays = len(schedule_config.get("weekdays", ["mon", "tue", "wed", "thu", "fri"]))
    total_features = features_per_day * weekdays
    total_news = news_per_day * weekdays

    # Build hook map from existing text posts (one hook per article title)
    hook_map = {}
    for post in text_posts:
        if post.article_title not in hook_map:
            # Extract hook from post text (everything before the URL)
            lines = post.text.split("\n\n")
            hook_lines = []
            for line in lines:
                if line.startswith("http") or line.startswith("#"):
                    break
                hook_lines.append(line)
            hook_map[post.article_title] = "\n\n".join(hook_lines)

    # Score and sort
    features_scored = sorted(features, key=_score_article, reverse=True)
    news_scored = sorted(news, key=_score_article, reverse=True)

    if auto:
        return _auto_curate(
            issue, features_scored, news_scored, hook_map,
            video_scripts, total_features, total_news, videos_per_week
        )

    return _interactive_curate(
        issue, features_scored, news_scored, hook_map,
        video_scripts, total_features, total_news, videos_per_week
    )


def _auto_curate(
    issue, features, news, hook_map,
    video_scripts, max_features, max_news, max_videos,
) -> CurationResult:
    """Auto-select top articles by score."""
    result = CurationResult(issue_number=issue.issue_number)

    # Build lookup: article title → VideoScript (for body_slides + voiceover_text)
    vs_map = {vs.article_title: vs for vs in video_scripts} if video_scripts else {}

    for i, article in enumerate(features[:max_features]):
        hook = hook_map.get(article.title, article.title)
        is_video = i < max_videos
        vs = vs_map.get(article.title)

        result.selected_features.append(CuratedArticle(
            title=article.title,
            section_id=article.section_id,
            url=article.url or "",
            is_feature=True,
            hook=hook,
            original_hook=hook,
            score=_score_article(article),
            for_video=is_video,
            body_slides=vs.body_slides if (is_video and vs) else [],
            voiceover_text=vs.voiceover_text if (is_video and vs) else "",
            is_external=getattr(article, "is_external", False),
            cinematic_score=vs.cinematic_score if vs else 0,
            video_tier=vs.video_tier if vs else "narrative",
            image_prompts=vs.image_prompts if (is_video and vs) else [],
            background_prompt=vs.background_prompt if (is_video and vs) else "",
        ))

    for article in news[:max_news]:
        hook = hook_map.get(article.title, article.title)
        result.selected_news.append(CuratedArticle(
            title=article.title,
            section_id=article.section_id,
            url=article.url or "",
            is_feature=False,
            hook=hook,
            original_hook=hook,
            score=_score_article(article),
            is_external=getattr(article, "is_external", False),
        ))

    console.print(
        f"[bold green]Auto-selected {len(result.selected_features)} features "
        f"({len(result.video_features)} for video) + "
        f"{len(result.selected_news)} news[/bold green]"
    )
    return result


def _interactive_curate(
    issue, features, news, hook_map,
    video_scripts, max_features, max_news, max_videos,
) -> CurationResult:
    """Interactive Rich terminal curation session."""
    result = CurationResult(issue_number=issue.issue_number)

    # Load engagement analytics (if available)
    section_avgs = {}
    try:
        from src.analytics import get_section_averages
        section_avgs = get_section_averages()
    except Exception:
        pass

    # --- Phase 1: Feature articles ---
    console.print(Panel.fit(
        f"[bold cyan]Feature Article Review[/bold cyan]\n"
        f"[dim]{len(features)} features available | Target: ~{max_features} for the week[/dim]\n\n"
        f"[bold]y[/bold] = approve  [bold]n[/bold] = skip  "
        f"[bold]e[/bold] = edit hook  [bold]v[/bold] = approve + video  "
        f"[bold]Enter[/bold] = approve",
        border_style="cyan",
    ))

    # Build lookup: article title → VideoScript (for body_slides + voiceover_text)
    vs_map = {vs.article_title: vs for vs in video_scripts} if video_scripts else {}

    # Group by section for variety
    sections_seen = {}
    for article in features:
        sid = article.section_id
        if sid not in sections_seen:
            sections_seen[sid] = []
        sections_seen[sid].append(article)

    video_count = 0
    article_index = 0

    for article in features:
        article_index += 1
        score = _score_article(article)
        hook = hook_map.get(article.title, article.title)

        # Truncate hook for display
        hook_preview = hook[:200] + ("..." if len(hook) > 200 else "")

        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(
            f"[bold]{article_index}/{len(features)}[/bold] "
            f"[yellow]{article.title}[/yellow]  "
            f"[dim]({article.section_id} | score: {score:.1f})[/dim]"
        )
        console.print(f"[dim]URL: {article.url or 'none'}[/dim]")
        # Show cinematic scoring from video script
        vs = vs_map.get(article.title)
        if vs and vs.cinematic_score > 0:
            tier_label = vs.video_tier.upper()
            tier_color = "bold green" if vs.video_tier == "cinematic" else "dim"
            console.print(
                f"[{tier_color}]Cinematic: {vs.cinematic_score}/10 → {tier_label}[/{tier_color}]"
            )
        if section_avgs and article.section_id in section_avgs:
            avg = section_avgs[article.section_id]
            max_avg = max(section_avgs.values()) or 1
            bar_len = int((avg / max_avg) * 10)
            bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
            console.print(
                f"[dim]Section engagement: {bar} {avg:.1f} avg[/dim]"
            )
        console.print(f"\n{hook_preview}")

        while True:
            choice = Prompt.ask(
                "\n[bold]Action[/bold]",
                choices=["y", "n", "e", "v", ""],
                default="",
                show_choices=False,
            )

            if choice in ("y", ""):
                result.selected_features.append(CuratedArticle(
                    title=article.title,
                    section_id=article.section_id,
                    url=article.url or "",
                    is_feature=True,
                    hook=hook,
                    original_hook=hook,
                    score=score,
                    is_external=getattr(article, "is_external", False),
                ))
                console.print("[green]Approved[/green]")
                break

            elif choice == "v":
                video_count += 1
                vs = vs_map.get(article.title)
                tier_label = vs.video_tier.upper() if vs else "NARRATIVE"
                cine_score = vs.cinematic_score if vs else 0
                result.selected_features.append(CuratedArticle(
                    title=article.title,
                    section_id=article.section_id,
                    url=article.url or "",
                    is_feature=True,
                    hook=hook,
                    original_hook=hook,
                    score=score,
                    for_video=True,
                    body_slides=vs.body_slides if vs else [],
                    voiceover_text=vs.voiceover_text if vs else "",
                    is_external=getattr(article, "is_external", False),
                    cinematic_score=vs.cinematic_score if vs else 0,
                    video_tier=vs.video_tier if vs else "narrative",
                    image_prompts=vs.image_prompts if vs else [],
                    background_prompt=vs.background_prompt if vs else "",
                ))
                console.print(
                    f"[green]Approved + Video ({video_count} videos) "
                    f"[cinematic: {cine_score}/10 → {tier_label}][/green]"
                )
                break

            elif choice == "n":
                console.print("[dim]Skipped[/dim]")
                break

            elif choice == "e":
                new_hook = _edit_in_editor(hook, label="hook")
                if new_hook != hook:
                    hook = new_hook
                    console.print("[green]Hook updated. Now approve (y/v) or skip (n).[/green]")
                    console.print(f"\n{hook[:200]}")

    # Feature summary
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(
        f"[bold]Features selected: {len(result.selected_features)}/{len(features)}[/bold]  "
        f"[bold]Videos: {video_count}[/bold]"
    )

    if video_count < max_videos and result.selected_features:
        console.print(
            f"[yellow]Only {video_count} videos selected "
            f"(target: {max_videos} for the week)[/yellow]"
        )

    # --- Phase 2: News cards ---
    console.print(Panel.fit(
        f"[bold cyan]News Card Selection[/bold cyan]\n"
        f"[dim]{len(news)} news cards available | Target: ~{max_news} for the week[/dim]",
        border_style="cyan",
    ))

    # Show top news in a table
    table = Table(show_lines=False)
    table.add_column("#", style="bold", width=4)
    table.add_column("Title", width=50)
    table.add_column("Section", width=12)
    table.add_column("Score", justify="right", width=6)

    display_count = min(len(news), 25)
    for i, article in enumerate(news[:display_count], 1):
        score = _score_article(article)
        ext = " [EXT]" if getattr(article, "is_external", False) else ""
        table.add_row(str(i), article.title + ext, article.section_id, f"{score:.1f}")

    console.print(table)

    # Bulk selection
    default_count = min(max_news, len(news))
    count = IntPrompt.ask(
        f"\n[bold]Select top N news cards[/bold]",
        default=default_count,
    )
    count = min(count, len(news))

    # Ask if any should be excluded
    exclude_input = Prompt.ask(
        "[dim]Exclude any? (comma-separated numbers, or Enter to skip)[/dim]",
        default="",
    )
    exclude_set = set()
    if exclude_input.strip():
        for part in exclude_input.split(","):
            part = part.strip()
            if part.isdigit():
                exclude_set.add(int(part))

    selected = 0
    for i, article in enumerate(news[:display_count], 1):
        if i in exclude_set:
            continue
        if selected >= count:
            break

        hook = hook_map.get(article.title, article.title)
        result.selected_news.append(CuratedArticle(
            title=article.title,
            section_id=article.section_id,
            url=article.url or "",
            is_feature=False,
            hook=hook,
            original_hook=hook,
            score=_score_article(article),
            is_external=getattr(article, "is_external", False),
        ))
        selected += 1

    # Final summary
    console.print(Panel.fit(
        f"[bold green]Curation Complete[/bold green]\n\n"
        f"Features: {len(result.selected_features)} selected "
        f"({len(result.video_features)} for video)\n"
        f"News: {len(result.selected_news)} selected\n"
        f"Total articles for the week: {len(result.all_selected)}",
        border_style="green",
    ))

    return result
