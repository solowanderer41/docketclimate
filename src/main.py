#!/usr/bin/env python3
"""
The Docket Social Media Automation — Main Orchestrator

Usage:
    python -m src.main                    # Full pipeline: scrape, generate, post
    python -m src.main scrape             # Just scrape the latest issue
    python -m src.main generate           # Generate posts from last scraped issue
    python -m src.main post               # Post previously generated content
    python -m src.main preview            # Preview what would be posted (dry run)
    python -m src.main test-connections   # Test all enabled platform connections
    python -m src.main video              # Generate videos only
    python -m src.main schedule           # Build a week's posting queue (interactive)
    python -m src.main schedule --auto    # Build queue using top-scored defaults
    python -m src.main post-today         # Post today's queued items
    python -m src.main post-today --dry   # Dry run of today's queue
    python -m src.main queue-status       # Show queue status
    python -m src.main collect-metrics    # Fetch engagement metrics for posted content
    python -m src.main metrics-report     # Show engagement analytics summary
    python -m src.main weekly-report      # Generate weekly engagement report
    python -m src.main review-week        # Weekly review: report + flag best work
    python -m src.main refresh-tokens     # Refresh Meta long-lived tokens
    python -m src.main refresh-tokens --dry  # Check token expiry without refreshing
    python -m src.main token-status       # Quick check of Meta token health
    python -m src.main repost-reel ID     # Delete old Reel and re-publish with animated video
    python -m src.main flag-good ID       # Flag a posted item as good copy / good video
    python -m src.main flag-good          # Interactive batch flagging
    python -m src.main show-exemplars     # Show flagged exemplary content
    python -m src.main add-exemplar       # Write a manual gold-standard exemplar
    python -m src.main compliance-check   # Run publishing law compliance checks
    python -m src.main compliance-check --fix  # Check + auto-fix queue items
    python -m src.main health-check       # Run system health checks
"""

import sys
import json
import yaml
import click
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from dotenv import load_dotenv

load_dotenv()

console = Console()
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
QUEUE_DIR = PROJECT_ROOT / "queue"
LOG_DIR = PROJECT_ROOT / "logs"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_latest_issue_path() -> Path:
    """Find the most recent scraped issue file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUT_DIR.glob("issue_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0]


def _log_run(action: str, results: dict):
    """Append run results to the log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "results": results,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """The Docket Social Media Automation"""
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
def run():
    """Full pipeline: scrape → generate → preview → post"""
    config = _load_config()

    console.print(Panel.fit(
        "[bold cyan]The Docket Social Media Automation[/bold cyan]\n"
        "Full pipeline run",
        border_style="cyan"
    ))

    # Step 1: Scrape
    console.print("\n[bold]Step 1/4: Scraping latest issue[/bold]")
    issue = _do_scrape()
    if not issue:
        return

    # Step 2: Generate content
    console.print("\n[bold]Step 2/4: Generating social media content[/bold]")
    text_posts, video_scripts = _do_generate(issue, config)

    # Step 3: Preview
    console.print("\n[bold]Step 3/4: Preview[/bold]")
    _do_preview(text_posts, video_scripts)

    # Step 4: Confirm and post
    if not Confirm.ask("\n[bold yellow]Ready to post to enabled platforms?[/bold yellow]"):
        console.print("[yellow]Aborted. Content saved for later.[/yellow]")
        return

    console.print("\n[bold]Step 4/4: Publishing[/bold]")
    _do_post(text_posts, video_scripts, config)


@cli.command()
def scrape():
    """Scrape the latest issue of The Docket"""
    issue = _do_scrape()
    if issue:
        console.print(f"\n[green]Done! {len(issue.articles)} articles saved.[/green]")


@cli.command()
def generate():
    """Generate posts from the last scraped issue"""
    config = _load_config()
    issue_path = _get_latest_issue_path()
    if not issue_path:
        console.print("[red]No scraped issue found. Run 'scrape' first.[/red]")
        return

    from src.scraper import DocketIssue
    issue = DocketIssue.load(issue_path)
    console.print(f"Loaded issue #{issue.issue_number} from {issue_path.name}")

    text_posts, video_scripts = _do_generate(issue, config)
    _do_preview(text_posts, video_scripts)


@cli.command()
def preview():
    """Preview what would be posted (dry run)"""
    config = _load_config()
    issue_path = _get_latest_issue_path()
    if not issue_path:
        console.print("[red]No scraped issue found. Run 'scrape' first.[/red]")
        return

    from src.scraper import DocketIssue
    issue = DocketIssue.load(issue_path)
    text_posts, video_scripts = _do_generate(issue, config)
    _do_preview(text_posts, video_scripts)


@cli.command()
def post():
    """Post previously generated content"""
    config = _load_config()
    issue_path = _get_latest_issue_path()
    if not issue_path:
        console.print("[red]No scraped issue found. Run 'scrape' first.[/red]")
        return

    from src.scraper import DocketIssue
    issue = DocketIssue.load(issue_path)
    text_posts, video_scripts = _do_generate(issue, config)

    _do_preview(text_posts, video_scripts)
    if not Confirm.ask("\n[bold yellow]Post this content?[/bold yellow]"):
        console.print("[yellow]Aborted.[/yellow]")
        return

    _do_post(text_posts, video_scripts, config)


@cli.command()
def video():
    """Generate videos only"""
    config = _load_config()
    issue_path = _get_latest_issue_path()
    if not issue_path:
        console.print("[red]No scraped issue found. Run 'scrape' first.[/red]")
        return

    from src.scraper import DocketIssue
    issue = DocketIssue.load(issue_path)

    from src.content_generator import generate_video_scripts
    video_scripts = generate_video_scripts(issue, config)

    if not video_scripts:
        console.print("[yellow]No video scripts generated.[/yellow]")
        return

    _do_generate_videos(video_scripts, config)


@cli.command(name="test-connections")
def test_connections():
    """Test all enabled platform connections"""
    config = _load_config()
    platforms = config.get("platforms", {})

    table = Table(title="Platform Connection Status")
    table.add_column("Platform", style="bold")
    table.add_column("Enabled")
    table.add_column("Status")

    for platform_name, platform_config in platforms.items():
        enabled = platform_config.get("enabled", False)

        if not enabled:
            table.add_row(platform_name, "[dim]No[/dim]", "[dim]Skipped[/dim]")
            continue

        try:
            mod = __import__(f"src.publishers.{platform_name}", fromlist=["test_connection"])
            mod.test_connection()
            table.add_row(platform_name, "[green]Yes[/green]", "[green]Connected ✓[/green]")
        except Exception as e:
            table.add_row(platform_name, "[green]Yes[/green]", f"[red]Failed: {e}[/red]")

    console.print(table)


@cli.command()
@click.option("--auto", is_flag=True, help="Skip interactive curation, use top-scored defaults")
def schedule(auto):
    """Build a week's posting queue with interactive curation."""
    config = _load_config()

    console.print(Panel.fit(
        "[bold cyan]The Docket — Weekly Schedule Builder[/bold cyan]\n"
        f"{'Auto mode (top-scored defaults)' if auto else 'Interactive curation'}",
        border_style="cyan"
    ))

    # Step 0: Collect metrics from last week (feeds into article scoring)
    analytics_config = config.get("analytics", {})
    if analytics_config.get("enabled", True):
        from src.scheduler import find_active_queue as _faq
        from src.analytics import fetch_metrics, save_metrics
        prev_queue_path = _faq(QUEUE_DIR)
        if prev_queue_path:
            console.print("[dim]Collecting metrics from last week's posts (feeds scoring)...[/dim]")
            try:
                prev_metrics = fetch_metrics(prev_queue_path, delay_hours=0)
                if prev_metrics:
                    metrics_path = PROJECT_ROOT / Path(analytics_config.get("metrics_path", "analytics/metrics.json"))
                    save_metrics(prev_metrics, metrics_path)
            except Exception as e:
                console.print(f"[dim]Metrics collection skipped: {e}[/dim]")

    # Step 1: Scrape (or load existing)
    console.print("\n[bold]Step 1/4: Loading issue data[/bold]")
    issue_path = _get_latest_issue_path()
    if not issue_path:
        console.print("[yellow]No scraped issue found. Scraping now...[/yellow]")
        issue = _do_scrape()
        if not issue:
            return
    else:
        from src.scraper import DocketIssue
        issue = DocketIssue.load(issue_path)
        console.print(
            f"[green]Loaded issue #{issue.issue_number} "
            f"({len(issue.articles)} articles)[/green]"
        )

        if not auto and Confirm.ask("Re-scrape latest issue?", default=False):
            issue = _do_scrape()
            if not issue:
                return

    # Step 2: Generate all draft content
    console.print("\n[bold]Step 2/4: Generating draft content for all articles[/bold]")
    from src.content_generator import generate_all_text_posts, generate_all_video_scripts
    text_posts = generate_all_text_posts(issue, config)
    video_scripts = generate_all_video_scripts(issue, config)

    # Step 3: Curate
    console.print("\n[bold]Step 3/4: Content curation[/bold]")
    from src.curator import curate_content
    curation = curate_content(issue, text_posts, video_scripts, config, auto=auto)

    # Save curation
    curation_path = OUTPUT_DIR / f"curation_{issue.issue_number or 'latest'}.json"
    curation.save(curation_path)

    # Step 4: Build schedule
    console.print("\n[bold]Step 4/4: Building week schedule[/bold]")
    from src.scheduler import generate_week_schedule, print_queue_status
    queue = generate_week_schedule(curation, config)

    # Save queue
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    queue_path = QUEUE_DIR / f"week_{issue.issue_number or 'latest'}_{queue.week_start}.json"
    queue.save(queue_path)

    # Show schedule
    print_queue_status(queue)

    _log_run("schedule", {
        "issue": issue.issue_number,
        "features": len(curation.selected_features),
        "news": len(curation.selected_news),
        "videos": len(curation.video_features),
        "queue_items": len(queue.items),
    })


@cli.command(name="post-today")
@click.option("--dry", is_flag=True, help="Dry run — show what would be posted")
def post_today(dry):
    """Post today's queued content to enabled platforms."""
    config = _load_config()

    from zoneinfo import ZoneInfo
    tz_name = config.get("schedule", {}).get("timezone", "America/Los_Angeles")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)

    console.print(Panel.fit(
        f"[bold cyan]The Docket — Daily Poster[/bold cyan]\n"
        f"{'DRY RUN' if dry else 'Live posting'} | "
        f"{now.strftime('%A, %B %d %Y %H:%M %Z')}",
        border_style="cyan"
    ))

    from src.queue_runner import run_daily
    run_daily(config, dry_run=dry)


@cli.command(name="queue-status")
def queue_status():
    """Show the current posting queue status."""
    from src.scheduler import find_active_queue, WeekQueue, print_queue_status

    queue_path = find_active_queue(QUEUE_DIR)
    if not queue_path:
        console.print("[yellow]No active queue found. Run 'schedule' first.[/yellow]")
        return

    queue = WeekQueue.load(queue_path)
    print_queue_status(queue)


# --- Internal pipeline functions ---

def _do_scrape():
    """Scrape the latest issue and save to disk."""
    from src.scraper import scrape_latest_issue

    try:
        issue = scrape_latest_issue()
        output_path = OUTPUT_DIR / f"issue_{issue.issue_number or 'latest'}.json"
        issue.save(output_path)
        return issue
    except Exception as e:
        console.print(f"[red]Scraping failed: {e}[/red]")
        return None


def _do_generate(issue, config):
    """Generate text posts and video scripts from issue data."""
    from src.content_generator import generate_text_posts, generate_video_scripts

    text_posts = generate_text_posts(issue, config)
    video_scripts = generate_video_scripts(issue, config)

    # Save generated content
    content_path = OUTPUT_DIR / f"content_{issue.issue_number or 'latest'}.json"
    content_data = {
        "text_posts": [
            {"platform": p.platform, "text": p.text, "article": p.article_title, "section": p.section}
            for p in text_posts
        ],
        "video_scripts": [
            {"title": v.title, "hook": v.hook, "slides": v.body_slides, "cta": v.cta, "voiceover": v.voiceover_text}
            for v in video_scripts
        ],
    }
    with open(content_path, "w") as f:
        json.dump(content_data, f, indent=2)
    console.print(f"[green]Content saved to {content_path}[/green]")

    return text_posts, video_scripts


def _do_preview(text_posts, video_scripts):
    """Display a preview of all generated content."""
    # Text posts
    table = Table(title="Text Posts")
    table.add_column("Platform", style="bold", width=10)
    table.add_column("Article", width=30)
    table.add_column("Post", width=60)

    for post in text_posts:
        table.add_row(post.platform, post.article_title[:30], post.text[:60] + "...")

    console.print(table)

    # Video scripts
    if video_scripts:
        console.print("\n[bold]Video Scripts:[/bold]")
        for i, script in enumerate(video_scripts, 1):
            console.print(Panel(
                f"[bold]{script.title}[/bold]\n\n"
                f"[cyan]Hook:[/cyan] {script.hook}\n\n"
                f"[cyan]Slides:[/cyan]\n" + "\n".join(f"  {j}. {s}" for j, s in enumerate(script.body_slides, 1)) + "\n\n"
                f"[cyan]CTA:[/cyan] {script.cta}\n\n"
                f"[dim]Voiceover ({len(script.voiceover_text)} chars):[/dim]\n{script.voiceover_text[:200]}...",
                title=f"Video {i}",
                border_style="blue"
            ))


def _do_generate_videos(video_scripts, config):
    """Generate video files from scripts."""
    from src.video.voiceover import generate_voiceover
    from src.video.generator import generate_video

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, script in enumerate(video_scripts, 1):
        console.print(f"\n[bold]Generating video {i}/{len(video_scripts)}: {script.title}[/bold]")

        # Step 1: Generate voiceover
        audio_path = OUTPUT_DIR / f"voiceover_{i}.mp3"
        try:
            console.print("  Generating voiceover...")
            generate_voiceover(script.voiceover_text, audio_path)
        except Exception as e:
            console.print(f"  [yellow]Voiceover failed ({e}), generating silent video[/yellow]")
            audio_path = None

        # Step 2: Generate video
        video_path = OUTPUT_DIR / f"video_{i}.mp4"
        try:
            console.print("  Generating video...")
            generate_video(script, audio_path, video_path, config.get("video", {}))
            console.print(f"  [green]Video saved: {video_path}[/green]")
        except Exception as e:
            console.print(f"  [red]Video generation failed: {e}[/red]")


def _do_post(text_posts, video_scripts, config):
    """Publish content to enabled platforms."""
    platforms = config.get("platforms", {})
    results = {"posted": [], "failed": [], "skipped": []}

    # Post text content
    for post in text_posts:
        platform_name = post.platform
        platform_config = platforms.get(platform_name, {})

        if not platform_config.get("enabled", False):
            results["skipped"].append(f"{platform_name}: {post.article_title}")
            continue

        try:
            publisher = _get_publisher(platform_name)
            result = publisher.publish(post)
            results["posted"].append(f"{platform_name}: {post.article_title}")
            console.print(f"  [green]✓ Posted to {platform_name}[/green]")
        except Exception as e:
            results["failed"].append(f"{platform_name}: {e}")
            console.print(f"  [red]✗ Failed on {platform_name}: {e}[/red]")

    # Post video content
    video_platforms = {name: cfg for name, cfg in platforms.items()
                       if cfg.get("type") == "video" and cfg.get("enabled", False)}

    if video_platforms and video_scripts:
        console.print("\n[bold]Publishing videos...[/bold]")
        _do_generate_videos(video_scripts, config)

        for platform_name in video_platforms:
            for i in range(len(video_scripts)):
                video_path = OUTPUT_DIR / f"video_{i+1}.mp4"
                if not video_path.exists():
                    continue

                try:
                    publisher = _get_publisher(platform_name)
                    if platform_name == "reels":
                        # Reels needs a public URL — user must upload video first
                        console.print(f"  [yellow]Reels requires a public video URL. Upload {video_path} and provide the URL.[/yellow]")
                    elif platform_name == "tiktok":
                        publisher.publish_video(str(video_path), video_scripts[i].hook)
                        console.print(f"  [green]✓ Posted video to {platform_name}[/green]")
                except Exception as e:
                    results["failed"].append(f"{platform_name} video: {e}")
                    console.print(f"  [red]✗ Failed on {platform_name}: {e}[/red]")

    # Summary
    _log_run("post", results)

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [green]Posted: {len(results['posted'])}[/green]")
    console.print(f"  [red]Failed: {len(results['failed'])}[/red]")
    console.print(f"  [dim]Skipped: {len(results['skipped'])}[/dim]")


def _get_publisher(platform_name: str):
    """Dynamically load the publisher module for a platform."""
    if platform_name == "bluesky":
        from src.publishers.bluesky import BlueskyPublisher
        pub = BlueskyPublisher()
        pub.login()
        return pub
    elif platform_name == "twitter":
        from src.publishers.twitter import TwitterPublisher
        pub = TwitterPublisher()
        pub.login()
        return pub
    elif platform_name == "threads":
        from src.publishers.threads import ThreadsPublisher
        pub = ThreadsPublisher()
        pub.login()
        return pub
    elif platform_name == "reels":
        from src.publishers.reels import ReelsPublisher
        pub = ReelsPublisher()
        pub.login()
        return pub
    elif platform_name == "tiktok":
        from src.publishers.tiktok import TikTokPublisher
        pub = TikTokPublisher()
        pub.login()
        return pub
    else:
        raise ValueError(f"Unknown platform: {platform_name}")


@cli.command(name="collect-metrics")
@click.option("--queue", default=None, help="Specific queue file to analyze")
@click.option("--no-delay", is_flag=True, help="Ignore the delay and fetch all posted items")
def collect_metrics(queue, no_delay):
    """Fetch engagement metrics for posted content."""
    from src.analytics import fetch_metrics, save_metrics, print_metrics_table
    from src.scheduler import find_active_queue

    config = _load_config()
    analytics_config = config.get("analytics", {})

    if not analytics_config.get("enabled", True):
        console.print("[yellow]Analytics is disabled in config.yaml[/yellow]")
        return

    # Find queue file
    if queue:
        queue_path = Path(queue)
    else:
        queue_path = find_active_queue(QUEUE_DIR)

    if not queue_path or not queue_path.exists():
        console.print("[yellow]No queue file found. Run 'schedule' first.[/yellow]")
        return

    console.print(f"[cyan]Collecting metrics from {queue_path.name}[/cyan]")

    delay_hours = 0 if no_delay else analytics_config.get("collect_delay_hours", 48)
    metrics = fetch_metrics(queue_path, delay_hours=delay_hours)

    if metrics:
        metrics_path = Path(analytics_config.get("metrics_path", "analytics/metrics.json"))
        save_metrics(metrics, PROJECT_ROOT / metrics_path)
        console.print()
        print_metrics_table(metrics)
    else:
        console.print("[dim]No metrics collected.[/dim]")


@cli.command(name="metrics-report")
def metrics_report():
    """Show engagement analytics summary."""
    from src.analytics import print_metrics_report

    config = _load_config()
    analytics_config = config.get("analytics", {})
    metrics_path = Path(analytics_config.get("metrics_path", "analytics/metrics.json"))
    print_metrics_report(PROJECT_ROOT / metrics_path)


@cli.command(name="weekly-report")
@click.option("--queue", default=None, help="Specific queue file to report on")
def weekly_report(queue):
    """Generate a comprehensive weekly engagement report.

    Collects metrics for all posted content, analyzes performance by platform,
    section, time slot, and day of week, then generates actionable insights.
    """
    from src.analytics import generate_weekly_report
    from src.scheduler import find_active_queue

    config = _load_config()
    analytics_config = config.get("analytics", {})

    if not analytics_config.get("enabled", True):
        console.print("[yellow]Analytics is disabled in config.yaml[/yellow]")
        return

    # Find queue file
    if queue:
        queue_path = Path(queue)
    else:
        queue_path = find_active_queue(QUEUE_DIR)

    if not queue_path or not queue_path.exists():
        console.print("[yellow]No queue file found. Run 'schedule' first.[/yellow]")
        return

    from zoneinfo import ZoneInfo
    tz_name = config.get("schedule", {}).get("timezone", "America/Los_Angeles")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)

    console.print(Panel.fit(
        f"[bold cyan]The Docket — Weekly Engagement Report[/bold cyan]\n"
        f"{now.strftime('%A, %B %d %Y %H:%M %Z')}",
        border_style="cyan"
    ))

    metrics_path = PROJECT_ROOT / Path(analytics_config.get("metrics_path", "analytics/metrics.json"))
    report = generate_weekly_report(queue_path, config, metrics_path)

    if report:
        _log_run("weekly-report", {
            "issue": report.get("issue_number"),
            "posts_tracked": report.get("summary", {}).get("total_posts_tracked", 0),
            "total_engagement": report.get("summary", {}).get("total_engagement", 0),
            "completion_rate": report.get("queue_completion", {}).get("completion_rate", 0),
        })


@cli.command(name="review-week")
@click.option("--queue", default=None, help="Specific queue file to review")
def review_week(queue):
    """Weekly review: analytics report + flag your best work via text prompt.

    Runs the weekly engagement report, then shows your top-performing posts
    with their full text. You describe which ones stood out and why in a
    single text prompt — no CLI flags, just plain language.

    Examples:
        python -m src.main review-week
        python -m src.main review-week --queue queue/week_18_2026-02-09.json
    """
    import re
    from src.analytics import generate_weekly_report, load_metrics
    from src.scheduler import find_active_queue, WeekQueue
    from src.exemplars import flag_from_queue_item, save_exemplar
    from rich.prompt import Prompt

    config = _load_config()
    analytics_config = config.get("analytics", {})
    exemplar_config = config.get("exemplars", {})

    if not analytics_config.get("enabled", True):
        console.print("[yellow]Analytics is disabled in config.yaml[/yellow]")
        return

    # --- Find queue ---
    if queue:
        queue_path = Path(queue)
    else:
        queue_path = find_active_queue(QUEUE_DIR)

    if not queue_path or not queue_path.exists():
        console.print("[yellow]No queue file found. Run 'schedule' first.[/yellow]")
        return

    # --- Run the weekly report ---
    from zoneinfo import ZoneInfo
    tz_name = config.get("schedule", {}).get("timezone", "America/Los_Angeles")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)

    console.print(Panel.fit(
        f"[bold cyan]The Docket — Weekly Review[/bold cyan]\n"
        f"{now.strftime('%A, %B %d %Y %H:%M %Z')}",
        border_style="cyan"
    ))

    metrics_path = PROJECT_ROOT / Path(
        analytics_config.get("metrics_path", "analytics/metrics.json")
    )
    report = generate_weekly_report(queue_path, config, metrics_path)

    if not report:
        console.print("[yellow]No report data — nothing to review.[/yellow]")
        return

    # --- Load queue + metrics for the review step ---
    wq = WeekQueue.load(queue_path)
    posted = [i for i in wq.items if i.status == "posted"]

    if not posted:
        console.print("[yellow]No posted items to review.[/yellow]")
        return

    all_metrics = load_metrics(metrics_path)
    queue_ids = {i.id for i in wq.items}
    week_metrics = [m for m in all_metrics if m.queue_item_id in queue_ids]
    metrics_map = {m.queue_item_id: m for m in week_metrics}

    # Sort posted items by engagement score (highest first)
    def _score(item):
        m = metrics_map.get(item.id)
        return m.engagement_score if m else 0
    posted_ranked = sorted(posted, key=_score, reverse=True)

    # --- Show top posts with full text for review ---
    console.print()
    console.print(Panel(
        "[bold]Review your best work this week.[/bold]\n"
        "Below are your top-performing posts — read through them and tell me\n"
        "which ones you'd flag as exemplary work, and why.\n\n"
        "[dim]Type something like:[/dim]\n"
        '  "1 and 3 — great hooks, the opening lines really land"\n'
        '  "2 — perfect pacing on the video slides"\n'
        '  "none" or "skip" to skip flagging',
        border_style="green",
        title="[bold green]✍ Content Review[/bold green]",
    ))
    console.print()

    # Show up to 10 posts for review (ranked by engagement)
    review_count = min(len(posted_ranked), 10)
    review_items = posted_ranked[:review_count]

    for idx, item in enumerate(review_items, 1):
        m = metrics_map.get(item.id)
        score = m.engagement_score if m else 0
        likes = m.likes if m else 0
        reposts = m.reposts if m else 0
        replies = m.replies if m else 0

        type_icon = "🎬" if item.content_type == "video" else "📝"
        stats = f"L:{likes} R:{reposts} C:{replies} → [bold]{score:.0f}[/bold]"

        console.print(f"[bold cyan]#{idx}[/bold cyan] {type_icon} [{item.platform}] "
                      f"[bold]{item.article_title}[/bold]")
        console.print(f"    [dim]{item.section} | {stats}[/dim]")

        # Show the actual content
        if item.content_type == "video" and item.video_script:
            vs = item.video_script
            hook = vs.get("hook", "")
            slides = vs.get("body_slides", [])
            console.print(f'    [italic]Hook: "{hook}"[/italic]')
            for i, slide in enumerate(slides[:4], 1):
                console.print(f"    [dim]  Slide {i}: {slide}[/dim]")
            if len(slides) > 4:
                console.print(f"    [dim]  ... +{len(slides) - 4} more slides[/dim]")
        else:
            # Show the post text (truncate for readability)
            text = item.text.strip()
            lines = text.split("\n")
            preview = "\n".join(lines[:4])
            if len(lines) > 4:
                preview += f"\n    [dim]... ({len(lines) - 4} more lines)[/dim]"
            console.print(f"    [italic]{preview}[/italic]")

        console.print()

    # --- Prompt for feedback ---
    feedback = Prompt.ask(
        "[bold green]Which posts stood out?[/bold green] "
        "(numbers + why, or 'skip')",
        default="skip",
    )

    if feedback.strip().lower() in ("skip", "none", "q", "quit", ""):
        console.print("[dim]No posts flagged. See you next week![/dim]")
        _log_run("review-week", {
            "issue": report.get("issue_number"),
            "posts_reviewed": review_count,
            "posts_flagged": 0,
        })
        return

    # --- Parse the response ---
    # Extract numbers from the feedback text
    numbers = re.findall(r'\b(\d{1,2})\b', feedback)
    selected_indices = []
    for n in numbers:
        idx = int(n)
        if 1 <= idx <= review_count:
            selected_indices.append(idx)

    if not selected_indices:
        console.print("[yellow]Couldn't find any post numbers in your response. "
                      "No posts flagged.[/yellow]")
        _log_run("review-week", {
            "issue": report.get("issue_number"),
            "posts_reviewed": review_count,
            "posts_flagged": 0,
        })
        return

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    selected_indices = unique_indices

    # Use the full feedback as the notes (the "why")
    # Strip out just the numbers to leave the reasoning
    notes = re.sub(r'^\s*[\d,\s]+[-—–:]*\s*', '', feedback).strip()
    if not notes or notes == feedback.strip():
        # If we couldn't cleanly separate numbers from text, use the whole thing
        notes = feedback.strip()

    exemplars_path = PROJECT_ROOT / exemplar_config.get(
        "path", "data/exemplars/exemplars.json"
    )

    flagged = []
    for idx in selected_indices:
        item = review_items[idx - 1]
        etype = "good_video" if item.content_type == "video" else "good_copy"
        m = metrics_map.get(item.id)
        score = m.engagement_score if m else 0.0

        exemplar = flag_from_queue_item(
            item, etype, issue_number=wq.issue_number,
            engagement_score=score, notes=notes,
        )
        save_exemplar(exemplar, exemplars_path)
        flagged.append((idx, item))

    # --- Confirmation ---
    console.print()
    flag_lines = []
    for idx, item in flagged:
        type_icon = "🎬" if item.content_type == "video" else "📝"
        flag_lines.append(f"  {type_icon} #{idx} — {item.article_title}")

    console.print(Panel(
        f"[bold green]Flagged {len(flagged)} post{'s' if len(flagged) != 1 else ''}:[/bold green]\n"
        + "\n".join(flag_lines)
        + (f"\n\n[dim]Notes: {notes}[/dim]" if notes else ""),
        border_style="green",
        title="[bold]✓ Exemplars Saved[/bold]",
    ))

    _log_run("review-week", {
        "issue": report.get("issue_number"),
        "posts_reviewed": review_count,
        "posts_flagged": len(flagged),
        "flagged_ids": [item.id for _, item in flagged],
    })


@cli.command(name="refresh-tokens")
@click.option("--dry", is_flag=True, help="Inspect tokens without refreshing")
def refresh_tokens(dry):
    """Refresh Meta long-lived tokens (Threads + Instagram/Reels).

    Meta access tokens expire after ~60 days. This command exchanges
    the current token for a fresh one, validates it, and updates .env.

    Run monthly via launchd to prevent token expiration.
    """
    from src.token_manager import refresh_all_meta_tokens

    from zoneinfo import ZoneInfo
    config = _load_config()
    tz_name = config.get("schedule", {}).get("timezone", "America/Los_Angeles")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)

    console.print(Panel.fit(
        f"[bold cyan]The Docket — Meta Token Refresh[/bold cyan]\n"
        f"{'DRY RUN (inspect only)' if dry else 'Refreshing tokens'} | "
        f"{now.strftime('%A, %B %d %Y %H:%M %Z')}",
        border_style="cyan"
    ))

    results = refresh_all_meta_tokens(dry_run=dry)

    # Summary
    console.print(f"\n[bold cyan]{'─' * 50}[/bold cyan]")
    console.print("[bold]Summary:[/bold]")
    for token_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "refreshed":
            days = result.get("expires_in_days", "?")
            console.print(f"  [green]✓ {token_name}: refreshed (~{days} days)[/green]")
        elif status == "dry_run":
            days = result.get("days_remaining")
            if days is not None:
                console.print(f"  [cyan]○ {token_name}: {days} days remaining[/cyan]")
            else:
                console.print(f"  [cyan]○ {token_name}: status unknown[/cyan]")
        elif status == "expired":
            console.print(f"  [red]✗ {token_name}: EXPIRED — manual renewal needed[/red]")
        elif status == "skipped":
            console.print(f"  [dim]- {token_name}: skipped (not set)[/dim]")
        else:
            reason = result.get("reason", "unknown error")
            console.print(f"  [red]✗ {token_name}: {reason}[/red]")

    _log_run("refresh-tokens", {
        "dry_run": dry,
        "results": {k: v.get("status") for k, v in results.items()},
    })


@cli.command(name="token-status")
def token_status():
    """Quick health check of Meta token expiry dates."""
    from src.token_manager import check_token_health

    console.print(Panel.fit(
        "[bold cyan]The Docket — Meta Token Status[/bold cyan]",
        border_style="cyan"
    ))

    health = check_token_health()

    table = Table(title="Meta Token Health")
    table.add_column("Token", style="bold", width=30)
    table.add_column("Platform", width=16)
    table.add_column("Status", width=12)
    table.add_column("Days Left", justify="right", width=10)
    table.add_column("Expires", width=20)

    for token_name, info in health.items():
        status = info.get("status", "unknown")
        days = info.get("days_remaining")
        expires = info.get("expires_at", "—")

        if status == "valid" and days is not None:
            if days > 30:
                status_display = "[green]Valid[/green]"
                days_display = f"[green]{days}[/green]"
            elif days > 7:
                status_display = "[yellow]Expiring Soon[/yellow]"
                days_display = f"[yellow]{days}[/yellow]"
            else:
                status_display = "[red]Critical[/red]"
                days_display = f"[red]{days}[/red]"
        elif status == "expired":
            status_display = "[red]Expired[/red]"
            days_display = "[red]0[/red]"
        elif status == "missing":
            status_display = "[dim]Not Set[/dim]"
            days_display = "[dim]—[/dim]"
        else:
            status_display = f"[yellow]{status}[/yellow]"
            days_display = "[dim]?[/dim]"

        table.add_row(
            token_name,
            info.get("label", ""),
            status_display,
            days_display,
            str(expires)[:20] if expires else "—",
        )

    console.print(table)

    # Alert if any tokens need attention
    for token_name, info in health.items():
        days = info.get("days_remaining")
        if days is not None and days <= 7:
            console.print(
                f"\n[bold red]⚠ {info['label']} token expires in {days} days! "
                f"Run: python -m src.main refresh-tokens[/bold red]"
            )
        elif days is not None and days <= 30:
            console.print(
                f"\n[yellow]⚡ {info['label']} token expires in {days} days. "
                f"Consider running: python -m src.main refresh-tokens[/yellow]"
            )


@cli.command(name="repost-reel")
@click.argument("item_id")
@click.option("--dry", is_flag=True, help="Dry run — show what would happen without deleting or posting")
@click.option("--keep-old", is_flag=True, help="Skip deleting the old Reel (just regenerate and post)")
def repost_reel(item_id, dry, keep_old):
    """Delete an old Reel and re-publish with the current video pipeline.

    Finds the queue item by ID, deletes the old Instagram Reel,
    regenerates the video using the animated pipeline, uploads it,
    publishes the new Reel, and updates the queue and analytics files.

    Example:
        python -m src.main repost-reel 18_fri_050
    """
    config = _load_config()

    # 1. Load the queue and find the item
    from src.scheduler import find_active_queue, WeekQueue

    queue_path = find_active_queue(QUEUE_DIR)
    if not queue_path:
        console.print("[red]No active queue found.[/red]")
        return

    queue = WeekQueue.load(queue_path)
    item = next((i for i in queue.items if i.id == item_id), None)
    if not item:
        console.print(f"[red]Queue item '{item_id}' not found.[/red]")
        return

    if item.content_type != "video":
        console.print(f"[red]Item '{item_id}' is type '{item.content_type}', not video.[/red]")
        return

    console.print(Panel.fit(
        f"[bold cyan]Repost Reel[/bold cyan]\n"
        f"Item: {item.id}\n"
        f"Article: {item.article_title}\n"
        f"Section: {item.section}\n"
        f"Old media: {item.post_uri or 'none'}\n"
        f"{'DRY RUN' if dry else 'LIVE'}",
        border_style="cyan",
    ))

    old_post_uri = item.post_uri

    if dry:
        console.print("[cyan]DRY RUN — would delete old Reel, regenerate video, and re-publish.[/cyan]")
        return

    # 2. Delete the old Reel from Instagram
    if old_post_uri and not keep_old:
        from src.publishers.reels import ReelsPublisher

        publisher = ReelsPublisher()
        publisher.login()
        console.print(f"[bold]Deleting old Reel: {old_post_uri}[/bold]")
        try:
            publisher.delete_media(old_post_uri)
        except Exception as e:
            console.print(
                f"[yellow]Delete failed ({e}), continuing with re-publish...[/yellow]"
            )

    # 3. Reset queue item state so the pipeline treats it as fresh
    item.status = "pending"
    item.attempts = 0
    item.last_error = None
    item.posted_at = None
    item.post_uri = None
    queue.save(queue_path)
    console.print("[green]Queue item reset to pending.[/green]")

    # 4. Run the video pipeline for this single item
    from src.queue_runner import _post_video_item

    success, result = _post_video_item(item, config)

    if success:
        queue.mark_posted(item.id, result)
        queue.save(queue_path)
        console.print(f"[bold green]New Reel published: {result}[/bold green]")

        # 5. Update analytics — remove old entry
        _update_analytics_for_repost(item_id)
    else:
        queue.mark_failed(item.id, result)
        queue.save(queue_path)
        console.print(f"[bold red]Repost failed: {result}[/bold red]")

    _log_run("repost-reel", {
        "item_id": item_id,
        "old_uri": old_post_uri,
        "new_uri": item.post_uri,
        "success": success,
    })


def _update_analytics_for_repost(item_id: str):
    """Remove the old metrics entry so it doesn't pollute analytics."""
    from src.analytics import load_metrics
    from dataclasses import asdict

    metrics_path = PROJECT_ROOT / "analytics" / "metrics.json"
    existing = load_metrics(metrics_path)
    filtered = [m for m in existing if m.queue_item_id != item_id]

    if len(filtered) < len(existing):
        with open(metrics_path, "w") as f:
            json.dump([asdict(m) for m in filtered], f, indent=2)
        console.print(
            f"[dim]Removed old metrics entry for {item_id} "
            f"({len(existing) - len(filtered)} entries removed).[/dim]"
        )


@cli.command(name="flag-good")
@click.argument("item_id", required=False, default=None)
@click.option("--type", "flag_type", type=click.Choice(["copy", "video", "auto"]),
              default="auto", help="Flag as good copy or good video (auto-detects)")
@click.option("--notes", default="", help="Optional note about why this is good")
@click.option("--queue", default=None, help="Specific queue file to search")
def flag_good(item_id, flag_type, notes, queue):
    """Flag a posted item as exemplary work (good copy or good video).

    With ITEM_ID: flags that specific item.
    Without ITEM_ID: shows all posted items for interactive selection.

    Examples:
        python -m src.main flag-good 19_mon_001
        python -m src.main flag-good 19_mon_001 --notes "killer opening"
        python -m src.main flag-good              # interactive batch mode
    """
    from src.scheduler import find_active_queue, WeekQueue
    from src.exemplars import (
        flag_from_queue_item, save_exemplar, load_exemplars,
        DEFAULT_EXEMPLARS_PATH,
    )
    from rich.prompt import Prompt

    config = _load_config()
    exemplar_config = config.get("exemplars", {})
    exemplars_path_str = exemplar_config.get("path", "data/exemplars/exemplars.json")
    exemplars_path = PROJECT_ROOT / exemplars_path_str

    # --- Find queue ---
    if queue:
        queue_path = Path(queue)
    else:
        queue_path = find_active_queue(QUEUE_DIR)

    if not queue_path or not queue_path.exists():
        console.print("[yellow]No queue file found. Run 'schedule' first.[/yellow]")
        return

    wq = WeekQueue.load(queue_path)
    posted = [i for i in wq.items if i.status == "posted"]

    if not posted:
        console.print("[yellow]No posted items found in the queue.[/yellow]")
        return

    # --- Look up engagement scores from metrics ---
    metrics_map = {}
    try:
        from src.analytics import load_metrics
        analytics_config = config.get("analytics", {})
        metrics_path_str = analytics_config.get("metrics_path", "analytics/metrics.json")
        all_metrics = load_metrics(PROJECT_ROOT / metrics_path_str)
        for m in all_metrics:
            metrics_map[m.queue_item_id] = m.engagement_score
    except Exception:
        pass  # metrics are optional enrichment

    # --- Interactive batch mode (no item_id) ---
    if not item_id:
        table = Table(title=f"Posted Items — {queue_path.name}", show_lines=False)
        table.add_column("#", width=4, justify="right")
        table.add_column("ID", width=16)
        table.add_column("Type", width=6)
        table.add_column("Platform", width=10)
        table.add_column("Article", width=40)
        table.add_column("Score", width=6, justify="right")

        for idx, item in enumerate(posted, 1):
            score = metrics_map.get(item.id, 0)
            score_str = f"{score:.0f}" if score else "-"
            table.add_row(
                str(idx),
                item.id,
                item.content_type[:5],
                item.platform,
                item.article_title[:38] + (".." if len(item.article_title) > 38 else ""),
                score_str,
            )

        console.print(table)
        selection = Prompt.ask(
            "\n[bold]Flag which items?[/bold] (comma-separated numbers, or 'q' to quit)",
            default="q",
        )

        if selection.strip().lower() == "q":
            return

        try:
            indices = [int(s.strip()) for s in selection.split(",") if s.strip()]
        except ValueError:
            console.print("[red]Invalid input — use comma-separated numbers.[/red]")
            return

        selected_items = []
        for idx in indices:
            if 1 <= idx <= len(posted):
                selected_items.append(posted[idx - 1])
            else:
                console.print(f"[yellow]Skipping invalid index: {idx}[/yellow]")

        if not selected_items:
            console.print("[yellow]No valid items selected.[/yellow]")
            return

        if not notes:
            notes = Prompt.ask("[dim]Notes (optional, Enter to skip)[/dim]", default="")

        for item in selected_items:
            etype = "good_video" if item.content_type == "video" else "good_copy"
            score = metrics_map.get(item.id, 0.0)
            exemplar = flag_from_queue_item(
                item, etype, issue_number=wq.issue_number,
                engagement_score=score, notes=notes,
            )
            save_exemplar(exemplar, exemplars_path)

        return

    # --- Single item mode ---
    item = next((i for i in wq.items if i.id == item_id), None)
    if not item:
        # Search all queue files if not in active queue
        all_queues = sorted(QUEUE_DIR.glob("week_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for qp in all_queues:
            try:
                other_wq = WeekQueue.load(qp)
                item = next((i for i in other_wq.items if i.id == item_id), None)
                if item:
                    wq = other_wq
                    break
            except Exception:
                continue

    if not item:
        console.print(f"[red]Queue item '{item_id}' not found in any queue.[/red]")
        return

    if item.status != "posted":
        console.print(
            f"[yellow]Item '{item_id}' has status '{item.status}'. "
            f"Only posted items can be flagged.[/yellow]"
        )
        return

    # Auto-detect type
    if flag_type == "auto":
        etype = "good_video" if item.content_type == "video" else "good_copy"
    else:
        etype = f"good_{flag_type}"

    score = metrics_map.get(item.id, 0.0)

    if not notes:
        notes = Prompt.ask("[dim]Notes (optional, Enter to skip)[/dim]", default="")

    exemplar = flag_from_queue_item(
        item, etype, issue_number=wq.issue_number,
        engagement_score=score, notes=notes,
    )
    save_exemplar(exemplar, exemplars_path)

    type_label = "Good Video" if etype == "good_video" else "Good Copy"
    console.print(Panel.fit(
        f"[bold green]{type_label}[/bold green]\n"
        f"Article: {item.article_title}\n"
        f"Platform: {item.platform}\n"
        f"Engagement: {score:.0f}\n"
        f"{'Notes: ' + notes if notes else ''}",
        border_style="green",
        title="[bold]Flagged[/bold]",
    ))

    _log_run("flag-good", {
        "item_id": item_id,
        "type": etype,
        "article": item.article_title,
    })


@cli.command(name="show-exemplars")
@click.option("--type", "filter_type", type=click.Choice(["all", "copy", "video"]),
              default="all", help="Filter by exemplar type")
@click.option("--section", default=None, help="Filter by section ID")
@click.option("--detail", default=None, help="Show full detail for a specific exemplar ID")
def show_exemplars(filter_type, section, detail):
    """Show flagged exemplary content.

    Examples:
        python -m src.main show-exemplars
        python -m src.main show-exemplars --type video
        python -m src.main show-exemplars --section lived
        python -m src.main show-exemplars --detail ex_19_mon_001
    """
    from src.exemplars import (
        load_exemplars, print_exemplars_table, print_exemplar_detail,
        DEFAULT_EXEMPLARS_PATH,
    )

    config = _load_config()
    exemplar_config = config.get("exemplars", {})
    exemplars_path_str = exemplar_config.get("path", "data/exemplars/exemplars.json")
    exemplars_path = PROJECT_ROOT / exemplars_path_str

    exemplars = load_exemplars(exemplars_path)

    if not exemplars:
        console.print("[yellow]No exemplars flagged yet. Use 'flag-good' to start.[/yellow]")
        return

    # Show detail for a specific exemplar
    if detail:
        match = next((e for e in exemplars if e.id == detail), None)
        if match:
            print_exemplar_detail(match)
        else:
            console.print(f"[red]Exemplar '{detail}' not found.[/red]")
        return

    # Apply filters
    if filter_type == "copy":
        exemplars = [e for e in exemplars if e.exemplar_type == "good_copy"]
    elif filter_type == "video":
        exemplars = [e for e in exemplars if e.exemplar_type == "good_video"]

    if section:
        exemplars = [e for e in exemplars if e.section == section]

    if not exemplars:
        console.print("[yellow]No exemplars match the filters.[/yellow]")
        return

    # Sort by flagged_at descending
    exemplars.sort(key=lambda e: e.flagged_at, reverse=True)

    print_exemplars_table(exemplars)

    copies = sum(1 for e in exemplars if e.exemplar_type == "good_copy")
    videos = sum(1 for e in exemplars if e.exemplar_type == "good_video")
    console.print(f"\n[dim]{len(exemplars)} exemplars shown ({copies} copy, {videos} video)[/dim]")


@cli.command(name="add-exemplar")
@click.option("--platform", required=True,
              type=click.Choice(["twitter", "bluesky", "threads"]),
              help="Target platform for this exemplar")
@click.option("--section", required=True,
              type=click.Choice(["lived", "systems", "science", "futures", "archive", "lab"]),
              help="Section ID this exemplar belongs to")
@click.option("--title", required=True, help="Article title or topic")
@click.option("--notes", default="", help="Why this post is good (optional)")
def add_exemplar(platform, section, title, notes):
    """Create a manual exemplar — a gold-standard post written by you.

    Opens your $EDITOR to compose the post text. Lines starting with #
    are stripped as comments.

    Manual exemplars rank highest when injected into LLM prompts
    (engagement_score = 100).

    Examples:
        python -m src.main add-exemplar --platform twitter --section lived --title "Article name"
        python -m src.main add-exemplar --platform bluesky --section science --title "Ocean warming" --notes "great hook"
    """
    from src.exemplars import (
        create_manual_exemplar, save_exemplar, load_exemplars,
    )

    config = _load_config()
    exemplar_config = config.get("exemplars", {})
    exemplars_path = PROJECT_ROOT / exemplar_config.get(
        "path", "data/exemplars/exemplars.json"
    )

    # Platform character limits
    platform_limits = {
        "twitter": config.get("platforms", {}).get("twitter", {}).get("max_chars", 280),
        "bluesky": config.get("platforms", {}).get("bluesky", {}).get("max_chars", 300),
        "threads": config.get("platforms", {}).get("threads", {}).get("max_chars", 500),
    }
    char_limit = platform_limits.get(platform, 300)

    # Open editor with a helpful template
    template = (
        f"# Write your exemplar post for {platform} ({char_limit} char limit)\n"
        f"# Article: {title}\n"
        f"# Section: {section}\n"
        f"# Lines starting with # will be removed.\n"
        f"# Save and close your editor when done.\n"
        f"\n"
    )

    text = click.edit(template)

    if text is None:
        console.print("[yellow]Editor closed without saving. No exemplar created.[/yellow]")
        return

    # Strip comment lines and clean up
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    clean_text = "\n".join(lines).strip()

    if not clean_text:
        console.print("[yellow]Empty post text. No exemplar created.[/yellow]")
        return

    # Validate length
    if len(clean_text) > char_limit:
        console.print(
            f"[red]Post is {len(clean_text)} chars — exceeds {platform} limit "
            f"of {char_limit}. Please shorten and retry.[/red]"
        )
        return

    if len(clean_text) < 40:
        console.print("[yellow]Post is very short (<40 chars). Consider adding more substance.[/yellow]")

    # Create and save
    exemplar = create_manual_exemplar(
        platform=platform,
        section=section,
        article_title=title,
        text=clean_text,
        notes=notes,
    )
    save_exemplar(exemplar, exemplars_path)

    console.print(Panel.fit(
        f"[bold green]Manual Exemplar Created[/bold green]\n\n"
        f"Platform: {platform} ({len(clean_text)}/{char_limit} chars)\n"
        f"Section: {section}\n"
        f"Article: {title}\n\n"
        f"[italic]{clean_text}[/italic]",
        border_style="green",
        title=f"[bold]{exemplar.id}[/bold]",
    ))

    # Check total count and hint about activation
    all_exemplars = load_exemplars(exemplars_path)
    total = len(all_exemplars)
    copies = sum(1 for e in all_exemplars if e.exemplar_type == "good_copy")

    if copies >= 5 and not exemplar_config.get("inject_into_prompts", False):
        console.print(
            f"\n[bold cyan]You have {copies} copy exemplars![/bold cyan] "
            f"Consider enabling prompt injection:\n"
            f"  Set [bold]exemplars.inject_into_prompts: true[/bold] in config.yaml"
        )
    elif total > 0:
        console.print(
            f"[dim]Exemplar store: {total} total ({copies} copy). "
            f"{'Need ' + str(5 - copies) + ' more copy exemplars to enable prompt injection.' if copies < 5 else ''}[/dim]"
        )

    _log_run("add-exemplar", {
        "platform": platform,
        "section": section,
        "article": title,
        "chars": len(clean_text),
        "exemplar_id": exemplar.id,
    })


@cli.command(name="compliance-check")
@click.option("--fix", is_flag=True, help="Apply auto-fixes and save queue")
@click.option("--item", "item_id", default=None, help="Check a specific queue item by ID")
def compliance_check_cmd(fix, item_id):
    """Run publishing law compliance checks on the active queue.

    Validates pending text posts against fair-use, attribution, and
    copyright principles.  Shows results in a terminal table.

    Examples:
        python -m src.main compliance-check              # check active queue
        python -m src.main compliance-check --fix        # check + auto-fix
        python -m src.main compliance-check --item ID    # check specific item
    """
    from src.scheduler import find_active_queue, WeekQueue
    from src.compliance import run_compliance_check, print_compliance_report

    config = _load_config()

    console.print(Panel.fit(
        "[bold cyan]The Docket — Compliance Check[/bold cyan]",
        border_style="cyan",
    ))

    comp_cfg = config.get("compliance", {})
    if not comp_cfg.get("enabled", False):
        console.print("[yellow]Compliance is disabled in config.yaml. "
                      "Set compliance.enabled: true to activate.[/yellow]")
        return

    queue_path = find_active_queue(QUEUE_DIR)
    if not queue_path:
        console.print("[red]No active queue found. Run 'schedule' first.[/red]")
        return

    queue = WeekQueue.load(queue_path)
    console.print(f"Queue: [bold]{queue_path.name}[/bold] "
                  f"({len(queue.items)} items)")

    # Filter to target items
    if item_id:
        targets = [i for i in queue.items if i.id == item_id]
        if not targets:
            console.print(f"[red]Item '{item_id}' not found in queue.[/red]")
            return
    else:
        targets = [i for i in queue.items
                   if i.content_type == "text" and i.status == "pending"]

    if not targets:
        console.print("[dim]No pending text items to check.[/dim]")
        return

    # Override auto_fix based on --fix flag
    check_config = dict(config)
    if not fix:
        # Don't auto-fix unless --fix is passed
        if "compliance" in check_config:
            check_config["compliance"] = dict(check_config["compliance"])
            check_config["compliance"]["auto_fix"] = False

    results = []
    modified = False
    for item in targets:
        result = run_compliance_check(
            post_text=item.text,
            source_text=None,
            article_url=item.url or None,
            platform=item.platform,
            config=check_config if not fix else config,
        )
        results.append((item.id, result))

        if fix and result.fixed_text:
            item.compliance_original = item.text
            item.text = result.fixed_text
            modified = True

        item.compliance_status = result.status
        item.compliance_detail = result.summary

        if fix and result.status == "blocked":
            item.status = "blocked"
            modified = True

    print_compliance_report(results)

    if fix and modified:
        queue.save(queue_path)
        console.print(f"\n[green]Queue updated with compliance fixes.[/green]")
    elif fix:
        console.print(f"\n[dim]No fixes needed.[/dim]")

    _log_run("compliance-check", {
        "queue": queue_path.name,
        "checked": len(results),
        "fixed": fix,
        "results": {item_id: r.status for item_id, r in results},
    })


@cli.command(name="health-check")
@click.option("--no-notify", is_flag=True, help="Check only — skip webhook notification")
def health_check(no_notify):
    """Run system health checks and optionally notify on problems.

    Checks: today's posting status, Meta token expiry, active queue,
    last pipeline run, and disk space.

    Examples:
        python -m src.main health-check              # check + notify on failure
        python -m src.main health-check --no-notify   # check only, no webhook
    """
    from src.watchdog import run_health_check, print_health_report

    config = _load_config()

    console.print(Panel.fit(
        "[bold cyan]The Docket — Health Check[/bold cyan]",
        border_style="cyan",
    ))

    result = run_health_check(config)
    print_health_report(result)

    # Send notification if any checks failed
    if not no_notify and result["status"] != "healthy":
        checks_failed = [
            c for c in result["checks"] if c["status"] in ("fail", "warn")
        ]
        try:
            from src.notifier import notify_health_alert
            notify_health_alert(checks_failed, config=config)
        except Exception as e:
            console.print(f"[yellow]Notification failed: {e}[/yellow]")

    _log_run("health-check", {
        "status": result["status"],
        "checks": {c["name"]: c["status"] for c in result["checks"]},
    })


@cli.command(name="pin-post")
@click.option("--platform", default=None,
              type=click.Choice(["twitter", "bluesky", "threads", "reels"]),
              help="Generate for a specific platform (default: all)")
@click.option("--post", is_flag=True, help="Actually post (default: just preview)")
def pin_post_cmd(platform, post):
    """Generate pinned introduction posts for profile pages.

    By default previews the text for each platform. Use --post to publish.
    You'll need to manually pin the post on each platform after publishing.
    """
    from src.content_generator import generate_pin_post

    platforms = [platform] if platform else ["twitter", "bluesky", "threads", "reels"]

    for p in platforms:
        text = generate_pin_post(p)
        console.print(f"\n[bold cyan]{'=' * 40}[/bold cyan]")
        console.print(f"[bold]{p.upper()}[/bold] ({len(text)} chars)")
        console.print(f"[cyan]{'=' * 40}[/cyan]")
        console.print(text)

        if post:
            try:
                from src.queue_runner import _get_publisher
                publisher = _get_publisher(p)
                if p == "reels":
                    console.print(f"[yellow]Reels pin post must be a static image — post manually[/yellow]")
                    continue
                from src.content_generator import TextPost
                post_obj = TextPost(
                    platform=p, text=text, hashtags=[], article_title="[Pin] Introduction",
                    section="meta",
                )
                result = publisher.publish(post_obj)
                console.print(f"[green]Posted to {p}: {result}[/green]")
                console.print(f"[yellow]Now pin this post on {p}![/yellow]")
            except Exception as e:
                console.print(f"[red]Failed to post to {p}: {e}[/red]")
        else:
            console.print(f"[dim](preview only — use --post to publish)[/dim]")

    if not post:
        console.print(f"\n[dim]Run with --post to publish, then pin manually on each platform.[/dim]")


if __name__ == "__main__":
    cli()
