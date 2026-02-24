"""
Engagement analytics for The Docket.

Fetches post metrics from each platform, stores them cumulatively,
and provides historical averages to improve article scoring and
curation decisions.

Supported platforms:
    - Bluesky: likes, reposts, replies (public API, no auth needed)
    - Twitter: likes, retweets, replies (API v2, Basic tier)
    - Threads: likes, replies, reposts (Graph API, needs threads_manage_insights)
    - Instagram Reels: likes, comments, shares, plays (Graph API, needs instagram_manage_insights)
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_METRICS_PATH = PROJECT_ROOT / "analytics" / "metrics.json"


@dataclass
class PostMetrics:
    """Engagement metrics for a single posted item."""
    queue_item_id: str
    platform: str
    post_uri: str
    article_title: str
    section: str
    is_feature: bool
    hook: str
    posted_at: str
    fetched_at: str
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    views: int | None = None
    engagement_score: float = 0.0

    def compute_score(self):
        """Weighted engagement: replies > reposts > likes."""
        self.engagement_score = self.likes + (self.reposts * 2) + (self.replies * 3)
        return self.engagement_score


# ---------------------------------------------------------------------------
# Platform-specific metric fetchers
# ---------------------------------------------------------------------------

def _fetch_bluesky_metrics(uri: str) -> dict:
    """
    Fetch engagement metrics for a Bluesky post.

    Uses the public AT Protocol API — no authentication required.
    """
    try:
        # Fix malformed URIs stored as dict strings (e.g. "{'uri': 'at://...'}")
        if uri.startswith("{") and "uri" in uri:
            import ast
            try:
                parsed = ast.literal_eval(uri)
                uri = parsed.get("uri", uri)
            except (ValueError, SyntaxError):
                pass

        response = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread",
            params={"uri": uri, "depth": 0},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        post = data.get("thread", {}).get("post", {})

        result = {
            "likes": post.get("likeCount", 0),
            "reposts": post.get("repostCount", 0),
            "replies": post.get("replyCount", 0),
            "views": None,
        }
        if any(v for v in result.values() if v):
            console.print(f"[dim]  Bluesky: {result['likes']}L {result['reposts']}R {result['replies']}C[/dim]")
        return result
    except Exception as e:
        console.print(f"[yellow]Bluesky metrics failed for {uri[:60]}: {e}[/yellow]")
        return {"likes": 0, "reposts": 0, "replies": 0, "views": None}


def _fetch_twitter_metrics(tweet_id: str) -> dict:
    """
    Fetch engagement metrics for a tweet via Twitter API v2.

    Uses Tweepy with OAuth 1.0a credentials from .env.
    Basic tier: likes, retweets, replies available. No impressions.
    """
    try:
        import tweepy

        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )

        # Clean tweet_id — handle dict strings like "{'tweet_id': '123'}"
        if isinstance(tweet_id, str) and "tweet_id" in tweet_id:
            import ast
            try:
                parsed = ast.literal_eval(tweet_id)
                tweet_id = parsed.get("tweet_id", tweet_id)
            except (ValueError, SyntaxError):
                pass

        response = client.get_tweet(
            tweet_id,
            tweet_fields=["public_metrics"],
        )

        if response and response.data:
            metrics = response.data.public_metrics or {}
            return {
                "likes": metrics.get("like_count", 0),
                "reposts": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "views": metrics.get("impression_count"),  # May be None on Basic tier
            }

        return {"likes": 0, "reposts": 0, "replies": 0, "views": None}

    except Exception as e:
        console.print(f"[yellow]Twitter metrics failed for {tweet_id}: {e}[/yellow]")
        return {"likes": 0, "reposts": 0, "replies": 0, "views": None}


def _fetch_threads_metrics(thread_id: str) -> dict:
    """
    Fetch engagement metrics for a Threads post via the Graph API.

    Requires threads_manage_insights scope. Gracefully returns zeros
    if the scope is not available.
    """
    try:
        access_token = os.getenv("META_ACCESS_TOKEN")
        if not access_token:
            return {"likes": 0, "reposts": 0, "replies": 0, "views": None}

        # Clean thread_id
        if isinstance(thread_id, str) and "thread_id" in thread_id:
            import ast
            try:
                parsed = ast.literal_eval(thread_id)
                thread_id = parsed.get("thread_id", thread_id)
            except (ValueError, SyntaxError):
                pass

        response = requests.get(
            f"https://graph.threads.net/v1.0/{thread_id}/insights",
            params={
                "metric": "likes,replies,reposts,views",
                "access_token": access_token,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # Parse insights response: {"data": [{"name": "likes", "values": [{"value": N}]}, ...]}
        metrics = {}
        for entry in data.get("data", []):
            name = entry.get("name", "")
            values = entry.get("values", [{}])
            metrics[name] = values[0].get("value", 0) if values else 0

        return {
            "likes": metrics.get("likes", 0),
            "reposts": metrics.get("reposts", 0),
            "replies": metrics.get("replies", 0),
            "views": metrics.get("views", None),
        }

    except Exception as e:
        console.print(f"[yellow]Threads metrics failed for {thread_id}: {e}[/yellow]")
        return {"likes": 0, "reposts": 0, "replies": 0, "views": None}


def _fetch_reels_metrics(media_id: str) -> dict:
    """
    Fetch engagement metrics for an Instagram Reel via the Graph API.

    Two-step approach for resilience:
      1. Try full fields (like_count, comments_count, share_count, views)
         — views requires instagram_manage_insights scope
      2. If that 400s, retry with basic fields only (like_count, comments_count)
         — these work without the insights scope

    This lets us collect engagement data before app verification completes,
    and views appear automatically once the scope is granted.
    """
    try:
        access_token = os.getenv("META_INSTAGRAM_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")
        if not access_token:
            return {"likes": 0, "reposts": 0, "replies": 0, "views": None}

        # Clean media_id — handle "reels:123" prefix and dict strings
        if isinstance(media_id, str):
            if media_id.startswith("reels:"):
                media_id = media_id[len("reels:"):]
            if "media_id" in media_id:
                import ast
                try:
                    parsed = ast.literal_eval(media_id)
                    media_id = parsed.get("media_id", media_id)
                except (ValueError, SyntaxError):
                    pass

        base_url = f"https://graph.instagram.com/v24.0/{media_id}"

        # Step 1: Try full fields (including views — needs insights scope)
        response = requests.get(
            base_url,
            params={
                "fields": "like_count,comments_count,share_count,views",
                "access_token": access_token,
            },
            timeout=10,
        )

        # Step 2: If insights scope missing, fall back to basic fields
        if response.status_code == 400:
            err_data = response.json().get("error", {})
            console.print(
                f"[dim]  Reels insights scope pending — fetching basic metrics only[/dim]"
            )
            response = requests.get(
                base_url,
                params={
                    "fields": "like_count,comments_count",
                    "access_token": access_token,
                },
                timeout=10,
            )
            if response.status_code == 400:
                # Basic fields also failed — token or media_id issue
                err_data = response.json().get("error", {})
                console.print(
                    f"[yellow]Reels basic metrics error: {err_data.get('message', 'unknown')}[/yellow]"
                )
                return {"likes": 0, "reposts": 0, "replies": 0, "views": None}

        response.raise_for_status()
        data = response.json()

        result = {
            "likes": data.get("like_count", 0),
            "reposts": data.get("share_count", 0),
            "replies": data.get("comments_count", 0),
            "views": data.get("views", None),
        }
        if any(v for v in result.values() if v):
            console.print(
                f"[dim]  Reels: {result['likes']}L {result['views'] or '?'}V "
                f"{result['replies']}C {result['reposts']}S[/dim]"
            )
        return result

    except Exception as e:
        console.print(f"[yellow]Reels metrics failed for {media_id}: {e}[/yellow]")
        return {"likes": 0, "reposts": 0, "replies": 0, "views": None}


# Platform fetcher dispatch
_FETCHERS = {
    "bluesky": _fetch_bluesky_metrics,
    "twitter": _fetch_twitter_metrics,
    "threads": _fetch_threads_metrics,
    "reels": _fetch_reels_metrics,
}


# ---------------------------------------------------------------------------
# Core analytics functions
# ---------------------------------------------------------------------------

def fetch_metrics(queue_path: Path, delay_hours: int = 48) -> list[PostMetrics]:
    """
    Fetch engagement metrics for all posted items in a queue.

    Args:
        queue_path: Path to the queue JSON file.
        delay_hours: Only fetch for items posted more than N hours ago
                     (to allow engagement to accumulate).

    Returns:
        List of PostMetrics with engagement data.
    """
    from src.scheduler import WeekQueue

    queue = WeekQueue.load(queue_path)
    cutoff = datetime.now() - timedelta(hours=delay_hours)
    results = []

    posted_items = [
        item for item in queue.items
        if item.status == "posted" and item.post_uri
    ]

    if not posted_items:
        console.print("[yellow]No posted items with URIs found in queue.[/yellow]")
        return results

    console.print(f"[cyan]Fetching metrics for {len(posted_items)} posted items...[/cyan]")

    for item in posted_items:
        # Check if posted long enough ago
        if item.posted_at:
            try:
                posted_time = datetime.fromisoformat(item.posted_at)
                if posted_time > cutoff:
                    console.print(
                        f"  [dim]Skipping {item.platform}/{item.article_title[:30]} "
                        f"(posted < {delay_hours}h ago)[/dim]"
                    )
                    continue
            except ValueError:
                pass  # If we can't parse the date, fetch anyway

        # Fetch metrics from the platform
        fetcher = _FETCHERS.get(item.platform)
        if not fetcher:
            console.print(f"  [dim]No fetcher for platform: {item.platform}[/dim]")
            continue

        raw = fetcher(item.post_uri)

        metrics = PostMetrics(
            queue_item_id=item.id,
            platform=item.platform,
            post_uri=item.post_uri,
            article_title=item.article_title,
            section=item.section,
            is_feature=item.is_feature,
            hook=item.text[:200],  # Store first 200 chars of hook
            posted_at=item.posted_at or "",
            fetched_at=datetime.now().isoformat(),
            likes=raw["likes"],
            reposts=raw["reposts"],
            replies=raw["replies"],
            views=raw.get("views"),
        )
        metrics.compute_score()
        results.append(metrics)

        console.print(
            f"  [{item.platform}] {item.article_title[:35]:35s} "
            f"L:{metrics.likes} R:{metrics.reposts} C:{metrics.replies} "
            f"→ {metrics.engagement_score:.0f}"
        )

    return results


def save_metrics(metrics: list[PostMetrics], path: Path = DEFAULT_METRICS_PATH):
    """
    Append metrics to the cumulative JSON file.

    Does not overwrite existing data — each collection run appends.
    Deduplicates by (queue_item_id, platform) to avoid double-counting
    when re-running collection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_metrics(path)
    existing_keys = {(m.queue_item_id, m.platform) for m in existing}

    new_metrics = [
        m for m in metrics
        if (m.queue_item_id, m.platform) not in existing_keys
    ]

    if not new_metrics:
        console.print("[dim]No new metrics to save (all already collected).[/dim]")
        return

    all_metrics = existing + new_metrics
    with open(path, "w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)

    console.print(
        f"[green]Saved {len(new_metrics)} new metrics "
        f"({len(all_metrics)} total) to {path}[/green]"
    )


def load_metrics(path: Path = DEFAULT_METRICS_PATH) -> list[PostMetrics]:
    """Load all historical metrics from the cumulative file."""
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = json.load(f)
        return [PostMetrics(**d) for d in data]
    except (json.JSONDecodeError, TypeError):
        return []


def get_section_averages(path: Path = DEFAULT_METRICS_PATH) -> dict[str, float]:
    """
    Compute average engagement score per content section.

    Returns:
        {"lived": 15.2, "systems": 12.8, ...}
    """
    metrics = load_metrics(path)
    if not metrics:
        return {}

    section_scores = defaultdict(list)
    for m in metrics:
        section_scores[m.section].append(m.engagement_score)

    return {
        section: sum(scores) / len(scores)
        for section, scores in section_scores.items()
    }


def get_platform_averages(path: Path = DEFAULT_METRICS_PATH) -> dict[str, float]:
    """
    Compute average engagement score per platform.

    Returns:
        {"bluesky": 8.5, "twitter": 12.3, ...}
    """
    metrics = load_metrics(path)
    if not metrics:
        return {}

    platform_scores = defaultdict(list)
    for m in metrics:
        platform_scores[m.platform].append(m.engagement_score)

    return {
        platform: sum(scores) / len(scores)
        for platform, scores in platform_scores.items()
    }


def get_top_posts(path: Path = DEFAULT_METRICS_PATH, limit: int = 10) -> list[PostMetrics]:
    """Get the top-performing posts by engagement score."""
    metrics = load_metrics(path)
    return sorted(metrics, key=lambda m: m.engagement_score, reverse=True)[:limit]


def print_metrics_table(metrics: list[PostMetrics]):
    """Print a Rich table of post metrics."""
    table = Table(title="Post Engagement Metrics", show_lines=False)
    table.add_column("Platform", width=10)
    table.add_column("Article", width=40)
    table.add_column("Section", width=10)
    table.add_column("Likes", justify="right", width=6)
    table.add_column("Reposts", justify="right", width=8)
    table.add_column("Replies", justify="right", width=8)
    table.add_column("Score", justify="right", width=7, style="bold")

    for m in metrics:
        table.add_row(
            m.platform,
            m.article_title[:40],
            m.section,
            str(m.likes),
            str(m.reposts),
            str(m.replies),
            f"{m.engagement_score:.0f}",
        )

    console.print(table)


def print_metrics_report(path: Path = DEFAULT_METRICS_PATH):
    """Print a comprehensive analytics report."""
    metrics = load_metrics(path)
    if not metrics:
        console.print("[yellow]No metrics data yet. Run 'collect-metrics' first.[/yellow]")
        return

    console.print(f"\n[bold cyan]Docket Analytics Report[/bold cyan]")
    console.print(f"[dim]{len(metrics)} data points collected[/dim]\n")

    # Section averages
    section_avgs = get_section_averages(path)
    if section_avgs:
        table = Table(title="Section Performance", show_lines=False)
        table.add_column("Section", width=20)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Posts", justify="right", width=8)

        section_counts = defaultdict(int)
        for m in metrics:
            section_counts[m.section] += 1

        for section, avg in sorted(section_avgs.items(), key=lambda x: x[1], reverse=True):
            table.add_row(section, f"{avg:.1f}", str(section_counts[section]))

        console.print(table)

    # Platform averages
    platform_avgs = get_platform_averages(path)
    if platform_avgs:
        console.print()
        table = Table(title="Platform Performance", show_lines=False)
        table.add_column("Platform", width=15)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Posts", justify="right", width=8)

        platform_counts = defaultdict(int)
        for m in metrics:
            platform_counts[m.platform] += 1

        for platform, avg in sorted(platform_avgs.items(), key=lambda x: x[1], reverse=True):
            table.add_row(platform, f"{avg:.1f}", str(platform_counts[platform]))

        console.print(table)

    # Top posts
    top = get_top_posts(path, limit=10)
    if top:
        console.print()
        console.print("[bold]Top 10 Posts[/bold]")
        print_metrics_table(top)

    # Feature vs News comparison
    feature_scores = [m.engagement_score for m in metrics if m.is_feature]
    news_scores = [m.engagement_score for m in metrics if not m.is_feature]

    if feature_scores and news_scores:
        console.print()
        feat_avg = sum(feature_scores) / len(feature_scores)
        news_avg = sum(news_scores) / len(news_scores)
        console.print(
            f"[bold]Feature vs News:[/bold] "
            f"Features avg {feat_avg:.1f} ({len(feature_scores)} posts) | "
            f"News avg {news_avg:.1f} ({len(news_scores)} posts)"
        )


# ---------------------------------------------------------------------------
# Time-slot performance analysis
# ---------------------------------------------------------------------------

def get_time_slot_performance(
    queue_path: Path,
    metrics: list[PostMetrics] | None = None,
    path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, dict]:
    """
    Analyze engagement by scheduled time slot.

    Cross-references queue items' scheduled_time with collected metrics
    to compute average engagement per time slot.

    Returns:
        {"09:30": {"avg_score": 12.5, "posts": 8, "total_likes": 20, ...}, ...}
    """
    from src.scheduler import WeekQueue

    if metrics is None:
        metrics = load_metrics(path)
    if not metrics:
        return {}

    # Build lookup: queue_item_id → scheduled_time
    try:
        queue = WeekQueue.load(queue_path)
    except Exception:
        return {}

    id_to_slot: dict[str, str] = {}
    for item in queue.items:
        if item.scheduled_time:
            try:
                dt = datetime.fromisoformat(item.scheduled_time)
                slot = dt.strftime("%H:%M")
                id_to_slot[item.id] = slot
            except ValueError:
                pass

    # Aggregate metrics by time slot
    slot_data: dict[str, dict] = defaultdict(lambda: {
        "scores": [], "likes": 0, "reposts": 0, "replies": 0, "count": 0,
    })

    for m in metrics:
        slot = id_to_slot.get(m.queue_item_id)
        if slot:
            d = slot_data[slot]
            d["scores"].append(m.engagement_score)
            d["likes"] += m.likes
            d["reposts"] += m.reposts
            d["replies"] += m.replies
            d["count"] += 1

    # Compute averages
    result = {}
    for slot in sorted(slot_data.keys()):
        d = slot_data[slot]
        if d["count"] > 0:
            result[slot] = {
                "avg_score": sum(d["scores"]) / len(d["scores"]),
                "posts": d["count"],
                "total_likes": d["likes"],
                "total_reposts": d["reposts"],
                "total_replies": d["replies"],
                "best_score": max(d["scores"]) if d["scores"] else 0,
            }

    return result


def get_day_of_week_performance(
    queue_path: Path,
    metrics: list[PostMetrics] | None = None,
    path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, dict]:
    """
    Analyze engagement by day of the week.

    Returns:
        {"Monday": {"avg_score": 12.5, "posts": 8, ...}, ...}
    """
    from src.scheduler import WeekQueue

    if metrics is None:
        metrics = load_metrics(path)
    if not metrics:
        return {}

    try:
        queue = WeekQueue.load(queue_path)
    except Exception:
        return {}

    # Build lookup: queue_item_id → day name
    id_to_day: dict[str, str] = {}
    for item in queue.items:
        id_to_day[item.id] = item.day.capitalize()

    day_data: dict[str, dict] = defaultdict(lambda: {
        "scores": [], "likes": 0, "reposts": 0, "replies": 0, "count": 0,
    })

    for m in metrics:
        day = id_to_day.get(m.queue_item_id)
        if day:
            d = day_data[day]
            d["scores"].append(m.engagement_score)
            d["likes"] += m.likes
            d["reposts"] += m.reposts
            d["replies"] += m.replies
            d["count"] += 1

    result = {}
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in day_order:
        if day in day_data:
            d = day_data[day]
            if d["count"] > 0:
                result[day] = {
                    "avg_score": sum(d["scores"]) / len(d["scores"]),
                    "posts": d["count"],
                    "total_likes": d["likes"],
                    "total_reposts": d["reposts"],
                    "total_replies": d["replies"],
                }

    return result


# ---------------------------------------------------------------------------
# Weekly engagement report
# ---------------------------------------------------------------------------

def generate_weekly_report(
    queue_path: Path,
    config: dict,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict:
    """
    Generate a comprehensive weekly engagement report.

    Collects metrics (if needed), analyzes performance across platforms,
    sections, time slots, and days, then prints a rich terminal report
    and saves a JSON summary.

    Returns the report data dict.
    """
    from src.scheduler import WeekQueue

    queue = WeekQueue.load(queue_path)
    analytics_config = config.get("analytics", {})

    # Collect fresh metrics with zero delay (end-of-week report)
    console.print("[cyan]Collecting latest metrics...[/cyan]")
    metrics = fetch_metrics(queue_path, delay_hours=0)
    if metrics:
        save_metrics(metrics, metrics_path)

    # Load all metrics for this queue
    all_metrics = load_metrics(metrics_path)
    # Filter to only this queue's items
    queue_item_ids = {item.id for item in queue.items}
    week_metrics = [m for m in all_metrics if m.queue_item_id in queue_item_ids]

    if not week_metrics:
        console.print("[yellow]No metrics available for this week's posts.[/yellow]")
        return {}

    # --- Compute report data ---
    report = _build_report_data(week_metrics, queue, queue_path)

    # --- Print the report ---
    _print_weekly_report(report, queue)

    # --- Save report JSON ---
    report_dir = Path(queue_path).parent.parent / "analytics"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"weekly_report_{queue.week_start}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n[green]Report saved to {report_file}[/green]")

    # Send weekly report notification
    try:
        from src.notifier import notify_weekly_report
        notify_weekly_report(report, config=config)
    except Exception:
        pass  # Notifications must never break the pipeline

    return report


def _build_report_data(
    week_metrics: list[PostMetrics],
    queue,
    queue_path: Path,
) -> dict:
    """Build the structured report data dict."""

    total_likes = sum(m.likes for m in week_metrics)
    total_reposts = sum(m.reposts for m in week_metrics)
    total_replies = sum(m.replies for m in week_metrics)
    total_engagement = total_likes + total_reposts + total_replies
    avg_score = sum(m.engagement_score for m in week_metrics) / len(week_metrics) if week_metrics else 0

    # Platform breakdown
    platform_data = defaultdict(lambda: {"likes": 0, "reposts": 0, "replies": 0, "scores": [], "count": 0})
    for m in week_metrics:
        d = platform_data[m.platform]
        d["likes"] += m.likes
        d["reposts"] += m.reposts
        d["replies"] += m.replies
        d["scores"].append(m.engagement_score)
        d["count"] += 1

    platforms = {}
    for platform, d in platform_data.items():
        platforms[platform] = {
            "posts": d["count"],
            "likes": d["likes"],
            "reposts": d["reposts"],
            "replies": d["replies"],
            "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
            "total_engagement": d["likes"] + d["reposts"] + d["replies"],
        }

    # Section breakdown
    section_data = defaultdict(lambda: {"scores": [], "count": 0, "likes": 0, "reposts": 0, "replies": 0})
    for m in week_metrics:
        d = section_data[m.section]
        d["scores"].append(m.engagement_score)
        d["count"] += 1
        d["likes"] += m.likes
        d["reposts"] += m.reposts
        d["replies"] += m.replies

    sections = {}
    for section, d in section_data.items():
        sections[section] = {
            "posts": d["count"],
            "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
            "total_engagement": d["likes"] + d["reposts"] + d["replies"],
        }

    # Time slot breakdown
    time_slots = get_time_slot_performance(queue_path, week_metrics)

    # Day of week breakdown
    day_perf = get_day_of_week_performance(queue_path, week_metrics)

    # Top 5 posts
    sorted_metrics = sorted(week_metrics, key=lambda m: m.engagement_score, reverse=True)
    top_posts = [
        {
            "title": m.article_title,
            "platform": m.platform,
            "section": m.section,
            "likes": m.likes,
            "reposts": m.reposts,
            "replies": m.replies,
            "score": m.engagement_score,
        }
        for m in sorted_metrics[:5]
    ]

    # Bottom 5 (underperformers)
    bottom_posts = [
        {
            "title": m.article_title,
            "platform": m.platform,
            "section": m.section,
            "score": m.engagement_score,
        }
        for m in sorted_metrics[-5:]
    ] if len(sorted_metrics) > 5 else []

    # Feature vs News
    feature_metrics = [m for m in week_metrics if m.is_feature]
    news_metrics = [m for m in week_metrics if not m.is_feature]
    feature_vs_news = {
        "feature_avg": sum(m.engagement_score for m in feature_metrics) / len(feature_metrics) if feature_metrics else 0,
        "feature_count": len(feature_metrics),
        "news_avg": sum(m.engagement_score for m in news_metrics) / len(news_metrics) if news_metrics else 0,
        "news_count": len(news_metrics),
    }

    # Queue completion stats
    total_items = len(queue.items)
    posted = len([i for i in queue.items if i.status == "posted"])
    failed = len([i for i in queue.items if i.status == "failed"])
    pending = len([i for i in queue.items if i.status == "pending"])

    return {
        "week_start": queue.week_start,
        "week_end": queue.week_end,
        "issue_number": queue.issue_number,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_posts_tracked": len(week_metrics),
            "total_likes": total_likes,
            "total_reposts": total_reposts,
            "total_replies": total_replies,
            "total_engagement": total_engagement,
            "avg_engagement_score": avg_score,
        },
        "queue_completion": {
            "total": total_items,
            "posted": posted,
            "failed": failed,
            "pending": pending,
            "completion_rate": (posted / total_items * 100) if total_items > 0 else 0,
        },
        "platforms": platforms,
        "sections": sections,
        "time_slots": time_slots,
        "day_of_week": day_perf,
        "top_posts": top_posts,
        "bottom_posts": bottom_posts,
        "feature_vs_news": feature_vs_news,
    }


def _print_weekly_report(report: dict, queue) -> None:
    """Print a beautifully formatted weekly report to the terminal."""

    summary = report.get("summary", {})
    completion = report.get("queue_completion", {})

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]📊 The Docket — Weekly Engagement Report[/bold cyan]\n"
        f"Issue #{report.get('issue_number', '?')} | "
        f"{report.get('week_start', '')} to {report.get('week_end', '')}",
        border_style="cyan",
    ))

    # Queue completion
    console.print(f"\n[bold]Queue Completion:[/bold] "
                  f"[green]{completion.get('posted', 0)}[/green] posted / "
                  f"{completion.get('total', 0)} total "
                  f"([bold]{completion.get('completion_rate', 0):.0f}%[/bold])")
    if completion.get("failed", 0):
        console.print(f"  [red]{completion['failed']} failed[/red]")
    if completion.get("pending", 0):
        console.print(f"  [yellow]{completion['pending']} still pending[/yellow]")

    # Overall engagement
    console.print(f"\n[bold]Overall Engagement:[/bold]")
    console.print(
        f"  Posts tracked: {summary.get('total_posts_tracked', 0)} | "
        f"Likes: {summary.get('total_likes', 0)} | "
        f"Reposts: {summary.get('total_reposts', 0)} | "
        f"Replies: {summary.get('total_replies', 0)}"
    )
    console.print(
        f"  Total interactions: [bold]{summary.get('total_engagement', 0)}[/bold] | "
        f"Avg score: [bold]{summary.get('avg_engagement_score', 0):.1f}[/bold]"
    )

    # Platform performance table
    platforms = report.get("platforms", {})
    if platforms:
        console.print()
        table = Table(title="Platform Performance", show_lines=False)
        table.add_column("Platform", style="bold", width=12)
        table.add_column("Posts", justify="center", width=7)
        table.add_column("Likes", justify="right", width=7)
        table.add_column("Reposts", justify="right", width=8)
        table.add_column("Replies", justify="right", width=8)
        table.add_column("Avg Score", justify="right", width=10, style="bold")
        table.add_column("Total", justify="right", width=7)

        for platform in sorted(platforms.keys()):
            d = platforms[platform]
            table.add_row(
                platform,
                str(d["posts"]),
                str(d["likes"]),
                str(d["reposts"]),
                str(d["replies"]),
                f"{d['avg_score']:.1f}",
                str(d["total_engagement"]),
            )

        console.print(table)

    # Section performance table
    sections = report.get("sections", {})
    if sections:
        console.print()
        table = Table(title="Section Performance", show_lines=False)
        table.add_column("Section", style="bold", width=15)
        table.add_column("Posts", justify="center", width=7)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Engagement", justify="right", width=12)

        for section, d in sorted(sections.items(), key=lambda x: x[1]["avg_score"], reverse=True):
            table.add_row(
                section,
                str(d["posts"]),
                f"{d['avg_score']:.1f}",
                str(d["total_engagement"]),
            )

        console.print(table)

    # Time slot performance table
    time_slots = report.get("time_slots", {})
    if time_slots:
        console.print()
        table = Table(title="Time Slot Performance", show_lines=False)
        table.add_column("Slot", style="bold", width=8)
        table.add_column("Posts", justify="center", width=7)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Best Score", justify="right", width=11)
        table.add_column("Likes", justify="right", width=7)
        table.add_column("Reposts", justify="right", width=8)
        table.add_column("Replies", justify="right", width=8)

        best_slot = max(time_slots.items(), key=lambda x: x[1]["avg_score"])
        for slot, d in sorted(time_slots.items()):
            slot_label = slot
            if slot == best_slot[0]:
                slot_label = f"[green]{slot} ★[/green]"
            table.add_row(
                slot_label,
                str(d["posts"]),
                f"{d['avg_score']:.1f}",
                f"{d['best_score']:.0f}",
                str(d["total_likes"]),
                str(d["total_reposts"]),
                str(d["total_replies"]),
            )

        console.print(table)

    # Day of week performance
    day_perf = report.get("day_of_week", {})
    if day_perf:
        console.print()
        table = Table(title="Day-of-Week Performance", show_lines=False)
        table.add_column("Day", style="bold", width=12)
        table.add_column("Posts", justify="center", width=7)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Engagement", justify="right", width=12)

        best_day = max(day_perf.items(), key=lambda x: x[1]["avg_score"])
        for day, d in day_perf.items():
            day_label = day
            if day == best_day[0]:
                day_label = f"[green]{day} ★[/green]"
            table.add_row(
                day_label,
                str(d["posts"]),
                f"{d['avg_score']:.1f}",
                str(d["total_likes"] + d["total_reposts"] + d["total_replies"]),
            )

        console.print(table)

    # Top posts
    top_posts = report.get("top_posts", [])
    if top_posts:
        console.print()
        console.print("[bold]🏆 Top 5 Posts:[/bold]")
        for i, p in enumerate(top_posts, 1):
            console.print(
                f"  {i}. [bold]{p['title'][:50]}[/bold]\n"
                f"     [{p['platform']}] {p['section']} — "
                f"L:{p['likes']} R:{p['reposts']} C:{p['replies']} "
                f"→ [bold]{p['score']:.0f}[/bold]"
            )

    # Feature vs News
    fvn = report.get("feature_vs_news", {})
    if fvn.get("feature_count") and fvn.get("news_count"):
        console.print()
        console.print(
            f"[bold]Feature vs News:[/bold] "
            f"Features avg [bold]{fvn['feature_avg']:.1f}[/bold] "
            f"({fvn['feature_count']} posts) | "
            f"News avg [bold]{fvn['news_avg']:.1f}[/bold] "
            f"({fvn['news_count']} posts)"
        )
        winner = "Features" if fvn["feature_avg"] > fvn["news_avg"] else "News"
        console.print(f"  → [cyan]{winner} performed better this week[/cyan]")

    # Actionable insights
    console.print()
    _print_insights(report)


def _print_insights(report: dict) -> None:
    """Generate and print actionable insights from the report data."""

    insights = []

    # Best time slot
    time_slots = report.get("time_slots", {})
    if time_slots:
        best_slot = max(time_slots.items(), key=lambda x: x[1]["avg_score"])
        worst_slot = min(time_slots.items(), key=lambda x: x[1]["avg_score"])
        if best_slot[0] != worst_slot[0]:
            diff = best_slot[1]["avg_score"] - worst_slot[1]["avg_score"]
            insights.append(
                f"Best posting time: [bold]{best_slot[0]}[/bold] "
                f"(avg score {best_slot[1]['avg_score']:.1f} vs "
                f"{worst_slot[0]} at {worst_slot[1]['avg_score']:.1f}, "
                f"+{diff:.0f}% advantage)"
            )

    # Best platform
    platforms = report.get("platforms", {})
    if len(platforms) > 1:
        best_plat = max(platforms.items(), key=lambda x: x[1]["avg_score"])
        insights.append(
            f"Top platform: [bold]{best_plat[0]}[/bold] "
            f"(avg score {best_plat[1]['avg_score']:.1f})"
        )

    # Best section
    sections = report.get("sections", {})
    if sections:
        best_section = max(sections.items(), key=lambda x: x[1]["avg_score"])
        insights.append(
            f"Top section: [bold]{best_section[0]}[/bold] "
            f"(avg score {best_section[1]['avg_score']:.1f})"
        )

    # Best day
    day_perf = report.get("day_of_week", {})
    if len(day_perf) > 1:
        best_day = max(day_perf.items(), key=lambda x: x[1]["avg_score"])
        insights.append(
            f"Best day: [bold]{best_day[0]}[/bold] "
            f"(avg score {best_day[1]['avg_score']:.1f})"
        )

    # Completion rate
    completion = report.get("queue_completion", {})
    rate = completion.get("completion_rate", 0)
    if rate < 100:
        insights.append(
            f"Queue completion at [yellow]{rate:.0f}%[/yellow] — "
            f"{completion.get('failed', 0)} failures, "
            f"{completion.get('pending', 0)} still pending"
        )

    # Token health warnings
    try:
        from src.token_manager import check_token_health
        health = check_token_health()
        for token_name, info in health.items():
            days = info.get("days_remaining")
            label = info.get("label", token_name)
            if days is not None and days <= 7:
                insights.append(
                    f"[red]⚠ {label} token expires in {days} days! "
                    f"Run: refresh-tokens[/red]"
                )
            elif days is not None and days <= 30:
                insights.append(
                    f"[yellow]⚡ {label} token expires in {days} days — "
                    f"refresh soon[/yellow]"
                )
            elif info.get("status") == "expired":
                insights.append(
                    f"[red]🔴 {label} token is EXPIRED! "
                    f"Manual renewal required.[/red]"
                )
    except Exception:
        pass  # Token check is non-critical for report

    if insights:
        console.print(Panel(
            "\n".join(f"  → {insight}" for insight in insights),
            title="[bold]💡 Key Insights[/bold]",
            border_style="cyan",
        ))
    else:
        console.print("[dim]Not enough data yet for actionable insights.[/dim]")
