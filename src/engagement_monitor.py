"""
Engagement Monitor for The Docket.

Monitors replies, mentions, and conversations across all platforms.
Designed to run periodically (every 4 hours during posting hours) to
surface high-priority engagement events that deserve a response.

Platform APIs:
    - Bluesky: getPostThread (public, depth=6 for replies)
    - Twitter: search conversation_id:{tweet_id}
    - Threads: GET /{thread_id}/replies via Graph API
    - Instagram: GET /{media_id}/comments via Graph API

Usage:
    python -m src.main check-engagement              # check last 24h
    python -m src.main check-engagement --hours 48   # check last 48h
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from dotenv import load_dotenv

load_dotenv(override=True)

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_DIR = PROJECT_ROOT / "queue"


@dataclass
class EngagementEvent:
    """A reply, mention, or quote-post directed at one of our posts."""
    platform: str
    post_id: str           # our post's platform-specific ID
    event_type: str        # "reply", "mention", "quote"
    author: str            # who replied
    text: str              # what they said
    timestamp: str         # ISO timestamp (or best approximation)
    our_post_text: str     # context: what they're replying to (truncated)
    priority: str          # "high", "medium", "low"
    article_title: str = ""
    queue_item_id: str = ""
    direct_link: str = ""  # link to the conversation


def _classify_priority(text: str, author: str = "") -> str:
    """
    Classify an engagement event's priority.

    High: contains a question, mentions us, substantive (>50 chars)
    Medium: moderate length (>30 chars), positive sentiment
    Low: short, emoji-only, generic
    """
    if not text:
        return "low"

    clean = text.strip()

    # High: questions or substantive replies
    if "?" in clean:
        return "high"
    if len(clean) > 80:
        return "high"

    # Medium: moderate-length meaningful reply
    if len(clean) > 30:
        # Check for positive engagement keywords
        positive_markers = [
            "thank", "great", "amazing", "important", "agree",
            "exactly", "this", "wow", "need", "share",
            "powerful", "heartbreaking", "terrifying", "crucial",
        ]
        if any(m in clean.lower() for m in positive_markers):
            return "medium"
        return "medium"

    # Low: short, emoji-only, etc.
    if len(clean) < 10:
        return "low"
    # Check if mostly emojis
    non_emoji = re.sub(r'[\U00010000-\U0010ffff]', '', clean, flags=re.UNICODE)
    if len(non_emoji.strip()) < 5:
        return "low"

    return "low"


# ── Platform-specific reply fetchers ──────────────────────────────────────


def _fetch_bluesky_replies(uri: str, our_text: str = "", article_title: str = "",
                           queue_item_id: str = "") -> list[EngagementEvent]:
    """Fetch replies to a Bluesky post using the public AT Protocol API."""
    events = []
    try:
        # Fix malformed URIs
        if uri.startswith("{") and "uri" in uri:
            import ast
            try:
                parsed = ast.literal_eval(uri)
                uri = parsed.get("uri", uri)
            except (ValueError, SyntaxError):
                pass

        response = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread",
            params={"uri": uri, "depth": 6, "parentHeight": 0},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        thread = data.get("thread", {})
        replies = thread.get("replies", [])

        for reply_thread in replies:
            reply_post = reply_thread.get("post", {})
            author_info = reply_post.get("author", {})
            record = reply_post.get("record", {})

            author = author_info.get("handle", "unknown")
            text = record.get("text", "")
            created = record.get("createdAt", "")

            # Build direct link
            reply_uri = reply_post.get("uri", "")
            rkey = reply_uri.split("/")[-1] if "/" in reply_uri else ""
            direct_link = f"https://bsky.app/profile/{author}/post/{rkey}" if rkey else ""

            events.append(EngagementEvent(
                platform="bluesky",
                post_id=uri,
                event_type="reply",
                author=author,
                text=text,
                timestamp=created,
                our_post_text=our_text[:120],
                priority=_classify_priority(text, author),
                article_title=article_title,
                queue_item_id=queue_item_id,
                direct_link=direct_link,
            ))

    except Exception as e:
        console.print(f"[dim]Bluesky reply fetch failed: {e}[/dim]")

    return events


def _fetch_twitter_replies(tweet_id: str, our_text: str = "", article_title: str = "",
                           queue_item_id: str = "") -> list[EngagementEvent]:
    """Fetch replies to a tweet using Twitter API v2 search."""
    events = []
    try:
        import tweepy

        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )

        # Clean tweet_id
        if isinstance(tweet_id, str) and "tweet_id" in tweet_id:
            import ast
            try:
                parsed = ast.literal_eval(tweet_id)
                tweet_id = parsed.get("tweet_id", tweet_id)
            except (ValueError, SyntaxError):
                pass

        # Search for replies in conversation
        response = client.search_recent_tweets(
            query=f"conversation_id:{tweet_id}",
            tweet_fields=["created_at", "author_id", "in_reply_to_user_id"],
            user_fields=["username"],
            expansions=["author_id"],
            max_results=20,
        )

        if not response or not response.data:
            return events

        # Build user lookup
        users = {}
        if response.includes and "users" in response.includes:
            for u in response.includes["users"]:
                users[str(u.id)] = u.username

        for tweet in response.data:
            author = users.get(str(tweet.author_id), f"user_{tweet.author_id}")
            text = tweet.text or ""
            created = tweet.created_at.isoformat() if tweet.created_at else ""

            events.append(EngagementEvent(
                platform="twitter",
                post_id=str(tweet_id),
                event_type="reply",
                author=author,
                text=text,
                timestamp=created,
                our_post_text=our_text[:120],
                priority=_classify_priority(text, author),
                article_title=article_title,
                queue_item_id=queue_item_id,
                direct_link=f"https://twitter.com/{author}/status/{tweet.id}",
            ))

    except Exception as e:
        console.print(f"[dim]Twitter reply fetch failed: {e}[/dim]")

    return events


def _fetch_threads_replies(thread_id: str, our_text: str = "", article_title: str = "",
                           queue_item_id: str = "") -> list[EngagementEvent]:
    """Fetch replies to a Threads post via the Graph API."""
    events = []
    try:
        access_token = os.getenv("META_ACCESS_TOKEN")
        if not access_token:
            return events

        # Clean thread_id
        if isinstance(thread_id, str) and "thread_id" in thread_id:
            import ast
            try:
                parsed = ast.literal_eval(thread_id)
                thread_id = parsed.get("thread_id", thread_id)
            except (ValueError, SyntaxError):
                pass

        response = requests.get(
            f"https://graph.threads.net/v1.0/{thread_id}/replies",
            params={
                "fields": "id,text,username,timestamp",
                "access_token": access_token,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        for reply in data.get("data", []):
            author = reply.get("username", "unknown")
            text = reply.get("text", "")
            timestamp = reply.get("timestamp", "")

            events.append(EngagementEvent(
                platform="threads",
                post_id=str(thread_id),
                event_type="reply",
                author=author,
                text=text,
                timestamp=timestamp,
                our_post_text=our_text[:120],
                priority=_classify_priority(text, author),
                article_title=article_title,
                queue_item_id=queue_item_id,
                direct_link="",  # Threads doesn't have stable reply URLs
            ))

    except Exception as e:
        console.print(f"[dim]Threads reply fetch failed: {e}[/dim]")

    return events


def _fetch_reels_comments(media_id: str, our_text: str = "", article_title: str = "",
                          queue_item_id: str = "") -> list[EngagementEvent]:
    """Fetch comments on an Instagram Reel via the Graph API."""
    events = []
    try:
        access_token = os.getenv("META_ACCESS_TOKEN")
        if not access_token:
            return events

        # Clean media_id
        if isinstance(media_id, str) and "media_id" in media_id:
            import ast
            try:
                parsed = ast.literal_eval(media_id)
                media_id = parsed.get("media_id", media_id)
            except (ValueError, SyntaxError):
                pass

        response = requests.get(
            f"https://graph.instagram.com/v21.0/{media_id}/comments",
            params={
                "fields": "id,text,username,timestamp",
                "access_token": access_token,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        for comment in data.get("data", []):
            author = comment.get("username", "unknown")
            text = comment.get("text", "")
            timestamp = comment.get("timestamp", "")

            events.append(EngagementEvent(
                platform="reels",
                post_id=str(media_id),
                event_type="reply",
                author=author,
                text=text,
                timestamp=timestamp,
                our_post_text=our_text[:120],
                priority=_classify_priority(text, author),
                article_title=article_title,
                queue_item_id=queue_item_id,
                direct_link="",
            ))

    except Exception as e:
        console.print(f"[dim]Reels comment fetch failed: {e}[/dim]")

    return events


# ── Main check function ──────────────────────────────────────────────────


_PLATFORM_FETCHERS = {
    "bluesky": _fetch_bluesky_replies,
    "twitter": _fetch_twitter_replies,
    "threads": _fetch_threads_replies,
    "reels": _fetch_reels_comments,
}


def check_recent_engagement(hours: int = 24) -> list[EngagementEvent]:
    """
    Check for replies/comments on all posts from the last N hours.

    Scans the active queue for posted items within the time window,
    fetches replies from each platform, and returns sorted by priority.
    """
    from src.scheduler import find_active_queue, WeekQueue
    from zoneinfo import ZoneInfo

    events: list[EngagementEvent] = []

    # Find queue
    queue_path = find_active_queue(QUEUE_DIR)
    if not queue_path:
        console.print("[yellow]No active queue found.[/yellow]")
        return events

    queue = WeekQueue.load(queue_path)

    # Filter to recently posted items
    cutoff = datetime.now().astimezone() - timedelta(hours=hours)
    recent_items = []
    for item in queue.items:
        if item.status != "posted" or not item.post_uri:
            continue
        if item.posted_at:
            try:
                posted_dt = datetime.fromisoformat(item.posted_at)
                if posted_dt.tzinfo is None:
                    posted_dt = posted_dt.astimezone()
                if posted_dt < cutoff:
                    continue
            except (ValueError, TypeError):
                pass  # Can't parse date, include it anyway
        recent_items.append(item)

    if not recent_items:
        console.print(f"[dim]No posts in the last {hours}h to check.[/dim]")
        return events

    console.print(f"[cyan]Checking {len(recent_items)} posts from the last {hours}h...[/cyan]")

    for item in recent_items:
        fetcher = _PLATFORM_FETCHERS.get(item.platform)
        if not fetcher:
            continue

        item_events = fetcher(
            item.post_uri,
            our_text=item.text,
            article_title=item.article_title,
            queue_item_id=item.id,
        )
        events.extend(item_events)

    # Sort: high → medium → low, then by timestamp descending
    priority_order = {"high": 0, "medium": 1, "low": 2}
    events.sort(key=lambda e: (priority_order.get(e.priority, 3), e.timestamp), reverse=False)
    # Actually sort high first, then within priority by newest first
    events.sort(key=lambda e: (priority_order.get(e.priority, 3), ""), reverse=False)

    return events


def print_engagement_table(events: list[EngagementEvent]) -> None:
    """Print engagement events as a Rich table."""
    if not events:
        console.print("[dim]No engagement events found.[/dim]")
        return

    table = Table(title=f"Engagement Events ({len(events)} total)", show_lines=True)
    table.add_column("Pri", width=4, justify="center")
    table.add_column("Platform", width=10)
    table.add_column("Author", width=18)
    table.add_column("Reply", width=50)
    table.add_column("Article", width=25)

    priority_styles = {
        "high": "[bold red]HIGH[/bold red]",
        "medium": "[yellow]MED[/yellow]",
        "low": "[dim]LOW[/dim]",
    }

    for event in events[:30]:  # Cap display at 30
        table.add_row(
            priority_styles.get(event.priority, "[dim]?[/dim]"),
            event.platform,
            f"@{event.author[:16]}",
            event.text[:48] + ("..." if len(event.text) > 48 else ""),
            event.article_title[:23] + (".." if len(event.article_title) > 23 else ""),
        )

    console.print(table)

    # Summary
    high = sum(1 for e in events if e.priority == "high")
    med = sum(1 for e in events if e.priority == "medium")
    low = sum(1 for e in events if e.priority == "low")
    console.print(f"\n[bold]{high}[/bold] high, [bold]{med}[/bold] medium, [bold]{low}[/bold] low priority")

    if high > 0:
        console.print("\n[bold red]High-priority replies need attention:[/bold red]")
        for e in events:
            if e.priority == "high":
                link_str = f" — {e.direct_link}" if e.direct_link else ""
                console.print(
                    f"  [{e.platform}] @{e.author}: \"{e.text[:80]}\""
                    f"{link_str}"
                )


def notify_high_priority(events: list[EngagementEvent], config: dict) -> None:
    """Send Slack notification for high-priority engagement events."""
    high_events = [e for e in events if e.priority == "high"]
    if not high_events:
        return

    try:
        from src.notifier import send_webhook

        lines = [f"*{len(high_events)} high-priority replies need attention:*\n"]
        for e in high_events[:5]:
            link_str = f" — <{e.direct_link}|View>" if e.direct_link else ""
            lines.append(
                f"• [{e.platform}] @{e.author}: \"{e.text[:100]}\"{link_str}"
            )

        if len(high_events) > 5:
            lines.append(f"\n_...and {len(high_events) - 5} more_")

        message = "\n".join(lines)
        send_webhook(message, config=config)
        console.print(f"[green]Slack notification sent for {len(high_events)} high-priority events.[/green]")
    except Exception as e:
        console.print(f"[yellow]Slack notification failed: {e}[/yellow]")
