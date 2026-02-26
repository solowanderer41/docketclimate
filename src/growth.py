"""
Growth Tracking for The Docket.

Tracks follower counts across all platforms via daily snapshots.
Computes growth rates, detects stalls, and checks whether the
cold-start phase can be graduated.

Platform APIs:
    - Bluesky: getProfile → followersCount (public, no auth)
    - Twitter: user lookup → public_metrics.followers_count
    - Threads: GET /me → threads_followers_count (requires auth)
    - Instagram: GET /me → followers_count (requires auth)

Usage:
    python -m src.main growth-snapshot          # fetch + save today's counts
    python -m src.main growth-snapshot --dry     # preview without saving
"""

import json
import os
from dataclasses import dataclass, asdict
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
GROWTH_PATH = PROJECT_ROOT / "analytics" / "growth.json"


@dataclass
class FollowerSnapshot:
    """A single follower count observation."""
    date: str             # ISO date string (YYYY-MM-DD)
    platform: str
    followers: int


# ── Platform-specific follower fetchers ───────────────────────────────────


def _fetch_bluesky_followers() -> int | None:
    """Fetch Bluesky follower count via public AT Protocol API."""
    try:
        handle = os.getenv("BLUESKY_HANDLE")
        if not handle:
            console.print("[dim]BLUESKY_HANDLE not set — skipping Bluesky[/dim]")
            return None

        response = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile",
            params={"actor": handle},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("followersCount", 0)

    except Exception as e:
        console.print(f"[yellow]Bluesky follower fetch failed: {e}[/yellow]")
        return None


def _fetch_twitter_followers() -> int | None:
    """Fetch Twitter follower count via API v2."""
    try:
        import tweepy

        client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )

        response = client.get_me(user_fields=["public_metrics"])
        if response and response.data:
            metrics = response.data.public_metrics or {}
            return metrics.get("followers_count", 0)
        return None

    except Exception as e:
        console.print(f"[yellow]Twitter follower fetch failed: {e}[/yellow]")
        return None


def _fetch_threads_followers() -> int | None:
    """Fetch Threads follower count via Graph API."""
    try:
        access_token = os.getenv("META_ACCESS_TOKEN")
        if not access_token:
            return None

        user_id = os.getenv("THREADS_USER_ID")
        if not user_id:
            # Try to get user ID from /me
            me_response = requests.get(
                "https://graph.threads.net/v1.0/me",
                params={"fields": "id", "access_token": access_token},
                timeout=10,
            )
            if me_response.ok:
                user_id = me_response.json().get("id")

        if not user_id:
            return None

        response = requests.get(
            f"https://graph.threads.net/v1.0/{user_id}",
            params={
                "fields": "threads_followers_count",
                "access_token": access_token,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("threads_followers_count")

    except Exception as e:
        console.print(f"[yellow]Threads follower fetch failed: {e}[/yellow]")
        return None


def _fetch_instagram_followers() -> int | None:
    """Fetch Instagram follower count via Graph API."""
    try:
        access_token = os.getenv("META_ACCESS_TOKEN")
        ig_user_id = os.getenv("INSTAGRAM_USER_ID")

        if not access_token or not ig_user_id:
            return None

        response = requests.get(
            f"https://graph.instagram.com/v21.0/{ig_user_id}",
            params={
                "fields": "followers_count",
                "access_token": access_token,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("followers_count")

    except Exception as e:
        console.print(f"[yellow]Instagram follower fetch failed: {e}[/yellow]")
        return None


# ── Snapshot management ──────────────────────────────────────────────────


_PLATFORM_FETCHERS = {
    "bluesky": _fetch_bluesky_followers,
    "twitter": _fetch_twitter_followers,
    "threads": _fetch_threads_followers,
    "instagram": _fetch_instagram_followers,
}


def load_snapshots(path: Path = GROWTH_PATH) -> list[FollowerSnapshot]:
    """Load all historical follower snapshots."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return [FollowerSnapshot(**d) for d in data]
    except Exception:
        return []


def save_snapshots(snapshots: list[FollowerSnapshot], path: Path = GROWTH_PATH):
    """Save all snapshots to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(s) for s in snapshots], f, indent=2)


def take_snapshot(dry_run: bool = False) -> list[FollowerSnapshot]:
    """
    Fetch current follower counts from all platforms and save a daily snapshot.

    Returns the new snapshots taken (one per platform that responded).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    existing = load_snapshots()

    # Check if we already have today's data
    today_platforms = {s.platform for s in existing if s.date == today}

    new_snapshots = []
    for platform, fetcher in _PLATFORM_FETCHERS.items():
        if platform in today_platforms:
            console.print(f"[dim]{platform}: already snapped today[/dim]")
            continue

        count = fetcher()
        if count is not None:
            snap = FollowerSnapshot(date=today, platform=platform, followers=count)
            new_snapshots.append(snap)
            console.print(f"  [green]{platform}: {count} followers[/green]")
        else:
            console.print(f"  [dim]{platform}: no data[/dim]")

    if new_snapshots and not dry_run:
        all_snapshots = existing + new_snapshots
        save_snapshots(all_snapshots)
        console.print(f"[green]Saved {len(new_snapshots)} new snapshots to {GROWTH_PATH}[/green]")

    return new_snapshots


# ── Growth analysis ──────────────────────────────────────────────────────


def get_growth_rate(
    platform: str,
    days: int = 7,
    path: Path = GROWTH_PATH,
) -> dict | None:
    """
    Compute follower growth rate over the last N days for a platform.

    Returns:
        {followers_start, followers_end, net_gain, daily_rate, pct_change}
        or None if insufficient data.
    """
    snapshots = load_snapshots(path)
    platform_snaps = [s for s in snapshots if s.platform == platform]
    platform_snaps.sort(key=lambda s: s.date)

    if len(platform_snaps) < 2:
        return None

    end = platform_snaps[-1]
    cutoff_date = (datetime.strptime(end.date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")

    # Find the snapshot closest to the cutoff date
    start = None
    for s in platform_snaps:
        if s.date <= cutoff_date:
            start = s
    if start is None:
        start = platform_snaps[0]

    actual_days = max(
        (datetime.strptime(end.date, "%Y-%m-%d") - datetime.strptime(start.date, "%Y-%m-%d")).days,
        1,
    )

    net_gain = end.followers - start.followers
    daily_rate = net_gain / actual_days
    pct_change = (net_gain / start.followers * 100) if start.followers > 0 else 0

    return {
        "followers_start": start.followers,
        "followers_end": end.followers,
        "net_gain": net_gain,
        "daily_rate": round(daily_rate, 2),
        "pct_change": round(pct_change, 1),
        "days_measured": actual_days,
    }


def check_stalls(stall_days: int = 7, path: Path = GROWTH_PATH) -> list[dict]:
    """
    Detect platforms where follower count has been flat for N+ days.

    Returns list of {platform, followers, days_flat} for stalled platforms.
    """
    snapshots = load_snapshots(path)
    stalled = []

    platforms = set(s.platform for s in snapshots)
    for platform in platforms:
        plat_snaps = sorted(
            [s for s in snapshots if s.platform == platform],
            key=lambda s: s.date,
        )

        if len(plat_snaps) < stall_days:
            continue

        recent = plat_snaps[-stall_days:]
        counts = [s.followers for s in recent]

        if len(set(counts)) == 1:
            # All the same count — stalled
            stalled.append({
                "platform": platform,
                "followers": counts[0],
                "days_flat": stall_days,
            })

    return stalled


def check_cold_start_graduation(
    config: dict,
    threshold: int = 50,
    path: Path = GROWTH_PATH,
) -> dict:
    """
    Check if all platforms have passed the cold-start follower threshold.

    Returns:
        {ready: bool, platforms: {name: {followers, above_threshold}}, recommendation: str}
    """
    cold_start = config.get("cold_start", {})
    if not cold_start.get("enabled", False):
        return {"ready": False, "recommendation": "Cold start already disabled."}

    snapshots = load_snapshots(path)

    # Get the latest count per platform
    latest: dict[str, int] = {}
    for s in sorted(snapshots, key=lambda s: s.date):
        latest[s.platform] = s.followers

    if not latest:
        return {
            "ready": False,
            "recommendation": "No follower data yet. Run growth-snapshot first.",
        }

    platforms_status = {}
    all_above = True
    for platform, count in latest.items():
        above = count >= threshold
        if not above:
            all_above = False
        platforms_status[platform] = {
            "followers": count,
            "above_threshold": above,
        }

    if all_above:
        recommendation = (
            f"All platforms exceed {threshold} followers. "
            f"Consider setting cold_start.enabled: false in config.yaml "
            f"to unlock full posting frequency."
        )
    else:
        below = [p for p, d in platforms_status.items() if not d["above_threshold"]]
        recommendation = (
            f"Keep cold_start enabled — {', '.join(below)} "
            f"{'is' if len(below) == 1 else 'are'} still below {threshold} followers."
        )

    return {
        "ready": all_above,
        "platforms": platforms_status,
        "recommendation": recommendation,
    }


# ── Display ──────────────────────────────────────────────────────────────


def print_growth_summary(path: Path = GROWTH_PATH, config: dict | None = None):
    """Print a Rich-formatted growth summary to the terminal."""
    snapshots = load_snapshots(path)
    if not snapshots:
        console.print("[yellow]No growth data yet. Run 'growth-snapshot' to start tracking.[/yellow]")
        return

    # Latest counts
    latest: dict[str, FollowerSnapshot] = {}
    for s in sorted(snapshots, key=lambda s: s.date):
        latest[s.platform] = s

    table = Table(title="Follower Growth Summary", show_lines=False)
    table.add_column("Platform", style="bold", width=12)
    table.add_column("Followers", justify="right", width=10)
    table.add_column("7d Change", justify="right", width=10)
    table.add_column("Daily Rate", justify="right", width=10)
    table.add_column("% Change", justify="right", width=9)

    for platform in sorted(latest.keys()):
        snap = latest[platform]
        rate = get_growth_rate(platform, days=7, path=path)

        followers = str(snap.followers)

        if rate:
            net = rate["net_gain"]
            net_str = f"+{net}" if net >= 0 else str(net)
            net_color = "green" if net > 0 else ("red" if net < 0 else "dim")
            daily = f"{rate['daily_rate']:+.1f}/d"
            pct = f"{rate['pct_change']:+.1f}%"
        else:
            net_str = "—"
            net_color = "dim"
            daily = "—"
            pct = "—"

        table.add_row(
            platform,
            followers,
            f"[{net_color}]{net_str}[/{net_color}]",
            daily,
            pct,
        )

    console.print(table)

    # Check for stalls
    stalled = check_stalls(path=path)
    if stalled:
        for s in stalled:
            console.print(
                f"\n[yellow]⚠ {s['platform']} flat at {s['followers']} "
                f"for {s['days_flat']}+ days — consider adjusting content mix[/yellow]"
            )

    # Cold-start graduation check
    if config:
        grad = check_cold_start_graduation(config, path=path)
        if grad.get("ready"):
            console.print(f"\n[bold green]🎓 {grad['recommendation']}[/bold green]")
        elif grad.get("platforms"):
            below = [
                f"{p} ({d['followers']})"
                for p, d in grad["platforms"].items()
                if not d["above_threshold"]
            ]
            if below:
                console.print(
                    f"\n[dim]Cold-start: {', '.join(below)} still below 50 followers[/dim]"
                )
