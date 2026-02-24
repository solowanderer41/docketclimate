"""
Notification dispatcher for The Docket.

Sends alerts via webhook (Slack, Discord, or generic JSON) when:
- Daily posting run has failures
- Weekly engagement report is ready
- Health watchdog detects problems

Notifications must NEVER break the pipeline. All public functions
catch exceptions internally and log warnings instead of raising.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests
import yaml
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent


def _load_notification_config(config: dict | None = None) -> dict:
    """Load notification settings from config dict or config.yaml.

    Returns the ``notifications`` section, or an empty dict if missing.
    """
    if config and "notifications" in config:
        return config["notifications"]

    try:
        with open(PROJECT_ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("notifications", {})
    except Exception:
        return {}


def send_notification(
    title: str,
    message: str,
    level: str = "info",
    config: dict | None = None,
) -> bool:
    """Send a notification via the configured webhook.

    Args:
        title: Short summary (e.g., "Daily Run: 2 failures").
        message: Detail body (markdown supported for Slack/Discord).
        level: Severity — ``"info"`` | ``"warning"`` | ``"error"``.
        config: Full project config dict. If None, reads config.yaml.

    Returns:
        True if sent successfully (or notifications disabled), False on error.
    """
    ncfg = _load_notification_config(config)

    if not ncfg.get("enabled", False):
        return True  # silently skip

    webhook_url = ncfg.get("webhook_url") or os.getenv("NOTIFICATION_WEBHOOK_URL", "")
    if not webhook_url:
        console.print("[dim]Notification skipped: no webhook_url configured.[/dim]")
        return True

    fmt = ncfg.get("format", "generic")
    emoji = {"info": "\u2139\ufe0f", "warning": "\u26a0\ufe0f", "error": "\U0001f6a8"}.get(level, "")
    color = {"info": "#4ecdc4", "warning": "#f2cc8f", "error": "#e07a5f"}.get(level, "#999999")

    try:
        if fmt == "slack":
            payload = {
                "text": f"{emoji} {title}",
                "attachments": [
                    {
                        "color": color,
                        "text": message,
                        "footer": "The Docket Social",
                        "ts": int(datetime.now().timestamp()),
                    }
                ],
            }
        elif fmt == "discord":
            payload = {
                "content": f"{emoji} **{title}**",
                "embeds": [
                    {
                        "description": message,
                        "color": int(color.lstrip("#"), 16),
                        "footer": {"text": "The Docket Social"},
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
            }
        else:
            # Generic JSON POST
            payload = {
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat(),
                "source": "docket-social",
            }

        resp = requests.post(
            webhook_url,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            console.print(
                f"[yellow]Webhook returned {resp.status_code}: "
                f"{resp.text[:100]}[/yellow]"
            )
            return False

        console.print(f"[dim]Notification sent: {title}[/dim]")
        return True

    except requests.Timeout:
        console.print("[yellow]Notification webhook timed out (10s).[/yellow]")
        return False
    except Exception as e:
        console.print(f"[yellow]Notification failed: {e}[/yellow]")
        return False


def notify_daily_summary(
    posted: int,
    failed: int,
    pending: int,
    issue_number: int | None = None,
    failed_items: list | None = None,
    config: dict | None = None,
) -> None:
    """Send notification after daily posting run (only when failures exist).

    Args:
        posted: Number of items successfully posted.
        failed: Number of items that failed.
        pending: Number of items still pending (later time slots).
        issue_number: Current issue number.
        failed_items: List of failed QueueItem-like objects (with .id, .platform,
            .article_title, .last_error attributes).
        config: Full project config dict.
    """
    ncfg = _load_notification_config(config)
    if not ncfg.get("notify_on_failure", True):
        return

    if failed == 0:
        return  # Only notify on failures

    today = datetime.now().strftime("%A, %B %d")
    issue_str = f" (Issue #{issue_number})" if issue_number else ""
    title = f"Daily Run{issue_str}: {failed} failure{'s' if failed != 1 else ''}"

    lines = [
        f"*{today}*",
        f"Posted: {posted} | Failed: {failed} | Pending: {pending}",
        "",
    ]

    if failed_items:
        lines.append("*Failed items:*")
        for item in failed_items[:10]:
            platform = getattr(item, "platform", "?")
            article = getattr(item, "article_title", "?")[:40]
            error = getattr(item, "last_error", "unknown")
            if error and len(error) > 80:
                error = error[:80] + "..."
            lines.append(f"  \u2022 `{platform}` {article} \u2014 {error}")

    message = "\n".join(lines)

    try:
        send_notification(title, message, level="error", config=config)
    except Exception as e:
        console.print(f"[yellow]Failed to send daily summary notification: {e}[/yellow]")


def notify_weekly_report(
    report: dict,
    config: dict | None = None,
) -> None:
    """Send notification when weekly engagement report is generated.

    Args:
        report: The weekly report dict (with keys like ``total_engagement``,
            ``platform_breakdown``, ``top_post``, etc.).
        config: Full project config dict.
    """
    ncfg = _load_notification_config(config)
    if not ncfg.get("notify_weekly_report", True):
        return

    week_start = report.get("week_start", "?")
    total = report.get("total_engagement", 0)
    platforms = report.get("platform_breakdown", {})

    title = f"Weekly Report: {week_start}"
    lines = [
        f"*Total engagement:* {total:.0f}" if isinstance(total, (int, float)) else f"*Total engagement:* {total}",
        "",
    ]

    if platforms:
        lines.append("*By platform:*")
        for pname, pdata in platforms.items():
            eng = pdata.get("total_engagement", 0) if isinstance(pdata, dict) else pdata
            lines.append(f"  \u2022 {pname}: {eng}")

    top_post = report.get("top_post", {})
    if top_post:
        top_title = top_post.get("article_title", "?")[:40]
        top_platform = top_post.get("platform", "?")
        top_score = top_post.get("engagement_score", 0)
        lines.append(f"\n*Top post:* {top_title} ({top_platform}, score: {top_score})")

    message = "\n".join(lines)

    try:
        send_notification(title, message, level="info", config=config)
    except Exception as e:
        console.print(f"[yellow]Failed to send weekly report notification: {e}[/yellow]")


def notify_health_alert(
    checks_failed: list[dict],
    config: dict | None = None,
) -> None:
    """Send notification when health watchdog finds problems.

    Args:
        checks_failed: List of dicts with keys ``check_name``, ``status``,
            ``detail``.
        config: Full project config dict.
    """
    ncfg = _load_notification_config(config)
    if not ncfg.get("notify_health_alert", True):
        return

    if not checks_failed:
        return

    has_fail = any(c.get("status") == "fail" for c in checks_failed)
    level = "error" if has_fail else "warning"
    title = f"Health Check: {len(checks_failed)} issue{'s' if len(checks_failed) != 1 else ''}"

    lines = []
    for check in checks_failed:
        icon = "\U0001f534" if check.get("status") == "fail" else "\U0001f7e1"
        name = check.get("name", "?")
        detail = check.get("detail", "")
        lines.append(f"  {icon} *{name}*: {detail}")

    message = "\n".join(lines)

    try:
        send_notification(title, message, level=level, config=config)
    except Exception as e:
        console.print(f"[yellow]Failed to send health alert notification: {e}[/yellow]")
