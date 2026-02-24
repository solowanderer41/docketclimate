"""
Health watchdog for The Docket.

Runs lightweight checks on the pipeline and reports problems:
- Did today's queue items get posted?
- Are Meta tokens still valid?
- Does an active queue exist with future items?
- Has launchd run recently?
- Is disk space adequate?

All checks return a consistent structure:
    {"name": str, "status": "pass"|"warn"|"fail", "detail": str}

The watchdog never raises exceptions — every check catches errors
internally and returns a fail/warn status instead.
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
QUEUE_DIR = PROJECT_ROOT / "queue"
LOG_DIR = PROJECT_ROOT / "logs"


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def _check_posts_today(config: dict) -> dict:
    """Check whether today's queue items were posted successfully."""
    try:
        from src.scheduler import find_active_queue, WeekQueue
        from zoneinfo import ZoneInfo

        tz_name = config.get("schedule", {}).get("timezone", "America/Los_Angeles")
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        today_date = now.strftime("%Y-%m-%d")
        today_day = now.strftime("%A").lower()

        queue_path = find_active_queue(QUEUE_DIR)
        if not queue_path:
            return {
                "name": "Posts today",
                "status": "warn",
                "detail": "No active queue found",
            }

        queue = WeekQueue.load(queue_path)
        today_items = [
            i for i in queue.items
            if (i.date == today_date) or
               (i.day and i.day.lower() == today_day)
        ]

        if not today_items:
            return {
                "name": "Posts today",
                "status": "pass",
                "detail": "No items scheduled for today",
            }

        posted = sum(1 for i in today_items if i.status == "posted")
        failed = sum(1 for i in today_items if i.status == "failed")
        pending = sum(1 for i in today_items if i.status == "pending")
        total = len(today_items)

        if failed > 0:
            return {
                "name": "Posts today",
                "status": "fail",
                "detail": f"{failed}/{total} failed, {posted} posted, {pending} pending",
            }
        elif pending > 0 and posted == 0:
            # All still pending — might be early in the day
            if now.hour < 10:
                return {
                    "name": "Posts today",
                    "status": "pass",
                    "detail": f"{pending} pending (before first slot)",
                }
            return {
                "name": "Posts today",
                "status": "warn",
                "detail": f"{pending} still pending, {posted} posted — may be stuck",
            }
        else:
            return {
                "name": "Posts today",
                "status": "pass",
                "detail": f"{posted}/{total} posted, {pending} pending",
            }

    except Exception as e:
        return {
            "name": "Posts today",
            "status": "fail",
            "detail": f"Check error: {e}",
        }


def _check_token_health() -> dict:
    """Check Meta token validity and expiration."""
    try:
        from src.token_manager import check_token_health

        health = check_token_health()

        issues = []
        worst = "pass"

        for token_name, info in health.items():
            status = info.get("status", "unknown")
            days = info.get("days_remaining")
            label = info.get("label", token_name)

            if status == "expired":
                issues.append(f"{label}: EXPIRED")
                worst = "fail"
            elif status == "missing":
                # Missing tokens are OK if platform isn't enabled
                pass
            elif days is not None and days < 3:
                issues.append(f"{label}: {days}d left")
                worst = "fail"
            elif days is not None and days < 14:
                issues.append(f"{label}: {days}d left")
                if worst != "fail":
                    worst = "warn"

        if not issues:
            return {
                "name": "Token health",
                "status": "pass",
                "detail": "All tokens valid",
            }

        return {
            "name": "Token health",
            "status": worst,
            "detail": "; ".join(issues),
        }

    except Exception as e:
        return {
            "name": "Token health",
            "status": "warn",
            "detail": f"Check error: {e}",
        }


def _check_active_queue() -> dict:
    """Check that an active queue with future items exists."""
    try:
        from src.scheduler import find_active_queue, WeekQueue

        queue_path = find_active_queue(QUEUE_DIR)
        if not queue_path:
            return {
                "name": "Active queue",
                "status": "fail",
                "detail": "No active queue found — run 'schedule' to create one",
            }

        queue = WeekQueue.load(queue_path)
        pending = sum(1 for i in queue.items if i.status == "pending")
        total = len(queue.items)

        if pending == 0:
            return {
                "name": "Active queue",
                "status": "warn",
                "detail": f"Queue exhausted ({total} items, 0 pending)",
            }

        return {
            "name": "Active queue",
            "status": "pass",
            "detail": f"{pending}/{total} items pending ({queue_path.name})",
        }

    except Exception as e:
        return {
            "name": "Active queue",
            "status": "fail",
            "detail": f"Check error: {e}",
        }


def _check_last_run() -> dict:
    """Check that launchd has run recently by examining log file mtime."""
    try:
        log_files = [
            LOG_DIR / "launchd.stdout.log",
            LOG_DIR / "launchd.stderr.log",
        ]

        # Also check daily log files
        today = datetime.now()
        for days_back in range(3):
            dt = today - timedelta(days=days_back)
            log_files.append(LOG_DIR / f"{dt.strftime('%Y-%m-%d')}.log")

        newest_mtime = None
        newest_file = None
        for lf in log_files:
            if lf.exists():
                mtime = lf.stat().st_mtime
                if newest_mtime is None or mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_file = lf

        if newest_mtime is None:
            return {
                "name": "Last run",
                "status": "warn",
                "detail": "No log files found — pipeline may not have run yet",
            }

        last_dt = datetime.fromtimestamp(newest_mtime)
        hours_ago = (datetime.now() - last_dt).total_seconds() / 3600

        if hours_ago > 48:
            return {
                "name": "Last run",
                "status": "fail",
                "detail": f"Last activity {hours_ago:.0f}h ago ({newest_file.name})",
            }
        elif hours_ago > 26:
            # More than a day — might have missed today's run
            return {
                "name": "Last run",
                "status": "warn",
                "detail": f"Last activity {hours_ago:.0f}h ago ({newest_file.name})",
            }
        else:
            return {
                "name": "Last run",
                "status": "pass",
                "detail": f"Last activity {hours_ago:.0f}h ago",
            }

    except Exception as e:
        return {
            "name": "Last run",
            "status": "warn",
            "detail": f"Check error: {e}",
        }


def _check_disk_space() -> dict:
    """Check available disk space in the project directory."""
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = usage.free / (1024 ** 3)

        if free_gb < 1.0:
            return {
                "name": "Disk space",
                "status": "fail",
                "detail": f"{free_gb:.1f} GB free — critically low",
            }
        elif free_gb < 5.0:
            return {
                "name": "Disk space",
                "status": "warn",
                "detail": f"{free_gb:.1f} GB free",
            }
        else:
            return {
                "name": "Disk space",
                "status": "pass",
                "detail": f"{free_gb:.1f} GB free",
            }

    except Exception as e:
        return {
            "name": "Disk space",
            "status": "warn",
            "detail": f"Check error: {e}",
        }


def run_health_check(config: dict | None = None) -> dict:
    """Run all health checks and return a summary.

    Returns:
        {
            "status": "healthy" | "degraded" | "critical",
            "checks": [{"name", "status", "detail"}, ...],
            "timestamp": ISO datetime,
        }
    """
    if config is None:
        config = _load_config()

    checks = [
        _check_posts_today(config),
        _check_token_health(),
        _check_active_queue(),
        _check_last_run(),
        _check_disk_space(),
    ]

    has_fail = any(c["status"] == "fail" for c in checks)
    has_warn = any(c["status"] == "warn" for c in checks)

    if has_fail:
        overall = "critical"
    elif has_warn:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }


def print_health_report(result: dict) -> None:
    """Display health check results in a Rich table."""
    overall = result["status"]
    status_colors = {
        "healthy": "green",
        "degraded": "yellow",
        "critical": "red",
    }
    color = status_colors.get(overall, "white")

    console.print(f"\n[bold {color}]System Status: {overall.upper()}[/bold {color}]")

    table = Table(show_lines=False)
    table.add_column("Check", style="bold", width=16)
    table.add_column("Status", width=8)
    table.add_column("Detail", width=55)

    check_icons = {
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }

    for check in result["checks"]:
        icon = check_icons.get(check["status"], check["status"])
        table.add_row(check["name"], icon, check["detail"])

    console.print(table)
