"""
Health watchdog for The Docket.

Runs lightweight checks on the pipeline and reports problems:
- Did today's queue items get posted?
- Are Meta tokens still valid?
- Can the ElevenLabs credential actually synthesize speech?
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
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# The credential check reads os.getenv directly, so .env must be loaded even
# when watchdog is imported outside the main CLI entry point. Idempotent, and
# does not override variables already set in the environment.
load_dotenv()

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


def _check_voiceover_credential(config: dict) -> dict:
    """Check that the ElevenLabs credential can actually synthesize speech.

    Probes with a real two-character TTS request rather than an account
    endpoint. A scoped key carrying ``text_to_speech`` but not ``user_read``
    returns 401 on ``/v1/user`` while working perfectly for the pipeline, so
    an account-endpoint probe would report a permanent false failure.

    Goes through the same SDK call the video pipeline uses, so an SDK-level
    break fails here too instead of passing a raw-HTTP check that the
    pipeline would not have survived.

    Costs ~2 characters of quota per run (daily), which is the price of
    catching a dead credential on day one. A broken key previously went
    undetected for three weeks because nothing verified it until a Reel
    was already being rendered.
    """
    name = "Voiceover key"
    try:
        platforms = config.get("platforms", {})
        video_enabled = any(
            p.get("enabled", False) and p.get("type") == "video"
            for p in platforms.values()
        )
        if not video_enabled:
            return {
                "name": name,
                "status": "pass",
                "detail": "No video platform enabled — skipped",
            }

        key = os.getenv("ELEVENLABS_API_KEY", "")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

        if not key:
            return {
                "name": name,
                "status": "fail",
                "detail": "ELEVENLABS_API_KEY not set — Reels will abort",
            }
        if not voice_id:
            return {
                "name": name,
                "status": "fail",
                "detail": "ELEVENLABS_VOICE_ID not set — Reels will abort",
            }
        # Cheap structural check before spending a request: the secret is
        # "sk_"-prefixed, and a UUID-shaped value is the key *ID* from the
        # dashboard, which 400s on every call.
        if not key.startswith("sk_"):
            return {
                "name": name,
                "status": "fail",
                "detail": "Key is not 'sk_'-prefixed — looks like a key ID, not a secret",
            }

        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=key, timeout=20)
        audio = b"".join(
            client.text_to_speech.convert(
                voice_id=voice_id,
                text="ok",
                model_id=os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
                output_format=os.getenv(
                    "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"
                ),
            )
        )

        if not audio:
            return {
                "name": name,
                "status": "fail",
                "detail": "TTS returned no audio — Reels would abort",
            }

        return {
            "name": name,
            "status": "pass",
            "detail": f"TTS OK ({len(audio) / 1024:.1f} KB, voice {voice_id[:8]}…)",
        }

    except Exception as e:
        try:
            from src.video.voiceover import _concise_error

            detail = _concise_error(e, limit=120)
        except Exception:
            detail = str(e)[:120]

        # Separate a permanently broken setup from a transient blip. Only the
        # former should page as critical — a warning that fires on every
        # network hiccup is noise, and noise is what gets alerts ignored.
        #
        # A 4xx means the request or credential is wrong and will stay wrong
        # until someone changes it: 400/401 bad key, 403 missing scope, 404
        # voice deleted, 422 bad model. Each aborts every Reel. The exception
        # is 429 (rate limited), which clears on its own — as do 5xx outages
        # and connection errors, which carry no status_code at all.
        code = getattr(e, "status_code", None)
        if isinstance(code, int) and 400 <= code < 500 and code != 429:
            status = "fail"
        else:
            status = "warn"

        return {"name": name, "status": status, "detail": f"TTS probe: {detail}"}


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
        _check_voiceover_credential(config),
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
