"""
Weekday content scheduler for The Docket.

Takes curated content and distributes it across Mon-Fri
into a JSON queue file. Each day gets a balanced mix of
feature articles and news cards.
"""

import errno
import fcntl
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.table import Table

from src.curator import CuratedArticle, CurationResult
from src.content_generator import (
    _build_post_text, _build_unseen_reality_hook, _generate_body_slides,
    _build_voiceover, build_teaser_post, build_wrapup_post, _tag_url,
    _build_video_caption, _select_hashtags, get_hook_variant,
    SECTION_HASHTAGS, ISSUE_URL, SUBSCRIBE_URL, PLATFORM_LIMITS,
    VIDEO_CTA_TEXT, VIDEO_CTA_SPOKEN,
)

console = Console()

WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday"]
ALL_DAY_NAMES = ["sunday"] + WEEKDAY_NAMES  # includes pre-issue Sunday
WEEKDAY_SHORT = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}

QUEUE_LOCK_NAME = ".queue.lock"

# Depth of queue locks held by THIS process, keyed by resolved lock path.
# flock is held per open-file-description, so a second acquisition in the same
# process would block against its own first one. Callers nest without knowing
# it (run_daily locks internally, and a CLI command may also be decorated), so
# track depth and only touch the kernel lock at depth 0.
_held_locks: dict[str, int] = {}
_held_locks_guard = threading.Lock()


class QueueLockBusy(RuntimeError):
    """Raised when another process already holds the queue lock."""


@contextmanager
def queue_lock(queue_dir: Path, timeout: float = 0.0):
    """Hold an exclusive advisory lock on the queue for a whole run.

    The queue file is read-modify-write: a run loads every item into memory,
    works for minutes (rendering video, uploading, and sleeping out the retry
    delay), then writes all items back. Two overlapping runs therefore both
    see an item as ``pending`` and both post it, and whichever finishes last
    overwrites the other's results — silently resurrecting posted items or
    discarding them.

    Making individual writes atomic does not help, because the window is
    between the read and the write, not inside the write. The only fix that
    closes it is serialising the runs themselves, which is what this does.

    Uses ``fcntl.flock`` on a lock file rather than a PID file: the kernel
    releases an flock when the holding process exits, so a crashed or killed
    run cannot leave a stale lock that blocks every subsequent run.

    Args:
        queue_dir: Directory holding the queue files; the lock lives here.
        timeout: Seconds to wait for the lock. The default of 0 fails
            immediately, which is what a scheduled run wants — stacking up
            blocked runs behind a slow one is worse than skipping this slot,
            since the next slot is minutes away. Interactive callers can pass
            a few seconds to ride out a run that is finishing up.

    Raises:
        QueueLockBusy: If the lock is held and `timeout` expires.
    """
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / QUEUE_LOCK_NAME
    key = os.path.realpath(lock_path)

    # Already held further up this call stack: nest instead of deadlocking
    # against ourselves.
    with _held_locks_guard:
        nested = _held_locks.get(key, 0) > 0
        if nested:
            _held_locks[key] += 1

    if nested:
        try:
            yield lock_path
        finally:
            with _held_locks_guard:
                _held_locks[key] -= 1
        return

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    acquired = False

    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as e:
                # Only EAGAIN/EWOULDBLOCK (and EACCES on some platforms) mean
                # "someone else holds it". Anything else is a real fault —
                # ENOLCK/EOPNOTSUPP from a filesystem without advisory locking,
                # EBADF from a bad descriptor. Reporting those as contention
                # would silently stop all posting while telling the operator to
                # look for a second process that does not exist.
                if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                    raise
                if time.monotonic() >= deadline:
                    # State the fact only. Each caller adds its own
                    # consequence — a scheduled run skips the slot, an
                    # interactive command tells the operator to retry.
                    raise QueueLockBusy(
                        f"Queue is locked by another run "
                        f"({_read_lock_holder(lock_path)})."
                    )
                time.sleep(0.25)

        with _held_locks_guard:
            _held_locks[key] = 1

        # Record the holder so a blocked run can say who it is waiting on.
        try:
            os.ftruncate(fd, 0)
            os.write(
                fd,
                f"pid={os.getpid()} started={datetime.now().isoformat()}\n".encode(),
            )
            os.fsync(fd)
        except OSError:
            pass  # Diagnostics only — never fail a run over this.

        yield lock_path

    finally:
        # Nested holders have already decremented back to 1 by now; if
        # acquisition failed the key was never registered and this is a no-op.
        with _held_locks_guard:
            _held_locks.pop(key, None)
        try:
            # Only unlock what we actually took. Unlocking on the failed-
            # acquisition path is not merely redundant: if that call raises it
            # replaces the in-flight QueueLockBusy, so the caller sees a bare
            # OSError instead of the contention it needs to handle.
            # Errors here are swallowed because closing the fd releases the
            # flock regardless, and a failed unlock must never mask the real
            # exception leaving this block.
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)


def _read_lock_holder(lock_path: Path) -> str:
    """Best-effort description of the process holding the lock."""
    try:
        text = lock_path.read_text().strip()
        return text or "unknown holder"
    except OSError:
        return "unknown holder"


@dataclass
class QueueItem:
    """A single queued post."""
    id: str
    day: str                    # "monday", "tuesday", etc.
    date: str                   # ISO date: "2026-02-09"
    content_type: str           # "text" or "video"
    platform: str               # "bluesky", "twitter", etc.
    article_title: str
    section: str
    is_feature: bool
    text: str                   # post text (for text posts) or caption (for video)
    url: str = ""               # article URL (for Twitter reply-with-link flow)
    hashtags: list[str] = field(default_factory=list)
    scheduled_time: str | None = None  # ISO datetime w/ tz: "2026-02-12T09:30:00-08:00"
    video_script: dict | None = None
    status: str = "pending"     # "pending", "posted", "failed", "blocked"
    attempts: int = 0
    last_error: str | None = None
    posted_at: str | None = None
    post_uri: str | None = None
    compliance_status: str | None = None     # "pass" | "warn" | "fixed" | "blocked"
    compliance_detail: str | None = None     # human-readable summary
    compliance_original: str | None = None   # pre-fix text (only when auto-fixed)
    hook_variant: int | None = None          # which hook variant was used (0-4) for A/B testing


@dataclass
class WeekQueue:
    """A full week's posting queue."""
    issue_number: int | None
    created_at: str
    week_start: str
    week_end: str
    items: list[QueueItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "issue_number": self.issue_number,
            "created_at": self.created_at,
            "week_start": self.week_start,
            "week_end": self.week_end,
            "items": [asdict(i) for i in self.items],
        }

    def save(self, path: Path):
        """Write the queue to disk atomically.

        Serialises to a temporary file in the same directory and then
        ``os.replace``s it into place, which is atomic on POSIX. A crash or
        kill partway through a write therefore leaves the previous queue
        intact rather than a truncated file — the queue is the only record of
        what has already been posted, so a corrupt one risks double-posting
        every item in the week.

        Note this makes each individual write atomic; it does not make
        read-modify-write cycles safe against concurrent runs. Callers that
        mutate the queue must hold :func:`queue_lock`.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        console.print(f"[green]Queue saved to {path}[/green]")

    @classmethod
    def load(cls, path: Path) -> "WeekQueue":
        with open(path) as f:
            data = json.load(f)
        queue = cls(
            issue_number=data["issue_number"],
            created_at=data["created_at"],
            week_start=data["week_start"],
            week_end=data["week_end"],
        )
        for item_data in data.get("items", []):
            queue.items.append(QueueItem(**item_data))
        return queue

    def get_today_items(self, up_to: datetime | None = None) -> list[QueueItem]:
        """Filter to pending items scheduled for today, due at or before `up_to`.

        "Today" is derived from `up_to` when it's provided (tz-aware), otherwise
        from the system clock. Using `up_to.date()` prevents the bug where the
        system timezone disagrees with the queue's scheduling timezone — without
        this, an item scheduled for Friday in PT could be filtered out by a
        runner whose system clock has already rolled to Saturday UTC.
        Items without a scheduled_time are always considered due.
        """
        if up_to is not None:
            today = up_to.strftime("%Y-%m-%d")
        else:
            today = datetime.now().strftime("%Y-%m-%d")
        today_pending = [
            i for i in self.items if i.date == today and i.status == "pending"
        ]
        if up_to is None:
            return today_pending

        due = []
        for item in today_pending:
            if item.scheduled_time is None:
                due.append(item)
            else:
                item_time = datetime.fromisoformat(item.scheduled_time)
                if item_time <= up_to:
                    due.append(item)
        return due

    def get_retryable(self, max_retries: int = 3, up_to: datetime | None = None) -> list[QueueItem]:
        """Get failed items that haven't exceeded retry limit and are due."""
        if up_to is not None:
            today = up_to.strftime("%Y-%m-%d")
        else:
            today = datetime.now().strftime("%Y-%m-%d")
        candidates = [
            i for i in self.items
            if i.date == today and i.status == "failed" and i.attempts < max_retries
        ]
        if up_to is None:
            return candidates

        due = []
        for item in candidates:
            if item.scheduled_time is None:
                due.append(item)
            else:
                item_time = datetime.fromisoformat(item.scheduled_time)
                if item_time <= up_to:
                    due.append(item)
        return due

    def get_orphaned_items(self, up_to: datetime) -> list[QueueItem]:
        """Return pending items whose scheduled date is before today.

        These items will never be picked up by `get_today_items` again because
        of the `i.date == today` filter. Used by the daily runner to surface
        silent attrition (e.g. items missed because launchd didn't fire that day,
        or because the item was scheduled past the final launchd slot).
        """
        today = up_to.strftime("%Y-%m-%d")
        return [
            i for i in self.items
            if i.status == "pending" and i.date and i.date < today
        ]

    def mark_posted(self, item_id: str, post_uri: str = ""):
        """Mark a queue item as successfully posted."""
        for item in self.items:
            if item.id == item_id:
                item.status = "posted"
                item.posted_at = datetime.now().isoformat()
                item.post_uri = post_uri
                break

    def mark_failed(self, item_id: str, error: str):
        """Mark a queue item as failed."""
        for item in self.items:
            if item.id == item_id:
                item.status = "failed"
                item.attempts += 1
                item.last_error = error
                break

    @property
    def stats(self) -> dict:
        """Summary stats for the queue."""
        by_day = {}
        for item in self.items:
            day = item.day
            if day not in by_day:
                by_day[day] = {"text": 0, "video": 0, "posted": 0, "failed": 0, "pending": 0}
            by_day[day][item.content_type] += 1
            by_day[day][item.status] += 1
        return by_day


def _next_weekday(start: datetime, target_weekday: int) -> datetime:
    """Find the next occurrence of a weekday (0=Monday)."""
    days_ahead = target_weekday - start.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return start + timedelta(days=days_ahead)


def _assign_time_slots(
    day_items: list[QueueItem],
    day_date: str,
    time_slots: list[str],
    tz: ZoneInfo,
    platform_time_slots: dict[str, list[str]] | None = None,
):
    """
    Distribute a day's items across platform-specific time slots.

    Each platform gets its own posting schedule from platform_time_slots.
    Falls back to the global time_slots list if a platform has no specific slots.

    Meta posts (Sunday teaser, Friday wrap-up) get fixed evening/afternoon
    slots so they don't collide with regular content timing.

    Sets scheduled_time as a timezone-aware ISO datetime string on each item.
    """
    platform_time_slots = platform_time_slots or {}

    # Separate meta posts (teaser/wrap-up) from regular content
    meta_items = [item for item in day_items if item.section == "meta"]
    regular_items = [item for item in day_items if item.section != "meta"]

    # Assign fixed times to meta posts
    for item in meta_items:
        if item.day == "sunday":
            # Sunday teaser: evening slot
            hour, minute = 19, 0
        elif "[Wrap-up]" in item.article_title:
            # Friday wrap-up: late afternoon after regular posts
            hour, minute = 16, 30
        else:
            hour, minute = 18, 0  # generic meta fallback

        scheduled_dt = datetime.fromisoformat(day_date).replace(
            hour=hour, minute=minute, second=0, microsecond=0,
            tzinfo=tz,
        )
        item.scheduled_time = scheduled_dt.isoformat()

    # Group regular items by platform so each platform uses its own slots
    items_by_platform: dict[str, list[QueueItem]] = {}
    for item in regular_items:
        items_by_platform.setdefault(item.platform, []).append(item)

    for platform, platform_items in items_by_platform.items():
        slots = platform_time_slots.get(platform, time_slots)
        if not slots:
            slots = time_slots

        for i, item in enumerate(platform_items):
            slot_index = i % len(slots)
            slot_time = slots[slot_index]  # e.g. "09:30"
            hour, minute = map(int, slot_time.split(":"))
            scheduled_dt = datetime.fromisoformat(day_date).replace(
                hour=hour, minute=minute, second=0, microsecond=0,
                tzinfo=tz,
            )
            item.scheduled_time = scheduled_dt.isoformat()


def generate_week_schedule(
    curation: CurationResult,
    config: dict,
    start_date: datetime | None = None,
) -> WeekQueue:
    """
    Distribute curated content across the weekdays of the target week.

    Creates QueueItems for each article × platform combination,
    distributed round-robin across Mon-Fri.
    """
    schedule_config = config.get("schedule", {})
    platforms_config = config.get("platforms", {})

    # Find text and video platforms
    text_platforms = {}
    video_platforms = {}
    for pname, pconf in platforms_config.items():
        if not pconf.get("enabled", False):
            continue
        if pconf.get("type") == "text":
            text_platforms[pname] = pconf.get("max_chars", PLATFORM_LIMITS.get(pname, 280))
        elif pconf.get("type") == "video":
            video_platforms[pname] = pconf

    if not text_platforms and not video_platforms:
        console.print("[yellow]No platforms enabled. Enable at least one in config.yaml.[/yellow]")

    # Calculate week dates
    now = start_date or datetime.now()
    # Find next Monday (or this Monday if today is Mon-Fri before post time)
    if now.weekday() < 5:  # Mon-Fri
        monday = now - timedelta(days=now.weekday())
    else:
        monday = _next_weekday(now, 0)

    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    week_dates = [(monday + timedelta(days=i)) for i in range(5)]

    queue = WeekQueue(
        issue_number=curation.issue_number,
        created_at=datetime.now().isoformat(),
        week_start=week_dates[0].strftime("%Y-%m-%d"),
        week_end=week_dates[-1].strftime("%Y-%m-%d"),
    )

    item_counter = 0
    cta_counter = 0  # global counter for cold-start CTA rotation across all text posts

    # Cold-start CTA rotation helpers
    _cold_start_cfg = config.get("cold_start", {})
    _is_cold_start = _cold_start_cfg.get("enabled", False)
    _follow_prompts = [
        "Follow @docketclimate for more.",
        "More tomorrow. Follow @docketclimate.",
        "Follow for the stories that fall between the headlines.",
    ]

    # --- Distribute features across days (round-robin) ---
    features = curation.selected_features
    for i, article in enumerate(features):
        day_index = i % 5
        day_name = WEEKDAY_NAMES[day_index]
        day_date = week_dates[day_index].strftime("%Y-%m-%d")

        # Text post for each platform
        raw_link = article.url or ISSUE_URL
        for platform, max_chars in text_platforms.items():
            item_counter += 1
            cta_counter += 1
            hashtags = _select_hashtags(
                article.title, article.section_id, platform,
                getattr(article, "hook", "") or "",
            )
            # Tag link with UTM for subscriber attribution
            content_tag = f"{article.section_id}_feature"
            campaign = f"issue_{curation.issue_number}" if curation.issue_number else ""
            link = _tag_url(raw_link, platform=platform, medium="text",
                            campaign=campaign, content=content_tag)

            # ── Cold-start CTA rotation ──────────────────────────
            # Always keep the article link (platforms need it for link
            # cards / facets / reply-with-link).  CTA rotation only
            # controls hashtags and follow prompts.
            hook_text = article.hook
            rotation_slot = cta_counter % 10
            if _is_cold_start:
                if rotation_slot in (0, 1, 2, 3, 4):
                    # 50%: clean look — link but no hashtags
                    hashtags = []
                elif rotation_slot in (5, 6, 7):
                    # 30%: follow-driving — link + follow prompt, no hashtags
                    hashtags = []
                    hook_text = hook_text + "\n\n" + _follow_prompts[cta_counter % len(_follow_prompts)]
                # else (8, 9): 20% — keep link + hashtags as-is

            # Build post text
            tag_str = " ".join(hashtags)
            if link:
                suffix = f"\n\n{link}\n{tag_str}" if tag_str else f"\n\n{link}"
            elif tag_str:
                suffix = f"\n\n{tag_str}"
            else:
                suffix = ""
            available = max_chars - len(suffix)
            if len(hook_text) > available:
                hook_text = hook_text[:available - 1] + "\u2026"
            text = f"{hook_text}{suffix}"

            queue.items.append(QueueItem(
                id=f"{curation.issue_number}_{day_name[:3]}_{item_counter:03d}",
                day=day_name,
                date=day_date,
                content_type="text",
                platform=platform,
                article_title=article.title,
                section=article.section_id,
                is_feature=True,
                text=text,
                url=link,
                hashtags=hashtags,
                hook_variant=get_hook_variant(article.title),
            ))

    # --- Distribute video features across days ---
    video_articles = curation.video_features
    for i, article in enumerate(video_articles):
        day_index = i % 5
        day_name = WEEKDAY_NAMES[day_index]
        day_date = week_dates[day_index].strftime("%Y-%m-%d")

        # Build video script data
        raw_link = article.url or ISSUE_URL
        link = _tag_url(raw_link, platform="reels", medium="video",
                        campaign=f"issue_{curation.issue_number}" if curation.issue_number else "",
                        content=f"{article.section_id}_video")
        caption = _build_video_caption(article.hook, article.section_id, article.title)

        # Enforce cold-start slide cap
        _body_slides = article.body_slides or []
        _vo_text = article.voiceover_text or []
        if _is_cold_start:
            _max_slides = _cold_start_cfg.get("video", {}).get("body_slides", 3)
            if len(_body_slides) > _max_slides:
                _body_slides = _body_slides[:_max_slides]
                _vo_text = _vo_text[:_max_slides] if isinstance(_vo_text, list) else _vo_text

        video_data = {
            "title": article.title,
            "hook": article.hook,
            "body_slides": _body_slides,
            "voiceover_text": _vo_text,
            "cta": VIDEO_CTA_TEXT,
            "voiceover_cta": VIDEO_CTA_SPOKEN[i % len(VIDEO_CTA_SPOKEN)],
            "url": link,
            "video_tier": getattr(article, "video_tier", "narrative"),
            "cinematic_score": getattr(article, "cinematic_score", 0),
            "image_prompts": getattr(article, "image_prompts", []),
            "background_prompt": getattr(article, "background_prompt", ""),
        }

        for platform in video_platforms:
            item_counter += 1
            queue.items.append(QueueItem(
                id=f"{curation.issue_number}_{day_name[:3]}_{item_counter:03d}",
                day=day_name,
                date=day_date,
                content_type="video",
                platform=platform,
                article_title=article.title,
                section=article.section_id,
                is_feature=True,
                text=caption,
                url=link,
                video_script=video_data,
                hook_variant=get_hook_variant(article.title),
            ))

    # --- Distribute news across days (round-robin) ---
    news = curation.selected_news
    for i, article in enumerate(news):
        day_index = i % 5
        day_name = WEEKDAY_NAMES[day_index]
        day_date = week_dates[day_index].strftime("%Y-%m-%d")

        raw_link = article.url or ISSUE_URL
        for platform, max_chars in text_platforms.items():
            item_counter += 1
            cta_counter += 1
            hashtags = _select_hashtags(
                article.title, article.section_id, platform,
                getattr(article, "hook", "") or "",
            )
            # Tag link with UTM for subscriber attribution
            content_tag = f"{article.section_id}_news"
            campaign = f"issue_{curation.issue_number}" if curation.issue_number else ""
            link = _tag_url(raw_link, platform=platform, medium="text",
                            campaign=campaign, content=content_tag)

            # ── Cold-start CTA rotation ──────────────────────────
            # Always keep the article link (platforms need it for link
            # cards / facets / reply-with-link).  CTA rotation only
            # controls hashtags and follow prompts.
            hook_text = article.hook
            rotation_slot = cta_counter % 10
            if _is_cold_start:
                if rotation_slot in (0, 1, 2, 3, 4):
                    # 50%: clean look — link but no hashtags
                    hashtags = []
                elif rotation_slot in (5, 6, 7):
                    # 30%: follow-driving — link + follow prompt, no hashtags
                    hashtags = []
                    hook_text = hook_text + "\n\n" + _follow_prompts[cta_counter % len(_follow_prompts)]
                # else (8, 9): 20% — keep link + hashtags as-is

            # Build post text
            tag_str = " ".join(hashtags)
            if link:
                suffix = f"\n\n{link}\n{tag_str}" if tag_str else f"\n\n{link}"
            elif tag_str:
                suffix = f"\n\n{tag_str}"
            else:
                suffix = ""
            available = max_chars - len(suffix)
            if len(hook_text) > available:
                hook_text = hook_text[:available - 1] + "\u2026"
            text = f"{hook_text}{suffix}"

            queue.items.append(QueueItem(
                id=f"{curation.issue_number}_{day_name[:3]}_{item_counter:03d}",
                day=day_name,
                date=day_date,
                content_type="text",
                platform=platform,
                article_title=article.title,
                section=article.section_id,
                is_feature=False,
                text=text,
                url=link,
                hashtags=hashtags,
                hook_variant=get_hook_variant(article.title),
            ))

    # --- Sunday teaser + Friday wrap-up (Fix M) ---
    cold_start = config.get("cold_start", {})
    skip_teaser_wrapup = cold_start.get("enabled", False) and cold_start.get("skip_teaser_wrapup", True)

    sunday_date = monday - timedelta(days=1)  # Sunday before the week
    sunday_date_str = sunday_date.strftime("%Y-%m-%d")
    friday_date_str = week_dates[4].strftime("%Y-%m-%d")  # Friday

    if skip_teaser_wrapup:
        console.print(
            "[dim]Cold-start mode: skipping Sunday teaser + Friday wrap-up "
            "(defer until 50+ followers)[/dim]"
        )
    else:
        # Gather top article titles for the wrap-up
        all_articles_titles = [a.title for a in (features + (curation.selected_news or []))]
        teaser_title = features[0].title if features else (curation.selected_news[0].title if curation.selected_news else "climate stories you need to see")
        total_article_count = len(curation.selected_features) + len(curation.selected_news)

        for platform, max_chars in text_platforms.items():
            # Sunday teaser
            item_counter += 1
            teaser_text = build_teaser_post(
                issue_number=curation.issue_number,
                teaser_article_title=teaser_title,
                article_count=total_article_count,
                platform=platform,
                max_chars=max_chars,
            )
            queue.items.append(QueueItem(
                id=f"{curation.issue_number}_sun_{item_counter:03d}",
                day="sunday",
                date=sunday_date_str,
                content_type="text",
                platform=platform,
                article_title="[Teaser] Upcoming Issue",
                section="meta",
                is_feature=False,
                text=teaser_text,
                url=SUBSCRIBE_URL,
                hashtags=[],
            ))

            # Friday wrap-up
            item_counter += 1
            wrapup_text = build_wrapup_post(
                issue_number=curation.issue_number,
                top_titles=all_articles_titles[:3],
                article_count=total_article_count,
                platform=platform,
                max_chars=max_chars,
            )
            queue.items.append(QueueItem(
                id=f"{curation.issue_number}_fri_wrap_{item_counter:03d}",
                day="friday",
                date=friday_date_str,
                content_type="text",
                platform=platform,
                article_title="[Wrap-up] Week in Review",
                section="meta",
                is_feature=False,
                text=wrapup_text,
                url=SUBSCRIBE_URL,
                hashtags=[],
            ))

        console.print(
            f"[dim]Added Sunday teaser + Friday wrap-up for "
            f"{len(text_platforms)} platforms[/dim]"
        )

    # Update week range to include Sunday if teaser was added
    if not skip_teaser_wrapup:
        queue.week_start = sunday_date_str

    # --- Compliance checks on generated queue items ---
    try:
        comp_cfg = config.get("compliance", {})
        if comp_cfg.get("enabled", False):
            from src.compliance import run_compliance_check

            checked = 0
            fixed = 0
            warned = 0
            blocked = 0

            for item in queue.items:
                if item.content_type != "text":
                    continue  # Skip video items for now

                result = run_compliance_check(
                    post_text=item.text,
                    source_text=None,  # not available at schedule time
                    article_url=item.url or None,
                    platform=item.platform,
                    config=config,
                )
                item.compliance_status = result.status
                item.compliance_detail = result.summary

                if result.status == "fixed" and result.fixed_text:
                    item.compliance_original = item.text
                    item.text = result.fixed_text
                    fixed += 1
                elif result.status == "blocked":
                    item.status = "blocked"
                    blocked += 1
                elif result.status == "warn":
                    warned += 1

                checked += 1

            # Log compliance results
            parts = []
            if fixed:
                parts.append(f"[cyan]{fixed} auto-fixed[/cyan]")
            if warned:
                parts.append(f"[yellow]{warned} warnings[/yellow]")
            if blocked:
                parts.append(f"[red]{blocked} blocked[/red]")
            passed = checked - fixed - warned - blocked
            if passed:
                parts.append(f"[green]{passed} passed[/green]")
            if parts:
                console.print(
                    f"[dim]Compliance: {', '.join(parts)} "
                    f"({checked} text items checked)[/dim]"
                )
    except Exception as e:
        console.print(f"[dim]Compliance check skipped: {e}[/dim]")

    # --- Assign time slots to each day's items ---
    time_slots = schedule_config.get("time_slots", ["09:00"])
    platform_time_slots = schedule_config.get("platform_time_slots", {})
    tz_name = schedule_config.get("timezone", "America/Los_Angeles")
    tz = ZoneInfo(tz_name)

    items_by_date: dict[str, list[QueueItem]] = {}
    for item in queue.items:
        items_by_date.setdefault(item.date, []).append(item)

    for day_date, day_items in items_by_date.items():
        _assign_time_slots(day_items, day_date, time_slots, tz, platform_time_slots)

    # Log platform-specific timing
    if platform_time_slots:
        platforms_with_slots = ", ".join(
            f"{p} ({len(s)} slots)" for p, s in platform_time_slots.items()
        )
        console.print(f"[dim]Platform-specific timing: {platforms_with_slots}[/dim]")

    console.print(
        f"[bold green]Scheduled {len(queue.items)} queue items "
        f"across {queue.week_start} to {queue.week_end}[/bold green]"
    )

    return queue


def print_queue_status(queue: WeekQueue):
    """Display a Rich table showing the queue status by day."""
    table = Table(title=f"Week Queue — Issue #{queue.issue_number}", show_lines=True)
    table.add_column("Day", style="bold", width=12)
    table.add_column("Date", width=12)
    table.add_column("Features", justify="center", width=10)
    table.add_column("News", justify="center", width=8)
    table.add_column("Videos", justify="center", width=8)
    table.add_column("Posted", justify="center", style="green", width=8)
    table.add_column("Pending", justify="center", style="yellow", width=8)
    table.add_column("Failed", justify="center", style="red", width=8)

    today = datetime.now().strftime("%Y-%m-%d")

    for day_name in ALL_DAY_NAMES:
        day_items = [i for i in queue.items if i.day == day_name]
        if not day_items:
            continue

        date = day_items[0].date
        features = len([i for i in day_items if i.is_feature and i.content_type == "text"])
        news = len([i for i in day_items if not i.is_feature])
        videos = len([i for i in day_items if i.content_type == "video"])
        posted = len([i for i in day_items if i.status == "posted"])
        pending = len([i for i in day_items if i.status == "pending"])
        failed = len([i for i in day_items if i.status == "failed"])

        day_label = day_name.capitalize()
        if date == today:
            day_label = f"[bold cyan]> {day_label}[/bold cyan]"

        table.add_row(
            day_label, date,
            str(features), str(news), str(videos),
            str(posted), str(pending), str(failed),
        )

    # Totals
    total = len(queue.items)
    total_posted = len([i for i in queue.items if i.status == "posted"])
    total_pending = len([i for i in queue.items if i.status == "pending"])
    total_failed = len([i for i in queue.items if i.status == "failed"])

    table.add_row(
        "[bold]Total[/bold]", "",
        "", "", "",
        f"[bold]{total_posted}[/bold]",
        f"[bold]{total_pending}[/bold]",
        f"[bold]{total_failed}[/bold]",
    )

    console.print(table)

    # Show today's detail if there are items
    today_items = [i for i in queue.items if i.date == today]
    if today_items:
        # Sort by scheduled_time for display
        today_items.sort(key=lambda i: i.scheduled_time or "")
        console.print(f"\n[bold]Today's posts ({len(today_items)}):[/bold]")
        for item in today_items:
            status_color = {"posted": "green", "pending": "yellow", "failed": "red"}.get(item.status, "white")
            icon = {"posted": "\u2713", "pending": "\u25cb", "failed": "\u2717"}.get(item.status, "?")
            time_str = ""
            if item.scheduled_time:
                st = datetime.fromisoformat(item.scheduled_time)
                time_str = f"[dim]{st.strftime('%H:%M')}[/dim] "
            console.print(
                f"  [{status_color}]{icon}[/{status_color}] "
                f"{time_str}"
                f"[{status_color}]{item.platform:10s}[/{status_color}] "
                f"{'[bold]F[/bold]' if item.is_feature else 'N'} "
                f"{item.article_title[:50]}"
            )

    # Show flag hint if there are posted items
    if total_posted > 0:
        console.print(
            f"\n[dim]Flag good posts: "
            f"python -m src.main flag-good <ITEM_ID>[/dim]"
        )


def _queue_sort_key(path: Path) -> tuple:
    """Order queue files by the week they cover, not by filesystem mtime.

    Reads ``week_start`` from the file, falling back to the ISO date in the
    filename (``week_48_2026-09-14.json``), and only then to mtime.
    """
    try:
        with open(path) as f:
            week_start = json.load(f).get("week_start") or ""
        if week_start:
            return (2, week_start, path.stat().st_mtime)
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if m:
        return (1, m.group(1), path.stat().st_mtime)

    return (0, "", path.stat().st_mtime)


def find_active_queue(queue_dir: Path) -> Path | None:
    """Find the queue file covering the most recent week.

    Selection is by the queue's own ``week_start``, NOT by file mtime.
    mtime is not a property of the content: any git operation that writes a
    tracked queue file — checkout, rebase, clone, stash pop — resets it. On
    2026-09-14 a rebase touched a dozen archived queue files and this
    function began returning a four-month-old May queue, which would have
    silently posted nothing on the next scheduled run because no item in it
    was dated today.
    """
    if not queue_dir.exists():
        return None
    files = sorted(queue_dir.glob("week_*.json"), key=_queue_sort_key, reverse=True)
    return files[0] if files else None
