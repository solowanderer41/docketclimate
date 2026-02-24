"""
Daily queue runner for The Docket.

Called by launchd/cron each weekday. Loads the active queue,
finds today's pending items, posts them with retry logic,
and updates the queue file.
"""

import json
import tempfile
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from PIL import Image
from rich.console import Console

from src.scheduler import WeekQueue, QueueItem, find_active_queue

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_DIR = PROJECT_ROOT / "queue"
LOG_DIR = PROJECT_ROOT / "logs"


def run_daily(config: dict, dry_run: bool = False):
    """
    Main daily execution:
    1. Find the active queue file
    2. Get today's items that are due (scheduled_time <= now)
    3. Post each item to its platform
    4. Update queue file with results
    5. Retry failed items
    """
    queue_path = find_active_queue(QUEUE_DIR)
    if not queue_path:
        console.print("[yellow]No active queue found. Run 'schedule' first.[/yellow]")
        return

    queue = WeekQueue.load(queue_path)
    console.print(
        f"[cyan]Loaded queue for issue #{queue.issue_number} "
        f"({queue.week_start} to {queue.week_end})[/cyan]"
    )

    # Determine "now" in the configured timezone
    schedule_config = config.get("schedule", {})
    tz_name = schedule_config.get("timezone", "America/Los_Angeles")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    console.print(f"[dim]Current time: {now.strftime('%Y-%m-%d %H:%M %Z')}[/dim]")

    # Get items due up to now
    today_items = queue.get_today_items(up_to=now)
    if not today_items:
        console.print("[dim]No pending items due at this time.[/dim]")

        # Check for retryable failures
        retryable = queue.get_retryable(
            schedule_config.get("retry_max", 3),
            up_to=now,
        )
        if retryable:
            console.print(f"[yellow]{len(retryable)} failed items to retry.[/yellow]")
            _process_items(retryable, queue, queue_path, config, dry_run)
        return

    console.print(
        f"[bold]Items due now: {len(today_items)}[/bold]"
    )

    # Process items
    _process_items(today_items, queue, queue_path, config, dry_run)

    # Retry failures (only items that were due)
    retry_max = schedule_config.get("retry_max", 3)
    retry_delay = schedule_config.get("retry_delay_minutes", 15)
    retryable = queue.get_retryable(retry_max, up_to=now)

    if retryable and not dry_run:
        console.print(
            f"\n[yellow]{len(retryable)} items failed. "
            f"Retrying in {retry_delay} minutes...[/yellow]"
        )
        time.sleep(retry_delay * 60)
        _process_items(retryable, queue, queue_path, config, dry_run)

    # Final summary
    today_str = now.strftime("%Y-%m-%d")
    all_today = [i for i in queue.items if i.date == today_str]
    posted = len([i for i in all_today if i.status == "posted"])
    failed = len([i for i in all_today if i.status == "failed"])
    pending = len([i for i in all_today if i.status == "pending"])

    console.print(f"\n[bold]Run Summary:[/bold]")
    console.print(f"  [green]Posted: {posted}[/green]")
    if failed:
        console.print(f"  [red]Failed: {failed}[/red]")
    if pending:
        console.print(f"  [yellow]Still pending (later slots): {pending}[/yellow]")

    # Log results
    _log_daily_run(posted, failed, pending, queue.issue_number)

    # Send failure notification (only fires when failed > 0)
    if failed > 0:
        try:
            from src.notifier import notify_daily_summary
            failed_items = [i for i in all_today if i.status == "failed"]
            notify_daily_summary(
                posted, failed, pending,
                issue_number=queue.issue_number,
                failed_items=failed_items,
                config=config,
            )
        except Exception:
            pass  # Notifications must never break the pipeline


def _process_items(
    items: list[QueueItem],
    queue: WeekQueue,
    queue_path: Path,
    config: dict,
    dry_run: bool,
):
    """Process a batch of queue items, posting each to its platform."""
    platforms_config = config.get("platforms", {})

    for item in items:
        platform_config = platforms_config.get(item.platform, {})

        if not platform_config.get("enabled", False):
            console.print(
                f"  [dim]Skip {item.platform} (disabled): {item.article_title[:40]}[/dim]"
            )
            continue

        if dry_run:
            console.print(
                f"  [cyan]DRY RUN[/cyan] [{item.platform}] "
                f"{'F' if item.is_feature else 'N'} "
                f"{item.article_title[:40]}  "
                f"[dim]({len(item.text)} chars)[/dim]"
            )
            continue

        # Post to platform
        success, result = _post_item(item, config)

        if success:
            queue.mark_posted(item.id, result)
            console.print(
                f"  [green]\u2713 {item.platform}[/green] "
                f"{item.article_title[:40]}"
            )
        else:
            queue.mark_failed(item.id, result)
            console.print(
                f"  [red]\u2717 {item.platform}[/red] "
                f"{item.article_title[:40]}: {result}"
            )

        # Save after each item (crash-safe)
        queue.save(queue_path)


def _extract_uri(result, platform: str) -> str:
    """Extract a clean URI string from a publisher's return value.

    Publishers return various types (dict, object, string).
    This normalises them all to a single URI string suitable for
    metrics collection.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        # Try common keys in priority order
        for key in ("uri", "thread_id", "tweet_id", "id", "media_id"):
            val = result.get(key)
            if val:
                if key == "tweet_id":
                    return f"twitter:{val}"
                return str(val)
        # Last resort: log warning and return first value found
        console.print(f"[yellow]Unknown result keys from {platform}: {list(result.keys())}[/yellow]")
        for val in result.values():
            if val:
                return str(val)
        return ""

    # Object with attributes
    for attr in ("uri", "id", "thread_id"):
        val = getattr(result, attr, None)
        if val:
            return str(val)

    return str(result) if result else ""


def _post_item(item: QueueItem, config: dict) -> tuple[bool, str]:
    """
    Post a single queue item to its target platform.

    Returns (success: bool, uri_or_error: str).
    """
    try:
        publisher = _get_publisher(item.platform)

        if item.content_type == "text":
            # Twitter: post hook without link, then reply with link
            # (external links penalize reach 30-50% on X)
            if item.platform == "twitter":
                import re as _re
                # Use item.url if available, otherwise extract from text
                article_url = getattr(item, "url", "") or ""
                if not article_url:
                    url_match = _re.search(r'https?://\S+', item.text)
                    if url_match:
                        article_url = url_match.group(0)

                if article_url:
                    # Strip URL (and trailing whitespace) from tweet text
                    hook_text = _re.sub(r'\s*https?://\S+', '', item.text).rstrip()
                    # Also strip trailing hashtag line if it's now at the end
                    hook_text = hook_text.rstrip()
                    # Truncate to 280 if needed
                    if len(hook_text) > 280:
                        hook_text = hook_text[:279] + "\u2026"
                    result = publisher.publish_with_reply(hook_text, article_url)
                else:
                    result = publisher.publish_text(item.text)
                uri = _extract_uri(result, "twitter")
                return True, uri

            # All other text platforms: post as-is
            from src.content_generator import TextPost
            post = TextPost(
                platform=item.platform,
                text=item.text,
                hashtags=item.hashtags,
                article_title=item.article_title,
                section=item.section,
            )
            result = publisher.publish(post)
            uri = _extract_uri(result, item.platform)
            return True, uri

        elif item.content_type == "video":
            return _post_video_item(item, config)

    except Exception as e:
        return False, str(e)


def _post_video_item(item: QueueItem, config: dict) -> tuple[bool, str]:
    """
    Generate, upload, and publish a video queue item.

    Full pipeline:
    1. Reconstruct VideoScript from queue item data
    2. Generate per-slide voiceover audio (fallback: silent video)
    3. Fetch stock B-roll background
    4. Generate video MP4 with synced per-slide audio
    5. Upload to cloud storage (get public URL)
    6. Publish to platform (Reels)
    7. Clean up temp files (automatic via TemporaryDirectory)

    Returns (success: bool, uri_or_error: str).
    """
    from src.video.generator import VideoScript, generate_video
    from src.video.voiceover import generate_voiceover, generate_voiceover_per_slide
    from src.video.uploader import upload_video

    video_config = config.get("video", {})
    voiceover_config = config.get("voiceover", {})
    # Inject voiceover timing settings into video_config for the generator
    video_config["voiceover_padding"] = voiceover_config.get("voiceover_padding", 0.3)
    video_config["min_slide_duration"] = voiceover_config.get("min_slide_duration", 1.5)
    vs = item.video_script or {}

    # --- Build body_slides (with backward-compat fallback) ---
    body_slides = vs.get("body_slides", [])
    body_slides_from_hook = False
    if not body_slides:
        # Synthesize from hook text (split on double-newline)
        hook_text = vs.get("hook", item.article_title)
        body_slides = [p.strip() for p in hook_text.split("\n\n") if p.strip()]
        body_slides_from_hook = True
        if not body_slides:
            body_slides = [item.article_title]

    title = vs.get("title", item.article_title)
    hook = vs.get("hook", "")
    cta = vs.get("cta", "")
    cta_spoken = vs.get("voiceover_cta", "The full story is at The Docket.")

    # Cold-start mode: skip separate CTA card for shorter Reels (Fix X4)
    _cold_start_cfg = config.get("cold_start", {})
    _cold_video_active = (
        _cold_start_cfg.get("enabled", False)
        and _cold_start_cfg.get("video", {}).get("integrate_cta", False)
    )
    if _cold_video_active:
        cta = ""  # no separate CTA slide → shorter video
        # CTA voiceover will also be skipped below

    # --- Build per-slide voiceover texts ---
    # Slide structure: Title, Hook, Body 1, Body 2, ..., CTA
    # Title slide: VO reads just the title. After the title audio finishes,
    # there's a beat (pause) before the hook starts — the queue_runner adds
    # voiceover_padding to the title slide duration and the generator holds
    # on the title image for a moment before cutting.
    # Hook slide: VO reads the hook text as a separate segment.
    slide_voiceover_texts = []
    slide_labels = []

    if body_slides_from_hook:
        # body_slides already contain the hook content — title only, no hook slide
        slide_voiceover_texts.append(title)
        slide_labels.append("title")
        for i, body in enumerate(body_slides, 1):
            slide_voiceover_texts.append(body)
            slide_labels.append(f"body_{i}")
    else:
        # Title slide VO (just the title — short, punchy)
        slide_voiceover_texts.append(title)
        slide_labels.append("title")

        # Hook slide VO (separate segment with its own image)
        hook_spoken = hook.replace("\n\n", " ").strip()
        if hook_spoken:
            slide_voiceover_texts.append(hook_spoken)
            slide_labels.append("hook")

        for i, body in enumerate(body_slides, 1):
            slide_voiceover_texts.append(body)
            slide_labels.append(f"body_{i}")

    if not _cold_video_active:
        slide_voiceover_texts.append(cta_spoken)
        slide_labels.append("cta")

    # --- Build VideoScript for the renderer ---
    # When body_slides came from hook, skip the separate Hook slide
    # to match the voiceover structure
    # Extract AI image generation fields from video script data
    video_tier = vs.get("video_tier", "narrative")
    image_prompts = vs.get("image_prompts", [])
    background_prompt = vs.get("background_prompt", "")

    # --- Regeneration safety net (Fix T) ---
    # Detect items queued before Fix O added cinematic scoring.
    # These have empty/default video_tier with no image prompts — they'd
    # fall through to irrelevant stock footage or dark gradient.
    # Regenerate their cinematic data via Claude API before rendering.
    _needs_regen = (
        video_tier in (None, "", "narrative")
        and not image_prompts
        and not background_prompt
        and not body_slides_from_hook  # only real pipeline items, not synthesized
    )

    if _needs_regen:
        console.print(
            f"    [yellow]Missing cinematic data — regenerating via Claude API...[/yellow]"
        )
        try:
            from src.content_generator import _generate_body_slides_llm
            from src.scraper import DocketIssue

            # Find the original article in output/issue_*.json
            _output_dir = PROJECT_ROOT / "output"
            _article = None
            if _output_dir.exists():
                # Load most recent issue file first
                _issue_files = sorted(
                    _output_dir.glob("issue_*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for _issue_path in _issue_files:
                    try:
                        _issue = DocketIssue.load(_issue_path)
                        for _a in _issue.articles:
                            if _a.title == item.article_title:
                                _article = _a
                                break
                        if _article:
                            break
                    except Exception:
                        continue

            if _article:
                console.print(
                    f"    [dim]Found article in issue data, calling Claude...[/dim]"
                )
                _regen = _generate_body_slides_llm(_article, hook)
                if _regen:
                    # Update local variables for rendering
                    video_tier = _regen.get("video_tier", "narrative")
                    image_prompts = _regen.get("image_prompts", [])
                    background_prompt = _regen.get("background_prompt", "")

                    # Also update body_slides/voiceover if regen produced them
                    if _regen.get("body_slides"):
                        body_slides = _regen["body_slides"]
                        body_slides_from_hook = False

                    # Persist to queue item's video_script (saved on queue.save())
                    vs["video_tier"] = video_tier
                    vs["cinematic_score"] = _regen.get("cinematic_score", 0)
                    vs["image_prompts"] = image_prompts
                    vs["background_prompt"] = background_prompt
                    if _regen.get("body_slides"):
                        vs["body_slides"] = body_slides
                    if _regen.get("voiceover_lines"):
                        vs["voiceover_lines"] = _regen["voiceover_lines"]
                    if _regen.get("closing_slide"):
                        vs["cta"] = _regen["closing_slide"]
                    if _regen.get("closing_voiceover"):
                        vs["voiceover_cta"] = _regen["closing_voiceover"]

                    console.print(
                        f"    [green]Regenerated: tier={video_tier}, "
                        f"score={_regen.get('cinematic_score', '?')}, "
                        f"images={len(image_prompts)}[/green]"
                    )
                else:
                    console.print(
                        f"    [yellow]Claude regen returned None, "
                        f"falling back to title-based prompt[/yellow]"
                    )
                    # Minimal fallback: synthesize a background prompt from title
                    background_prompt = (
                        f"Atmospheric documentary photograph related to: "
                        f"{item.article_title}. Natural lighting, editorial style, "
                        f"9:16 portrait, no text or UI elements."
                    )
                    vs["background_prompt"] = background_prompt
                    vs["video_tier"] = "narrative"
            else:
                console.print(
                    f"    [yellow]Article not found in issue data, "
                    f"using title-based prompt fallback[/yellow]"
                )
                background_prompt = (
                    f"Atmospheric documentary photograph related to: "
                    f"{item.article_title}. Natural lighting, editorial style, "
                    f"9:16 portrait, no text or UI elements."
                )
                vs["background_prompt"] = background_prompt
                vs["video_tier"] = "narrative"

        except Exception as _regen_err:
            console.print(
                f"    [yellow]Regeneration failed ({_regen_err}), "
                f"using title-based prompt fallback[/yellow]"
            )
            # Minimal fallback so at least Tier 2 generates a relevant AI bg
            background_prompt = (
                f"Atmospheric documentary photograph related to: "
                f"{item.article_title}. Natural lighting, editorial style, "
                f"9:16 portrait, no text or UI elements."
            )
            vs["background_prompt"] = background_prompt
            vs["video_tier"] = "narrative"

    if body_slides_from_hook:
        script = VideoScript(
            title=title,
            hook="",  # no separate hook slide — content is in body_slides
            body_slides=body_slides,
            cta=cta,
            voiceover_text="",  # unused in per-slide mode
            section=item.section,
            video_tier=video_tier,
        )
    else:
        script = VideoScript(
            title=title,
            hook=hook,
            body_slides=body_slides,
            cta=cta,
            voiceover_text="",  # unused in per-slide mode
            section=item.section,
            video_tier=video_tier,
        )

    console.print(
        f"    [dim]Voiceover slides ({len(slide_voiceover_texts)}):[/dim]"
    )
    for lbl, txt in zip(slide_labels, slide_voiceover_texts):
        console.print(f"      [dim]{lbl}: {txt[:60]}{'...' if len(txt) > 60 else ''}[/dim]")

    with tempfile.TemporaryDirectory(prefix="docket_video_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        audio_dir = tmp_path / "audio"
        video_path = tmp_path / f"{item.id}.mp4"

        # Step 1: Generate per-slide voiceover (graceful fallback to silent)
        voiceover_paths = None
        use_per_slide = voiceover_config.get("per_slide", True)

        if use_per_slide:
            try:
                console.print(
                    f"    [dim]Generating per-slide voiceover "
                    f"({len(slide_voiceover_texts)} segments)...[/dim]"
                )
                voiceover_paths = generate_voiceover_per_slide(
                    slide_voiceover_texts, audio_dir, slide_labels,
                )
            except Exception as e:
                console.print(
                    f"    [yellow]Per-slide voiceover failed ({e}), "
                    f"continuing with silent video[/yellow]"
                )
                voiceover_paths = None
        else:
            # Legacy single-file mode
            try:
                audio_path = tmp_path / "voiceover.mp3"
                voiceover_text = " ".join(slide_voiceover_texts)
                console.print(
                    f"    [dim]Generating voiceover "
                    f"({len(voiceover_text)} chars)...[/dim]"
                )
                generate_voiceover(voiceover_text, audio_path)
                voiceover_paths = audio_path
            except Exception as e:
                console.print(
                    f"    [yellow]Voiceover failed ({e}), "
                    f"continuing with silent video[/yellow]"
                )
                voiceover_paths = None

        # Step 2: Generate AI images (Tier 1 or Tier 2)
        # Slide structure: Title, Hook, Body 1..N — each needs its own image.
        # CTA reuses the last image. Count required images and generate
        # supplemental prompts for any slides not covered by image_prompts.
        _has_hook = bool(hook.strip()) and not body_slides_from_hook
        _images_needed = 1 + (1 if _has_hook else 0) + len(body_slides)  # title + hook? + body
        if video_tier == "cinematic" and image_prompts and len(image_prompts) < _images_needed:
            _deficit = _images_needed - len(image_prompts)
            console.print(
                f"    [yellow]Need {_images_needed} images but only "
                f"{len(image_prompts)} prompts — generating {_deficit} more[/yellow]"
            )
            # Build supplemental prompts tied to slide text that lacks an image.
            # Image assignment order: title, hook, body_1..N
            # Existing prompts cover the first len(image_prompts) slides.
            _all_slide_texts = [title]
            if _has_hook:
                _all_slide_texts.append(hook.replace("\n\n", " ").strip())
            _all_slide_texts.extend(body_slides)
            # The slides that need new prompts are the tail ones
            _uncovered = _all_slide_texts[len(image_prompts):]
            for _slide_text in _uncovered:
                _supplemental = (
                    f"Atmospheric documentary photograph visualizing: "
                    f"{_slide_text}. Photojournalistic style, natural lighting, "
                    f"cinematic composition, 9:16 portrait, no text or UI elements."
                )
                image_prompts.append(_supplemental)
            console.print(
                f"    [dim]Total image prompts: {len(image_prompts)}[/dim]"
            )

        ai_config = video_config.get("ai_images", {})
        slide_image_paths = []
        background_image_path = None

        if ai_config.get("enabled", False):
            try:
                from src.video.imagegen import generate_slide_images, generate_background_image
                images_dir = tmp_path / "images"

                if video_tier == "cinematic" and image_prompts:
                    console.print(
                        f"    [cyan]Generating {len(image_prompts)} AI images "
                        f"(Tier 1: Cinematic)...[/cyan]"
                    )
                    slide_image_paths = generate_slide_images(
                        image_prompts, images_dir, ai_config,
                    )
                    # Fallback: if <50% succeeded, downgrade to narrative
                    succeeded = sum(1 for p in slide_image_paths if p is not None)
                    if succeeded < len(image_prompts) // 2:
                        console.print(
                            f"    [yellow]Only {succeeded}/{len(image_prompts)} images "
                            f"generated, downgrading to Tier 2[/yellow]"
                        )
                        video_tier = "narrative"
                        slide_image_paths = []

                if video_tier == "narrative" and background_prompt:
                    console.print(
                        f"    [cyan]Generating AI background image "
                        f"(Tier 2: Narrative)...[/cyan]"
                    )
                    background_image_path = generate_background_image(
                        background_prompt, images_dir, ai_config,
                    )
            except Exception as e:
                console.print(
                    f"    [yellow]AI image generation failed ({e}), "
                    f"falling back to stock/gradient[/yellow]"
                )
                slide_image_paths = []
                background_image_path = None

        # Step 3: Fetch stock B-roll clip (only if no AI images)
        stock_clip_path = None
        if not slide_image_paths and background_image_path is None:
            try:
                from src.video.stock import get_background_clip

                console.print(f"    [dim]Searching for stock B-roll...[/dim]")
                stock_clip_path = get_background_clip(
                    title=script.title,
                    section=script.section,
                    output_dir=tmp_path,
                )
                if stock_clip_path:
                    console.print(f"    [dim]Stock clip ready: {stock_clip_path.name}[/dim]")
                else:
                    console.print(f"    [dim]No stock clip, using gradient background.[/dim]")
            except Exception as e:
                console.print(
                    f"    [yellow]Stock footage step failed ({e}), "
                    f"using gradient background[/yellow]"
                )

        # Step 4: Generate video with per-slide audio
        # Slide structure: Title (VO + beat), Hook (VO), Body 1..N (VO), CTA (VO)
        # Each slide gets its own voiceover segment — no alignment padding needed.

        console.print(f"    [dim]Rendering video ({video_tier})...[/dim]")
        generate_video(
            script, voiceover_paths, video_path, video_config,
            stock_clip_path=stock_clip_path,
            video_tier=video_tier,
            slide_image_paths=slide_image_paths or None,
            background_image_path=background_image_path,
        )

        # Step 5: Upload to cloud storage
        storage_key = f"videos/{item.id}.mp4"
        console.print(f"    [dim]Uploading to cloud storage...[/dim]")
        public_url = upload_video(video_path, storage_key)

        # Step 5b: Generate and upload custom thumbnail/cover image
        cover_url = None
        try:
            from src.video.generator import generate_thumbnail

            thumbnail_path = tmp_path / "thumbnail.png"

            # Load the best available AI image for the thumbnail background
            thumb_ai_image = None
            if slide_image_paths:
                for p in slide_image_paths:
                    if p and Path(p).exists():
                        thumb_ai_image = Image.open(p)
                        break
            if thumb_ai_image is None and background_image_path and Path(background_image_path).exists():
                thumb_ai_image = Image.open(background_image_path)

            generate_thumbnail(script, thumbnail_path, video_config, ai_image=thumb_ai_image)
            thumb_storage_key = f"thumbnails/{item.id}.png"
            cover_url = upload_video(thumbnail_path, thumb_storage_key)
            console.print(f"    [dim]Thumbnail uploaded: {thumb_storage_key}[/dim]")
        except Exception as e:
            console.print(f"    [yellow]Thumbnail generation failed (non-fatal): {e}[/yellow]")
            cover_url = None

        # Step 6: Publish to platform
        if item.platform == "reels":
            from src.publishers.reels import ReelsPublisher

            publisher = ReelsPublisher()
            publisher.login()
            caption = item.text  # Hook + hashtags set by scheduler
            console.print(f"    [dim]Publishing to Reels...[/dim]")
            result = publisher.publish_video(public_url, caption, cover_url=cover_url)
            media_id = result.get("media_id", "")
            return True, f"reels:{media_id}"

        elif item.platform == "tiktok":
            from src.publishers.tiktok import TikTokPublisher

            publisher = TikTokPublisher()
            publisher.login()
            caption = item.text
            console.print(f"    [dim]Publishing to TikTok...[/dim]")
            result = publisher.publish_video(str(video_path), caption)
            publish_id = result.get("publish_id", "")
            return True, f"tiktok:{publish_id}"

        else:
            return False, f"Unsupported video platform: {item.platform}"


def _get_publisher(platform_name: str):
    """Dynamically load the publisher for a platform (reuses main.py pattern)."""
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


def _log_daily_run(posted: int, failed: int, pending: int, issue_number: int | None):
    """Log the daily run results."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "post-today",
        "issue_number": issue_number,
        "posted": posted,
        "failed": failed,
        "pending": pending,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
