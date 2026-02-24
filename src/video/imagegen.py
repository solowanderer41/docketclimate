"""
AI image generation for Docket Social using Flux 2 Pro via BFL API.

Generates photorealistic editorial imagery for Instagram Reels in two modes:
    - Tier 1 (Cinematic): One unique image per body slide
    - Tier 2 (Narrative): A single article-themed background image

BFL API docs: https://docs.bfl.ml/
Flux 2 Pro: Photorealistic image model, native 9:16 at 1088×1920.

Required environment variable:
    BFL_API_KEY — API key from https://api.bfl.ai/
"""

import hashlib
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

BFL_API_KEY = os.getenv("BFL_API_KEY")
BFL_GENERATE_URL = "https://api.bfl.ai/v1/flux-2-pro"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "ai_image_cache"


def _get_cache_path(prompt: str, cache_dir: Path) -> Path:
    """Return a deterministic cache path based on the SHA256 of the full prompt."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.png"


def _build_full_prompt(user_prompt: str, style_prefix: str) -> str:
    """Prepend the style guide to the user's image prompt."""
    prefix = style_prefix.strip()
    user = user_prompt.strip()
    if prefix:
        return f"{prefix}. {user}"
    return user


def _generate_single_image(
    prompt: str,
    output_path: Path,
    *,
    api_key: str,
    width: int = 1088,
    height: int = 1920,
    timeout: int = 30,
    max_retries: int = 3,
) -> Path:
    """Submit a generation request to BFL, poll for completion, download the image.

    Args:
        prompt: The full prompt (style prefix + user prompt).
        output_path: Where to save the downloaded PNG.
        api_key: BFL API key.
        width: Image width in pixels.
        height: Image height in pixels.
        timeout: HTTP timeout per request in seconds.
        max_retries: Number of retries on transient failures.

    Returns:
        Path to the downloaded image.

    Raises:
        RuntimeError: If generation fails after all retries.
    """
    headers = {"x-key": api_key, "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # 1. Submit generation request
            resp = httpx.post(
                BFL_GENERATE_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            task = resp.json()
            task_id = task.get("id", "unknown")
            polling_url = task.get("polling_url")

            if not polling_url:
                raise RuntimeError(f"BFL API returned no polling_url: {task}")

            console.print(f"[dim]BFL task {task_id[:8]}... submitted (attempt {attempt})[/dim]")

            # 2. Poll for completion
            poll_headers = {"x-key": api_key}
            max_polls = 120  # 60 seconds at 0.5s intervals
            for _ in range(max_polls):
                time.sleep(0.5)
                poll_resp = httpx.get(polling_url, headers=poll_headers, timeout=timeout)
                poll_resp.raise_for_status()
                result = poll_resp.json()
                status = result.get("status")

                if status == "Ready":
                    image_url = result.get("result", {}).get("sample")
                    if not image_url:
                        raise RuntimeError(f"BFL returned Ready but no sample URL: {result}")

                    # 3. Download image (signed URL, 10-min expiry)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    img_resp = httpx.get(image_url, timeout=timeout, follow_redirects=True)
                    img_resp.raise_for_status()
                    output_path.write_bytes(img_resp.content)

                    size_kb = output_path.stat().st_size / 1024
                    console.print(
                        f"[dim]AI image downloaded: {output_path.name} ({size_kb:.0f} KB)[/dim]"
                    )
                    return output_path

                if status in ("Error", "Failed", "Content Moderated"):
                    raise RuntimeError(f"BFL generation failed: {status} — {result}")

            raise RuntimeError(f"BFL polling timed out after {max_polls * 0.5:.0f}s")

        except httpx.HTTPStatusError as e:
            last_error = e
            console.print(f"[yellow]BFL attempt {attempt}/{max_retries} failed: {e}[/yellow]")
            # Don't retry non-transient HTTP errors (billing, auth, forbidden)
            if e.response.status_code in (401, 402, 403):
                break
            if attempt < max_retries:
                time.sleep(2 * attempt)  # exponential backoff

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            console.print(f"[yellow]BFL attempt {attempt}/{max_retries} failed: {e}[/yellow]")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # exponential backoff

    raise RuntimeError(f"BFL image generation failed after {max_retries} attempts: {last_error}")


def generate_slide_images(
    prompts: list[str],
    output_dir: Path,
    config: dict,
) -> list[Path | None]:
    """Generate one AI image per prompt (Tier 1: Cinematic mode).

    Never raises — returns None for any individual image that fails,
    so the video pipeline can fall back gracefully.

    Args:
        prompts: List of image prompts (one per body slide).
        output_dir: Directory for downloaded images.
        config: The ``ai_images`` config dict from config.yaml.

    Returns:
        List of Paths (or None for failures), same length as prompts.
    """
    if not BFL_API_KEY:
        console.print("[dim]No BFL_API_KEY set, skipping AI image generation.[/dim]")
        return [None] * len(prompts)

    style_prefix = config.get("style_prefix", "")
    width = config.get("width", 1088)
    height = config.get("height", 1920)
    timeout = config.get("timeout_seconds", 30)
    max_retries = config.get("max_retries", 3)
    cache_dir = Path(config.get("cache_dir", DEFAULT_CACHE_DIR))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path | None] = []

    console.print(f"[cyan]Generating {len(prompts)} AI images (Tier 1: Cinematic)...[/]")

    for i, prompt in enumerate(prompts):
        full_prompt = _build_full_prompt(prompt, style_prefix)

        # Check cache
        cached = _get_cache_path(full_prompt, cache_dir)
        if cached.exists():
            console.print(f"  [dim]Slide {i + 1}/{len(prompts)}: cache hit[/dim]")
            results.append(cached)
            continue

        try:
            console.print(f"  [cyan]Slide {i + 1}/{len(prompts)}:[/] generating...")
            path = _generate_single_image(
                full_prompt,
                cached,
                api_key=BFL_API_KEY,
                width=width,
                height=height,
                timeout=timeout,
                max_retries=max_retries,
            )
            results.append(path)
        except Exception as e:
            console.print(f"  [yellow]Slide {i + 1} failed: {e}[/yellow]")
            results.append(None)

    succeeded = sum(1 for r in results if r is not None)
    console.print(f"[cyan]AI images: {succeeded}/{len(prompts)} generated successfully.[/]")
    return results


def generate_background_image(
    prompt: str,
    output_dir: Path,
    config: dict,
) -> Path | None:
    """Generate a single AI background image (Tier 2: Narrative mode).

    Never raises — returns None on failure so the video pipeline
    can fall back to stock footage or gradient.

    Args:
        prompt: Image prompt describing the article's visual theme.
        output_dir: Directory for downloaded image.
        config: The ``ai_images`` config dict from config.yaml.

    Returns:
        Path to the generated image, or None if unavailable.
    """
    if not BFL_API_KEY:
        console.print("[dim]No BFL_API_KEY set, skipping AI background image.[/dim]")
        return None

    style_prefix = config.get("style_prefix", "")
    width = config.get("width", 1088)
    height = config.get("height", 1920)
    timeout = config.get("timeout_seconds", 30)
    max_retries = config.get("max_retries", 3)
    cache_dir = Path(config.get("cache_dir", DEFAULT_CACHE_DIR))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    full_prompt = _build_full_prompt(prompt, style_prefix)

    # Check cache
    cached = _get_cache_path(full_prompt, cache_dir)
    if cached.exists():
        console.print("[dim]AI background image: cache hit[/dim]")
        return cached

    try:
        console.print("[cyan]Generating AI background image (Tier 2: Narrative)...[/]")
        return _generate_single_image(
            full_prompt,
            cached,
            api_key=BFL_API_KEY,
            width=width,
            height=height,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as e:
        console.print(f"[yellow]AI background image failed: {e}[/yellow]")
        return None
