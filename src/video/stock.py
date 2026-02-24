"""
Pexels stock video integration for Docket Social.

Searches the Pexels API for portrait-oriented stock video clips to use
as looping B-roll backgrounds in Instagram Reels. Downloads are cached
locally to avoid re-fetching the same clip.

Pexels API docs: https://www.pexels.com/api/documentation/
License: Free for commercial use, no attribution required (but appreciated).

Required environment variable:
    PEXELS_API_KEY — Free API key from https://www.pexels.com/api/
"""

import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "stock_cache"
CACHE_MAX_AGE_DAYS = 7

# Default search terms per newsletter section.
# Combined with keywords extracted from article titles.
SECTION_KEYWORDS = {
    "lived": "community people homes neighborhood daily life",
    "systems": "infrastructure power grid industry",
    "science": "nature environment landscape ocean weather",
    "futures": "renewable energy solar technology",
    "archive": "historical documentary landscape",
    "lab": "innovation technology green startup",
}

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "for", "and", "or", "on", "at", "by", "with", "from", "as", "how",
    "our", "its", "this", "that", "it", "be", "has", "have", "can",
    "could", "will", "would", "but", "not", "what", "why", "when",
    "where", "who", "which", "there", "nothing", "changes", "about",
    "meeting", "more", "than", "just", "also", "into", "been", "being",
    "do", "does", "did", "done", "no", "so", "if", "than", "too",
})


def extract_keywords(title: str, section: str) -> str:
    """Build a Pexels search query from article title and section.

    Combines section default keywords with 2-3 meaningful words from
    the title (stop words removed).
    """
    section_terms = SECTION_KEYWORDS.get(section, "")

    # Extract words from title, remove punctuation and stop words
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    title_terms = " ".join(meaningful[:5])

    query = f"{section_terms} {title_terms}".strip()
    return query if query else "nature atmosphere climate"


def search_stock_video(
    query: str,
    orientation: str = "portrait",
    per_page: int = 5,
) -> list[dict]:
    """Search Pexels for stock video clips.

    Args:
        query: Search terms (e.g., "urban city heat waves").
        orientation: "portrait", "landscape", or "square".
        per_page: Number of results (max 80).

    Returns:
        List of video dicts from the Pexels API response.
    """
    if not PEXELS_API_KEY:
        raise EnvironmentError(
            "PEXELS_API_KEY not set. Get a free key at https://www.pexels.com/api/"
        )

    response = httpx.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": PEXELS_API_KEY},
        params={
            "query": query,
            "orientation": orientation,
            "size": "medium",
            "per_page": per_page,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("videos", [])


def download_stock_clip(
    video_files: list[dict],
    output_path: Path,
    preferred_quality: str = "hd",
) -> Path:
    """Download the best-matching video file from a Pexels video result.

    Prefers the requested quality level. Falls back to the largest
    available file.

    Args:
        video_files: The ``video_files`` list from a Pexels video object.
        output_path: Where to save the downloaded MP4.
        preferred_quality: "hd" or "uhd".

    Returns:
        Path to the downloaded file.
    """
    # Try to find the preferred quality
    target = None
    for vf in video_files:
        if vf.get("quality") == preferred_quality:
            target = vf
            break

    # Fallback: pick the file with the highest resolution
    if target is None:
        target = max(video_files, key=lambda vf: vf.get("width", 0) * vf.get("height", 0))

    download_url = target["link"]
    console.print(
        f"[dim]Downloading stock clip: "
        f"{target.get('width')}x{target.get('height')} "
        f"({target.get('quality', '?')})[/dim]"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", download_url, timeout=60, follow_redirects=True) as stream:
        stream.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in stream.iter_bytes(chunk_size=65536):
                f.write(chunk)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[dim]Stock clip downloaded: {output_path.name} ({size_mb:.1f} MB)[/dim]")
    return output_path


def clean_cache(max_age_days: int = CACHE_MAX_AGE_DAYS):
    """Remove cached stock clips older than max_age_days."""
    if not CACHE_DIR.exists():
        return

    cutoff = time.time() - (max_age_days * 86400)
    removed = 0
    for f in CACHE_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1

    if removed:
        console.print(f"[dim]Cleaned {removed} expired stock clips from cache.[/dim]")


def get_background_clip(
    title: str,
    section: str,
    output_dir: Path,
) -> Path | None:
    """Search, download, and cache a stock B-roll clip for a video.

    This is the main entry point. It never raises — returns None on
    any failure so the video pipeline can fall back to gradient backgrounds.

    Args:
        title: Article title (used for keyword extraction).
        section: Newsletter section ID (e.g., "lived", "science").
        output_dir: Working directory (unused if cache hit).

    Returns:
        Path to the stock video clip, or None if unavailable.
    """
    if not PEXELS_API_KEY:
        console.print("[dim]No PEXELS_API_KEY set, skipping stock footage.[/dim]")
        return None

    try:
        # Opportunistic cache cleanup
        clean_cache()

        # Build search query
        query = extract_keywords(title, section)
        console.print(f"[dim]Stock search: \"{query}\"[/dim]")

        # Search Pexels
        videos = search_stock_video(query, orientation="portrait", per_page=5)

        if not videos:
            # Retry with just section keywords (broader search)
            fallback_query = SECTION_KEYWORDS.get(section, "nature atmosphere")
            console.print(f"[dim]No results, retrying: \"{fallback_query}\"[/dim]")
            videos = search_stock_video(fallback_query, orientation="portrait", per_page=5)

        if not videos:
            console.print("[yellow]No stock footage found on Pexels.[/yellow]")
            return None

        # Pick the first result
        video = videos[0]
        video_id = video["id"]

        # Check cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = CACHE_DIR / f"{video_id}.mp4"
        if cached_path.exists():
            console.print(f"[dim]Stock clip cache hit: {cached_path.name}[/dim]")
            return cached_path

        # Download
        video_files = video.get("video_files", [])
        if not video_files:
            console.print("[yellow]Stock video has no downloadable files.[/yellow]")
            return None

        return download_stock_clip(video_files, cached_path, preferred_quality="hd")

    except Exception as e:
        console.print(f"[yellow]Stock footage error: {e}[/yellow]")
        return None
