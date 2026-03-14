"""
After Effects cloud rendering for Docket Social via Nexrender.

Replaces the MoviePy video generation step with professional AE template
rendering. Designed as a drop-in alternative to generator.generate_video()
with the same interface contract.

Data flow:
    VideoScript → build_nexrender_job() → Nexrender REST API → poll → download MP4

Rendering backend:
    Nexrender (open-source, self-hosted): REST API on a machine with AE installed.
    GitHub: https://github.com/inlife/nexrender

Required environment variables:
    NEXRENDER_SERVER_URL — e.g. http://localhost:3000
    NEXRENDER_SECRET     — shared secret for the nexrender server
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx
from dotenv import load_dotenv
from rich.console import Console

if TYPE_CHECKING:
    from src.video.generator import VideoScript

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# Section color mapping (mirrors config.yaml video.visual.section_colors)
# ---------------------------------------------------------------------------

DEFAULT_SECTION_COLORS = {
    "lived": "#e07a5f",
    "systems": "#f2cc8f",
    "science": "#4ecdc4",
    "futures": "#7b68ee",
    "archive": "#a0937d",
    "lab": "#81b29a",
    "default": "#4ecdc4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_str: str) -> List[float]:
    """Convert a hex color string to a normalized RGBA array for After Effects.

    AE uses 0.0–1.0 float values for color channels.

    Args:
        hex_str: Color in "#RRGGBB" or "RRGGBB" format.

    Returns:
        [r, g, b, a] with values in 0.0–1.0 range. Alpha is always 1.0.

    Examples:
        >>> _hex_to_rgba("#4ecdc4")
        [0.306, 0.804, 0.769, 1.0]
    """
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: #{hex_str}")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return [round(r, 3), round(g, 3), round(b, 3), 1.0]


# ---------------------------------------------------------------------------
# Nexrender REST API client
# ---------------------------------------------------------------------------


class NexrenderClient:
    """Thin wrapper around the Nexrender server REST API.

    Nexrender docs: https://github.com/inlife/nexrender
    Authentication: Shared secret via ``nexrender-secret`` header.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        secret: Optional[str] = None,
        timeout: int = 30,
    ):
        self.server_url = (
            server_url or os.getenv("NEXRENDER_SERVER_URL", "")
        ).rstrip("/")
        self.secret = secret or os.getenv("NEXRENDER_SECRET", "")

        if not self.server_url:
            raise EnvironmentError(
                "NEXRENDER_SERVER_URL not set. Add it to .env — "
                "see .env.example for details."
            )

        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["nexrender-secret"] = self.secret

        self._client = httpx.Client(
            base_url=self.server_url,
            headers=headers,
            timeout=timeout,
        )

    # -- job lifecycle --------------------------------------------------------

    def create_job(self, job_payload: dict) -> dict:
        """Submit a new render job.

        Args:
            job_payload: Full nexrender job JSON (template, assets, actions).

        Returns:
            Job dict including ``uid`` and ``state``.
        """
        resp = self._client.post("/api/v1/jobs", json=job_payload)
        resp.raise_for_status()
        return resp.json()

    def get_job(self, job_uid: str) -> dict:
        """Poll a render job's current state.

        Returns:
            Job dict with ``state`` ("queued", "started", "finished", "error")
            and, when finished, ``output`` containing the download URL/path.
        """
        resp = self._client.get(f"/api/v1/jobs/{job_uid}")
        resp.raise_for_status()
        return resp.json()

    def download_output(self, job_uid: str, output_path: Path) -> Path:
        """Download the rendered MP4 to a local path.

        Uses the output URL from the completed job.

        Returns:
            The output_path where the file was saved.
        """
        job = self.get_job(job_uid)
        output_url = job.get("output", "")
        if not output_url:
            raise RuntimeError(
                f"Job {job_uid} has no output URL. "
                f"State: {job.get('state')}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with httpx.stream("GET", output_url, timeout=120) as stream:
            stream.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in stream.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        return output_path

    def wait_for_job(
        self,
        job_uid: str,
        poll_interval: int = 10,
        timeout: int = 600,
    ) -> dict:
        """Poll until the render job completes or times out.

        Args:
            job_uid: The job UID.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            The final job dict.

        Raises:
            TimeoutError: If the job doesn't finish in time.
            RuntimeError: If the job fails.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job(job_uid)
            state = job.get("state", "unknown")

            if state == "finished":
                return job
            if state == "error":
                error = job.get("errorMessage", "Unknown error")
                raise RuntimeError(
                    f"Nexrender job {job_uid} failed: {error}"
                )

            console.print(
                f"    [dim]Render state: {state} "
                f"(elapsed {int(time.time() - (deadline - timeout))}s)[/dim]"
            )
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Nexrender job {job_uid} timed out after {timeout}s"
        )

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

# Template keys are configured in config.yaml under video.aftereffects.templates
TEMPLATE_KEYS = ("cinematic", "narrative", "stock_broll", "gradient")


def select_template(
    video_tier: str,
    has_slide_images: bool,
    has_background_image: bool,
    has_stock_clip: bool,
    templates: Dict[str, str],
) -> str:
    """Choose the appropriate AE template based on available assets.

    Selection logic mirrors the MoviePy generator's tier/fallback chain:
        1. cinematic — if tier is "cinematic" and per-slide AI images exist
        2. narrative — if tier is "narrative" and a background image exists
        3. stock_broll — if stock footage is available (any tier)
        4. gradient — fallback when no external imagery is available

    Args:
        video_tier: "cinematic" or "narrative".
        has_slide_images: True if per-slide AI images are available.
        has_background_image: True if a single background AI image is available.
        has_stock_clip: True if stock B-roll footage is available.
        templates: Dict mapping template keys to template URLs (R2-hosted .aep).

    Returns:
        The template URL to use.

    Raises:
        ValueError: If the selected template key has no configured URL.
    """
    if video_tier == "cinematic" and has_slide_images:
        key = "cinematic"
    elif has_background_image:
        key = "narrative"
    elif has_stock_clip:
        key = "stock_broll"
    else:
        key = "gradient"

    template_url = templates.get(key)
    if not template_url:
        raise ValueError(
            f"No template URL configured for '{key}'. "
            f"Set video.aftereffects.templates.{key} in config.yaml."
        )
    return template_url


# ---------------------------------------------------------------------------
# Asset staging
# ---------------------------------------------------------------------------


def upload_assets_for_render(
    voiceover_paths: "Optional[List[Optional[Path]]]" = None,
    slide_image_paths: Optional[List[Optional[Path]]] = None,
    background_image_path: Optional[Path] = None,
    stock_clip_path: Optional[Path] = None,
) -> dict:
    """Stage local assets to publicly accessible URLs for nexrender.

    Uses the existing R2 uploader to push files to cloud storage and
    return public HTTPS URLs that the nexrender worker can fetch during rendering.

    Returns:
        Dict with keys: "voiceover_urls", "slide_image_urls",
        "background_image_url", "stock_clip_url".
    """
    from src.video.uploader import upload_video

    result = {
        "voiceover_urls": [],
        "slide_image_urls": [],
        "background_image_url": None,
        "stock_clip_url": None,
    }

    # Voiceover audio files
    if isinstance(voiceover_paths, list):
        for i, path in enumerate(voiceover_paths):
            if path and Path(path).exists():
                key = f"ae_staging/audio/slide_{i}.mp3"
                url = upload_video(Path(path), key)
                result["voiceover_urls"].append(url)
            else:
                result["voiceover_urls"].append(None)
    elif voiceover_paths and Path(voiceover_paths).exists():
        key = "ae_staging/audio/voiceover.mp3"
        url = upload_video(Path(voiceover_paths), key)
        result["voiceover_urls"].append(url)

    # Per-slide AI images (Tier 1: Cinematic)
    if slide_image_paths:
        for i, path in enumerate(slide_image_paths):
            if path and Path(path).exists():
                key = f"ae_staging/images/slide_{i}.png"
                url = upload_video(Path(path), key)
                result["slide_image_urls"].append(url)
            else:
                result["slide_image_urls"].append(None)

    # Single background image (Tier 2: Narrative)
    if background_image_path and Path(background_image_path).exists():
        key = "ae_staging/images/background.png"
        result["background_image_url"] = upload_video(
            Path(background_image_path), key
        )

    # Stock B-roll clip
    if stock_clip_path and Path(stock_clip_path).exists():
        key = "ae_staging/stock/broll.mp4"
        result["stock_clip_url"] = upload_video(
            Path(stock_clip_path), key
        )

    return result


# ---------------------------------------------------------------------------
# Nexrender job builder
# ---------------------------------------------------------------------------


def build_nexrender_job(
    script: "VideoScript",
    asset_urls: dict,
    video_config: dict,
    video_tier: str = "narrative",
    template_url: str = "",
    composition: str = "main",
) -> dict:
    """Convert a VideoScript + staged asset URLs into a nexrender job payload.

    Maps the structured content to AE template dynamic layers using the
    naming convention defined in templates/ae/README.md.

    Args:
        script: The structured video content.
        asset_urls: Output of upload_assets_for_render().
        video_config: The ``video`` section of config.yaml.
        video_tier: "cinematic" or "narrative".
        template_url: R2 URL to the .aep template file.
        composition: AE composition name to render (default "main").

    Returns:
        Full nexrender job dict with template, assets, and actions.
    """
    visual_config = video_config.get("visual", {})
    section_colors = visual_config.get("section_colors", DEFAULT_SECTION_COLORS)
    section_color = section_colors.get(
        script.section, section_colors.get("default", "#4ecdc4")
    )

    assets = []  # type: List[dict]

    # --- Text layers ---
    text_layers = {
        "title_text": script.title,
        "hook_text": script.hook or "",
        "cta_text": script.cta or "",
    }
    for layer_name, value in text_layers.items():
        assets.append({
            "type": "data",
            "layerName": layer_name,
            "property": "Source Text",
            "value": value,
        })

    # Body slide text
    for i, slide_text in enumerate(script.body_slides):
        assets.append({
            "type": "data",
            "layerName": f"body_text_{i + 1}",
            "property": "Source Text",
            "value": slide_text,
        })
    # Clear unused body slots (templates may have up to 5)
    for i in range(len(script.body_slides), 5):
        assets.append({
            "type": "data",
            "layerName": f"body_text_{i + 1}",
            "property": "Source Text",
            "value": "",
        })

    # --- Numeric control layers ---
    numeric_layers = {
        "num_body_slides": len(script.body_slides),
        "voiceover_padding": video_config.get("voiceover_padding", 0.3),
        "min_slide_duration": video_config.get("min_slide_duration", 1.5),
        "default_slide_duration": video_config.get("duration_per_slide", 5),
    }
    for layer_name, value in numeric_layers.items():
        assets.append({
            "type": "data",
            "layerName": layer_name,
            "property": "Effects.Slider Control.Slider",
            "value": value,
        })

    # --- Color theming ---
    color_layers = {
        "section_color": section_color,
        "progress_bar_color": section_color,
        "background_color": video_config.get("background_color", "#1a1a2e"),
        "text_color": video_config.get("text_color", "#e0e0e0"),
    }
    for layer_name, hex_val in color_layers.items():
        assets.append({
            "type": "data",
            "layerName": layer_name,
            "property": "Effects.Fill.Color",
            "value": _hex_to_rgba(hex_val),
        })

    # --- Audio layers ---
    voiceover_urls = asset_urls.get("voiceover_urls", [])
    for i, url in enumerate(voiceover_urls):
        if url:
            assets.append({
                "type": "audio",
                "layerName": f"audio_slide_{i}",
                "src": url,
            })

    # --- Tier-specific parameters ---
    if video_tier == "cinematic":
        cinematic_config = video_config.get("cinematic", {})
        cinematic_params = {
            "ken_burns_zoom": cinematic_config.get("ken_burns_zoom", 1.12),
            "ken_burns_pan_px": cinematic_config.get("ken_burns_pan_px", 50),
            "crossfade_duration": cinematic_config.get("crossfade_duration", 0.5),
            "subtitle_font_size": cinematic_config.get("subtitle_font_size", 44),
            "subtitle_y_fraction": cinematic_config.get("subtitle_y_fraction", 0.72),
        }
        for layer_name, value in cinematic_params.items():
            assets.append({
                "type": "data",
                "layerName": layer_name,
                "property": "Effects.Slider Control.Slider",
                "value": value,
            })

        # Per-slide AI images
        slide_image_urls = asset_urls.get("slide_image_urls", [])
        for i, url in enumerate(slide_image_urls):
            if url:
                assets.append({
                    "type": "image",
                    "layerName": f"slide_image_{i}",
                    "src": url,
                })

    elif video_tier == "narrative":
        narrative_config = video_config.get("narrative", {})
        assets.append({
            "type": "data",
            "layerName": "overlay_opacity",
            "property": "Effects.Slider Control.Slider",
            "value": narrative_config.get("overlay_opacity", 0.55),
        })

        bg_url = asset_urls.get("background_image_url")
        if bg_url:
            assets.append({
                "type": "image",
                "layerName": "background_image",
                "src": bg_url,
            })

    # Stock footage URL (for stock_broll template)
    stock_url = asset_urls.get("stock_clip_url")
    if stock_url:
        assets.append({
            "type": "video",
            "layerName": "stock_footage",
            "src": stock_url,
        })
        stock_config = visual_config.get("stock_footage", {})
        assets.append({
            "type": "data",
            "layerName": "overlay_opacity",
            "property": "Effects.Slider Control.Slider",
            "value": stock_config.get("overlay_opacity", 0.55),
        })

    # --- Watermark ---
    watermark_config = visual_config.get("watermark", {})
    assets.append({
        "type": "data",
        "layerName": "watermark_text",
        "property": "Source Text",
        "value": watermark_config.get("text", "THE DOCKET"),
    })

    # --- Build encoding params from config ---
    encoding_config = video_config.get("aftereffects", {}).get("encoding", {})
    encode_params = {
        "-c:v": "libx264",
        "-preset": encoding_config.get("preset", "slow"),
        "-crf": str(encoding_config.get("crf", 18)),
        "-pix_fmt": encoding_config.get("pixel_format", "yuv420p"),
        "-c:a": "aac",
        "-b:a": encoding_config.get("audio_bitrate", "192k"),
    }

    # --- Assemble full job ---
    job = {
        "template": {
            "src": template_url,
            "composition": composition,
        },
        "assets": assets,
        "actions": {
            "postrender": [
                {
                    "module": "@nexrender/action-encode",
                    "output": "output.mp4",
                    "params": encode_params,
                }
            ],
        },
    }

    return job


# ---------------------------------------------------------------------------
# Top-level render function (matches generate_video() interface)
# ---------------------------------------------------------------------------


def render_video(
    script: "VideoScript",
    voiceover_path: "Optional[List[Optional[Path]]]" = None,
    output_path: Optional[Path] = None,
    config: Optional[dict] = None,
    stock_clip_path: Optional[Path] = None,
    video_tier: str = "narrative",
    slide_image_paths: Optional[List[Optional[Path]]] = None,
    background_image_path: Optional[Path] = None,
) -> Path:
    """Generate a video via After Effects cloud rendering.

    Drop-in replacement for generator.generate_video() — same signature,
    same return value. Stages assets to R2, submits a nexrender job,
    polls for completion, and downloads the finished MP4.

    Args:
        script: The structured content for the video.
        voiceover_path: Per-slide audio paths (list), single Path, or None.
        output_path: Where to write the final MP4.
        config: Video config dict (the ``video`` section of config.yaml).
        stock_clip_path: Path to stock B-roll clip, or None.
        video_tier: "cinematic" or "narrative".
        slide_image_paths: List of AI image paths (Tier 1).
        background_image_path: Single background image path (Tier 2).

    Returns:
        The path to the exported MP4 file.

    Raises:
        EnvironmentError: If NEXRENDER_SERVER_URL is not configured.
        TimeoutError: If the render doesn't finish within the timeout.
        RuntimeError: If the render fails.
    """
    import yaml

    if config is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        config = full_config.get("video", {})

    if output_path is None:
        output_path = Path("output") / "ae_render.mp4"

    ae_config = config.get("aftereffects", {})
    templates = ae_config.get("templates", {})
    composition = ae_config.get("composition", "main")
    poll_interval = ae_config.get("poll_interval_seconds", 10)
    render_timeout = ae_config.get("render_timeout_seconds", 600)

    # Step 1: Stage assets to R2 (public URLs for nexrender to fetch)
    console.print("    [dim]Staging assets for AE render...[/dim]")
    asset_urls = upload_assets_for_render(
        voiceover_paths=voiceover_path,
        slide_image_paths=slide_image_paths,
        background_image_path=background_image_path,
        stock_clip_path=stock_clip_path,
    )

    staged_count = (
        len([u for u in asset_urls["voiceover_urls"] if u])
        + len([u for u in asset_urls["slide_image_urls"] if u])
        + (1 if asset_urls["background_image_url"] else 0)
        + (1 if asset_urls["stock_clip_url"] else 0)
    )
    console.print(f"    [dim]Staged {staged_count} assets to R2[/dim]")

    # Step 2: Select template
    template_url = select_template(
        video_tier=video_tier,
        has_slide_images=bool(slide_image_paths),
        has_background_image=background_image_path is not None,
        has_stock_clip=stock_clip_path is not None,
        templates=templates,
    )
    console.print(f"    [dim]Using AE template: {template_url}[/dim]")

    # Step 3: Build nexrender job payload
    job_payload = build_nexrender_job(
        script=script,
        asset_urls=asset_urls,
        video_config=config,
        video_tier=video_tier,
        template_url=template_url,
        composition=composition,
    )

    # Step 4: Submit render job
    console.print("    [cyan]Submitting AE render to nexrender...[/cyan]")
    client = NexrenderClient()
    try:
        render_job = client.create_job(job_payload)
        job_uid = render_job["uid"]
        console.print(f"    [dim]Render job: {job_uid}[/dim]")

        # Step 5: Poll for completion
        render_result = client.wait_for_job(
            job_uid,
            poll_interval=poll_interval,
            timeout=render_timeout,
        )
        console.print(
            f"    [green]AE render complete "
            f"(state: {render_result.get('state', '?')})[/green]"
        )

        # Step 6: Download rendered MP4
        console.print(f"    [dim]Downloading rendered video...[/dim]")
        client.download_output(job_uid, output_path)
        console.print(f"    [green]Saved: {output_path}[/green]")

    finally:
        client.close()

    return output_path
