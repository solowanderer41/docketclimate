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


def _wrap_text(text: str, font_size: int, box_width: int = 920) -> str:
    """Wrap text with newlines so it fits within a given pixel width.

    AE point text doesn't auto-wrap, and AE 2026 doesn't allow setting
    boxText via ExtendScript. So we pre-wrap before injection.

    Uses an approximation: Arial at a given font size has an average
    character width of ~0.55× the font size.

    Args:
        text: The raw text string.
        font_size: Font size in px used in the AE layer.
        box_width: Usable pixel width (default 920 = 1080 - 80px margins).

    Returns:
        Text with ``\\n`` inserted at word boundaries.
    """
    avg_char_w = font_size * 0.55
    max_chars = int(box_width / avg_char_w)
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        if len(test) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return "\n".join(lines)


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

    # --- Text layers (pre-wrapped for AE point text) ---
    # Font sizes must match the AE template layer sizes
    LAYER_FONT_SIZES = {
        "title_text": 64,
        "hook_text": 64,
        "cta_text": 48,
        "closing_text": 58,
    }
    BODY_FONT_SIZE = 54

    text_layers = {
        "title_text": script.title,
        "hook_text": script.hook or "",
        "cta_text": script.cta or "",
    }
    for layer_name, value in text_layers.items():
        fs = LAYER_FONT_SIZES.get(layer_name, 54)
        assets.append({
            "type": "data",
            "layerName": layer_name,
            "property": "Source Text",
            "value": _wrap_text(value, fs),
        })

    # Body slide text
    for i, slide_text in enumerate(script.body_slides):
        assets.append({
            "type": "data",
            "layerName": f"body_text_{i + 1}",
            "property": "Source Text",
            "value": _wrap_text(slide_text, BODY_FONT_SIZE),
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

        # Per-slide AI images (template layers are 1-indexed: bg_image_1, bg_image_2, ...)
        slide_image_urls = asset_urls.get("slide_image_urls", [])
        for i, url in enumerate(slide_image_urls):
            if url:
                assets.append({
                    "type": "image",
                    "layerName": f"bg_image_{i + 1}",
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


def _swap_desktop_assets(
    slide_image_paths: Optional[List[Optional[Path]]] = None,
    voiceover_text: Optional[str] = None,
    voiceover_path: Optional[Path] = None,
) -> None:
    """Swap Desktop placeholder files with real assets before AE render.

    AE 2026 broke replaceSource() in ExtendScript, so we can't swap footage
    at render time. Instead, the .aep template references fixed Desktop paths
    (placeholder_1.png, voiceover.mp3, etc.) and we overwrite those files
    before submitting the render job.

    Args:
        slide_image_paths: AI image paths to copy (resized to 1080x1920).
        voiceover_text: Text to generate TTS audio from.
        voiceover_path: Pre-generated voiceover MP3 path (skips TTS).
    """
    from PIL import Image

    desktop = Path.home() / "Desktop"

    # Swap slide images (resize to exact 1080x1920 — dimension mismatch
    # breaks the AE footage items and causes black frames)
    if slide_image_paths:
        for i, img_path in enumerate(slide_image_paths, 1):
            dst = desktop / f"placeholder_{i}.png"
            if img_path and Path(img_path).exists():
                img = Image.open(img_path)
                if img.size != (1080, 1920):
                    img = img.resize((1080, 1920), Image.LANCZOS)
                img.save(dst)
                console.print(f"    [dim]Swapped placeholder_{i}.png[/dim]")
            else:
                # Restore dark placeholder if no image provided
                img = Image.new("RGB", (1080, 1920), (18, 18, 18))
                img.save(dst)

    # Swap voiceover audio
    vo_dst = desktop / "voiceover.mp3"
    if voiceover_path and Path(voiceover_path).exists():
        import shutil
        shutil.copy2(voiceover_path, vo_dst)
        console.print(f"    [dim]Copied voiceover to Desktop[/dim]")
    elif voiceover_text:
        from src.video.voiceover import generate_voiceover
        generate_voiceover(voiceover_text, vo_dst)
        console.print(f"    [dim]Generated voiceover to Desktop[/dim]")


def _restore_desktop_placeholders() -> None:
    """Restore clean dark placeholders after render completes."""
    from PIL import Image

    desktop = Path.home() / "Desktop"
    for name in ["title", "hook", "1", "2", "3", "closing"]:
        img = Image.new("RGB", (1080, 1920), (18, 18, 18))
        img.save(desktop / f"placeholder_{name}.png")


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
    """Generate a video via After Effects + Nexrender.

    Drop-in replacement for generator.generate_video() — same signature,
    same return value.

    Uses Desktop-path-swap approach for AE 2026 compatibility:
    1. Copy AI images (resized to 1080x1920) to Desktop placeholder paths
    2. Generate/copy voiceover MP3 to Desktop
    3. Submit nexrender job with text-only data assets
    4. Poll for completion, copy output to output_path
    5. Restore clean placeholders

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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ae_config = config.get("aftereffects", {})
    composition = ae_config.get("composition", "main")
    poll_interval = ae_config.get("poll_interval_seconds", 10)
    render_timeout = ae_config.get("render_timeout_seconds", 600)

    # Template URL from R2
    storage_public_url = os.getenv("STORAGE_PUBLIC_URL", "")
    template_url = f"{storage_public_url}/cinematic.aep"

    # Step 1: Swap Desktop placeholders with real assets
    console.print("    [dim]Preparing assets for AE render...[/dim]")

    # Resolve voiceover: combine per-slide paths into single, or use text
    single_vo_path = None
    if isinstance(voiceover_path, list):
        # Find first existing per-slide voiceover
        for vp in voiceover_path:
            if vp and Path(vp).exists():
                single_vo_path = Path(vp)
                break
    elif voiceover_path and Path(voiceover_path).exists():
        single_vo_path = Path(voiceover_path)

    _swap_desktop_assets(
        slide_image_paths=slide_image_paths,
        voiceover_text=getattr(script, "voiceover_text", None) if not single_vo_path else None,
        voiceover_path=single_vo_path,
    )

    # Step 2: Build text-only nexrender job
    # (images and audio are loaded from Desktop paths by the .aep template)
    TITLE_FS, HOOK_FS, BODY_FS, CLOSING_FS, CTA_FS = 72, 42, 42, 54, 48

    closing_text = getattr(script, "closing_slide", "") or ""
    cta_text = getattr(script, "cta", "") or "Follow for more stories like this at The Docket."

    assets = [
        {"type": "data", "layerName": "title_text", "property": "Source Text",
         "value": _wrap_text(script.title.upper(), TITLE_FS)},
        {"type": "data", "layerName": "hook_text", "property": "Source Text",
         "value": _wrap_text(script.hook or "", HOOK_FS)},
        {"type": "data", "layerName": "closing_text", "property": "Source Text",
         "value": _wrap_text(closing_text, CLOSING_FS)},
    ]

    for i, slide_text in enumerate(script.body_slides):
        assets.append({
            "type": "data",
            "layerName": f"body_text_{i + 1}",
            "property": "Source Text",
            "value": _wrap_text(slide_text, BODY_FS),
        })

    job_payload = {
        "template": {"src": template_url, "composition": composition},
        "assets": assets,
        "actions": {
            "postrender": [
                {"module": "@nexrender/action-encode", "preset": "mp4",
                 "output": "render_output.mp4"},
                {"module": "@nexrender/action-copy",
                 "output": str(output_path.resolve())},
            ]
        },
    }

    # Step 3: Submit and wait
    console.print("    [cyan]Submitting AE render to nexrender...[/cyan]")
    client = NexrenderClient()
    try:
        render_job = client.create_job(job_payload)
        job_uid = render_job["uid"]
        console.print(f"    [dim]Render job: {job_uid}[/dim]")

        render_result = client.wait_for_job(
            job_uid,
            poll_interval=poll_interval,
            timeout=render_timeout,
        )
        console.print(
            f"    [green]AE render complete "
            f"(state: {render_result.get('state', '?')})[/green]"
        )

        if not output_path.exists():
            raise RuntimeError(
                f"Render finished but output not found at {output_path}"
            )

        console.print(
            f"    [green]Saved: {output_path} "
            f"({output_path.stat().st_size // 1024} KB)[/green]"
        )

    finally:
        client.close()
        # Restore clean placeholders so stale images don't leak into next render
        _restore_desktop_placeholders()

    return output_path
