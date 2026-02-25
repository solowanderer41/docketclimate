"""Video generation for Docket Social using moviepy and Pillow.

Creates vertical (9:16) short-form videos from structured content scripts
with optional voiceover audio overlay.

Visual features (all animated per-frame):
    - Breathing gradient backgrounds with per-slide hue variation
    - Floating particle field (bokeh-like glowing circles)
    - Text fade-in at slide start
    - Section-based accent colors (6 sections + default)
    - Text shadows for depth and readability
    - "THE DOCKET" watermark (bottom-right, subtle)
    - Animated progress bar showing slide position
    - Top accent bar for branded framing
"""

import colorsys
import math
import random
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

console = Console()

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class VideoScript:
    """Structured content for a single video."""

    title: str
    hook: str
    body_slides: list[str] = field(default_factory=list)
    cta: str = ""
    voiceover_text: str = ""
    section: str = ""  # section_id for color theming (e.g. "lived", "science")
    video_tier: str = "narrative"          # "cinematic" or "narrative"
    image_prompts: list[str] = field(default_factory=list)
    background_prompt: str = ""


# ---------------------------------------------------------------------------
# Config + color helpers
# ---------------------------------------------------------------------------


def _load_video_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the video section from config.yaml."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("video", {})


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string like '#1a1a2e' to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


_FONT_CACHE_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "extracted"
_OSWALD_ZIP = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "oswald.zip"
_oswald_bold_path: Path | None = None  # cached after first extraction


def _ensure_oswald_extracted() -> Path | None:
    """Extract Oswald font from bundled zip if not already extracted.

    Returns path to Oswald-Bold.ttf, or None if unavailable.
    Result is cached after first successful call.
    """
    global _oswald_bold_path
    if _oswald_bold_path is not None:
        return _oswald_bold_path

    # Check if already extracted
    bold_path = _FONT_CACHE_DIR / "Oswald-Bold.ttf"
    if bold_path.exists():
        _oswald_bold_path = bold_path
        return bold_path

    if not _OSWALD_ZIP.exists():
        return None

    try:
        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_OSWALD_ZIP, "r") as zf:
            zf.extractall(_FONT_CACHE_DIR)
        if bold_path.exists():
            _oswald_bold_path = bold_path
            return bold_path
        # Zip may have a different structure — find any Bold .ttf
        for f in _FONT_CACHE_DIR.rglob("*Bold*.ttf"):
            _oswald_bold_path = f
            return f
        # Last resort: any .ttf in the extracted dir
        for f in _FONT_CACHE_DIR.rglob("*.ttf"):
            _oswald_bold_path = f
            return f
    except Exception:
        pass
    return None


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a font, preferring bundled Oswald then system fonts."""
    # Try bundled Oswald first (works on any platform)
    oswald_path = _ensure_oswald_extracted()
    if oswald_path:
        try:
            return ImageFont.truetype(str(oswald_path), size)
        except (OSError, IOError):
            pass

    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def _get_title_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the Docket brand title font — bold, condensed, striking.

    Primary: Futura Condensed ExtraBold (index 4 in Futura.ttc) — a
    heavy condensed geometric sans-serif that screams headline.
    Secondary: DIN Condensed Bold — another high-impact condensed font.
    Both read as bold news headlines unique to The Docket brand.
    """
    # Futura.ttc indices: 0=Medium, 1=MediumItalic, 2=Bold,
    #                     3=CondensedMedium, 4=CondensedExtraBold
    _FUTURA_TTC = "/System/Library/Fonts/Supplemental/Futura.ttc"
    if Path(_FUTURA_TTC).exists():
        try:
            return ImageFont.truetype(_FUTURA_TTC, size, index=4)
        except (OSError, IOError):
            pass

    # Bundled Oswald-Bold — condensed and bold, good for headlines
    oswald_path = _ensure_oswald_extracted()
    if oswald_path:
        try:
            return ImageFont.truetype(str(oswald_path), size)
        except (OSError, IOError):
            pass

    title_font_candidates = [
        "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for font_path in title_font_candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue
    # Last resort: use the regular font
    return _get_font(size)


def _shift_hue(rgb: tuple[int, int, int], degrees: float) -> tuple[int, int, int]:
    """Shift the hue of an RGB color by a number of degrees (0-360)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def _crop_frame_to_portrait(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale and center-crop a video frame to exact target dimensions.

    Handles landscape or mismatched stock footage by scaling to cover
    the target area, then center-cropping the excess.
    """
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    frame_aspect = w / h

    if frame_aspect > target_aspect:
        # Frame is wider — scale by height, crop width
        scale = target_h / h
    else:
        # Frame is taller — scale by width, crop height
        scale = target_w / w

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize via PIL (faster than scipy for single frames)
    img = Image.fromarray(frame)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(img)

    # Center-crop to exact target
    y_off = (new_h - target_h) // 2
    x_off = (new_w - target_w) // 2
    return resized[y_off : y_off + target_h, x_off : x_off + target_w]


def _blend_color(
    base: tuple[int, int, int],
    target: tuple[int, int, int],
    strength: float,
) -> tuple[int, int, int]:
    """Blend base color toward target by strength (0.0 = base, 1.0 = target)."""
    return tuple(
        int(base[i] + (target[i] - base[i]) * strength) for i in range(3)
    )


# ---------------------------------------------------------------------------
# Particle system
# ---------------------------------------------------------------------------


@dataclass
class Particle:
    """A single floating bokeh particle."""
    x: float          # 0-1 normalized x position
    y: float          # 0-1 normalized y position
    radius: float     # pixel radius
    speed: float      # upward drift per second (fraction of height)
    opacity: float    # 0-1
    phase: float      # random phase for pulsing


def _generate_particles(count: int, config: dict, seed: int = 0) -> list[Particle]:
    """Generate a deterministic set of particles for a slide.

    Args:
        count: Number of particles.
        config: Particle config dict with min_radius, max_radius, speed, opacity.
        seed: Random seed for reproducibility per-slide.
    """
    rng = random.Random(seed)
    min_r = config.get("min_radius", 2)
    max_r = config.get("max_radius", 8)
    base_speed = config.get("speed", 0.3)
    max_opacity = config.get("opacity", 0.12)

    particles = []
    for _ in range(count):
        r = rng.uniform(min_r, max_r)
        # Larger particles are slower and more transparent (depth effect)
        size_factor = (r - min_r) / max(max_r - min_r, 1)
        particles.append(Particle(
            x=rng.random(),
            y=rng.random(),
            radius=r,
            speed=base_speed * (0.5 + 0.5 * (1 - size_factor)),
            opacity=max_opacity * (0.4 + 0.6 * (1 - size_factor)),
            phase=rng.uniform(0, 2 * math.pi),
        ))
    return particles


# ---------------------------------------------------------------------------
# Animated frame renderer (called every frame at ~30fps)
# ---------------------------------------------------------------------------


def _render_animated_frame(
    t: float,
    *,
    text: str,
    width: int,
    height: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    font_size: int,
    accent_color_hex: str | None,
    section_accent_rgb: tuple[int, int, int],
    slide_index: int,
    total_slides: int,
    slide_duration: float,
    visual_config: dict,
    particles: list[Particle],
    font: ImageFont.FreeTypeFont,
    # Pre-computed text layout (avoids recalculating every frame)
    wrapped_lines: list[str],
    line_positions: list[tuple[int, int]],  # (x, y) for each line
    total_text_height: int,
    y_start: int,
    # Stock B-roll background (None = use gradient fallback)
    bg_clip=None,
    overlay_opacity: float = 0.55,
    # Kinetic text config (None = use single fade-in fallback)
    kinetic_config: dict | None = None,
    # AI image backgrounds (Tier 1 / Tier 2)
    slide_bg_image: "Image.Image | None" = None,      # Tier 1: per-slide AI image
    single_bg_image: "Image.Image | None" = None,      # Tier 2: single AI background
    cinematic_config: dict | None = None,
    narrative_overlay: float = 0.55,
    # Subtitle layout for cinematic mode
    subtitle_lines: list[str] | None = None,
    subtitle_positions: list[tuple[int, int]] | None = None,
    subtitle_font: "ImageFont.FreeTypeFont | None" = None,
    # Slide type: "title" = big bold title over image (light vignette),
    #             "body" = cinematic subtitles, "overlay" = dark overlay + centered text
    slide_type: str = "body",
    # Title-specific layout (only used when slide_type == "title")
    title_lines: list[str] | None = None,
    title_positions: list[tuple[int, int]] | None = None,
    title_font: "ImageFont.FreeTypeFont | None" = None,
) -> np.ndarray:
    """Render a single animated frame for time t within a slide.

    This is the hot path — called 30x per second per slide.
    Performance matters here, so we minimize allocations.
    """
    visual = visual_config
    gradient_config = visual.get("gradient", {})
    breathing_config = visual.get("breathing", {})
    particle_config = visual.get("particles", {})
    shadow_config = visual.get("text_shadow", {})
    text_fade_config = visual.get("text_fade", {})
    progress_config = visual.get("progress_bar", {})
    top_accent_config = visual.get("top_accent", {})
    watermark_config = visual.get("watermark", {})

    # --- Background priority: AI cinematic > AI narrative > stock B-roll > gradient ---
    _using_ai_bg = False
    _cinematic_mode = False
    _title_mode = False

    if slide_bg_image is not None:
        # AI image with Ken Burns pan/zoom
        _using_ai_bg = True
        # Cinematic subtitle mode for body slides; title slides get their own
        # rendering; overlay slides get dark overlay + centered text.
        _cinematic_mode = (slide_type == "body")
        _title_mode = (slide_type == "title")
        cine = cinematic_config or {}
        kb_zoom = cine.get("ken_burns_zoom", 1.12)
        kb_pan = cine.get("ken_burns_pan_px", 50)

        # Ease-in-out curve: smooth acceleration/deceleration
        progress = t / max(slide_duration, 0.01)
        progress = max(0.0, min(1.0, progress))
        eased = progress * progress * (3.0 - 2.0 * progress)

        # Alternating direction per slide for visual variety:
        #   Even slides: zoom in + pan right/down
        #   Odd slides:  zoom out + pan left/up
        if slide_index % 2 == 0:
            # Zoom in (1.0 → kb_zoom), pan right + down
            zoom = 1.0 + (kb_zoom - 1.0) * eased
            pan_x = int(kb_pan * eased)
            pan_y = int(kb_pan * 0.5 * eased)
        else:
            # Zoom out (kb_zoom → 1.0), pan left + up
            zoom = kb_zoom - (kb_zoom - 1.0) * eased
            pan_x = int(-kb_pan * eased)
            pan_y = int(-kb_pan * 0.5 * eased) if (slide_index % 4 == 1) else int(kb_pan * 0.3 * eased)

        # Scale up image for zoom headroom (always use max zoom to ensure enough pixels)
        max_zoom = max(kb_zoom, zoom) + 0.02  # small safety margin
        src_w = int(width * max_zoom)
        src_h = int(height * max_zoom)
        zoomed = slide_bg_image.resize((src_w, src_h), Image.LANCZOS)

        # Center-crop with pan offset
        cx = (src_w - width) // 2 + pan_x
        cy = (src_h - height) // 2 + pan_y
        cx = max(0, min(cx, src_w - width))
        cy = max(0, min(cy, src_h - height))
        img = zoomed.crop((cx, cy, cx + width, cy + height))

        # Apply dark overlay for overlay slides (non-body) for text readability
        if slide_type == "overlay":
            overlay_alpha = narrative_overlay  # ~0.55
            img_arr = np.array(img)
            dark = np.full_like(img_arr, 0, dtype=np.uint8)
            blended = (img_arr.astype(np.float32) * (1 - overlay_alpha)
                       + dark.astype(np.float32) * overlay_alpha)
            img = Image.fromarray(blended.astype(np.uint8))
        elif slide_type == "title":
            # Title slide: lighter vignette (35% center, heavier at edges)
            # so the image shows through but bold title text still pops
            img_arr = np.array(img)
            h, w = img_arr.shape[:2]
            # Create radial vignette mask: lighter in center, darker at edges
            y_grid, x_grid = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            dist = np.sqrt(((x_grid - cx) / cx) ** 2 + ((y_grid - cy) / cy) ** 2)
            # Vignette: 0.25 at center → 0.65 at edges
            vignette = 0.25 + 0.40 * np.clip(dist, 0, 1)
            vignette = vignette[:, :, np.newaxis]
            blended = img_arr.astype(np.float32) * (1 - vignette)
            img = Image.fromarray(blended.astype(np.uint8))

    elif single_bg_image is not None:
        # Tier 2: Narrative — single AI background with dark overlay
        _using_ai_bg = True
        bg_arr = np.array(single_bg_image)
        overlay_arr = np.full_like(bg_arr, 0, dtype=np.uint8)
        alpha = narrative_overlay
        blended = (bg_arr.astype(np.float32) * (1 - alpha) + overlay_arr.astype(np.float32) * alpha)
        img = Image.fromarray(blended.astype(np.uint8))

    elif bg_clip is not None:
        # Use stock footage frame as background (looping)
        loop_t = t % bg_clip.duration
        bg_frame = bg_clip.get_frame(loop_t)
        bg_frame = _crop_frame_to_portrait(bg_frame, width, height)

        # Dark overlay for text readability
        overlay_arr = np.full_like(bg_frame, 0, dtype=np.uint8)
        alpha = overlay_opacity
        blended = (bg_frame.astype(np.float32) * (1 - alpha) + overlay_arr.astype(np.float32) * alpha)
        img = Image.fromarray(blended.astype(np.uint8))
    else:
        # Fallback: breathing gradient background
        breath_offset = 0.0
        if breathing_config.get("enabled", False):
            speed = breathing_config.get("speed", 0.4)
            amplitude = breathing_config.get("amplitude", 0.08)
            breath_offset = math.sin(t * speed * 2 * math.pi) * amplitude

        if gradient_config.get("enabled", False):
            strength = gradient_config.get("strength", 0.35) + breath_offset
            strength = max(0.05, min(0.8, strength))  # clamp
            angle_deg = gradient_config.get("angle", 160)

            lighter = tuple(min(255, int(c + (255 - c) * strength * 0.5)) for c in bg_color)
            end_color = _blend_color(lighter, section_accent_rgb, 0.25)

            hue_offset = (slide_index % 5 - 2) * 4.0
            end_color = _shift_hue(end_color, hue_offset)

            angle_rad = math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)

            y_coords, x_coords = np.mgrid[0:height, 0:width]
            x_norm = x_coords / max(width - 1, 1)
            y_norm = y_coords / max(height - 1, 1)
            projection = x_norm * dx + y_norm * dy
            p_min, p_max = projection.min(), projection.max()
            grad_t = (projection - p_min) / max(p_max - p_min, 1e-6)

            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                pixels[:, :, c] = (bg_color[c] * (1 - grad_t) + end_color[c] * grad_t).astype(np.uint8)

            img = Image.fromarray(pixels)
        else:
            img = Image.new("RGB", (width, height), bg_color)

    # --- Particles (only over gradient, not stock footage or AI images) ---
    if particle_config.get("enabled", False) and particles and bg_clip is None and not _using_ai_bg:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(overlay)

        for p in particles:
            # Drift upward, wrap around
            py = (p.y - p.speed * t) % 1.0
            px = p.x + math.sin(t * 0.5 + p.phase) * 0.01  # gentle horizontal sway
            px = px % 1.0

            # Pulsing opacity
            pulse = 0.7 + 0.3 * math.sin(t * 1.5 + p.phase)
            alpha = int(255 * p.opacity * pulse)

            cx = int(px * width)
            cy = int(py * height)
            r = int(p.radius)

            # Draw soft circle (accent-tinted)
            pr, pg, pb = section_accent_rgb
            pdraw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(pr, pg, pb, alpha),
            )

        img_rgba = img.convert("RGBA")
        img = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    draw = ImageDraw.Draw(img)

    # --- Top accent bar (skip in cinematic mode) ---
    if top_accent_config.get("enabled", False) and not _cinematic_mode:
        bar_h = top_accent_config.get("height", 3)
        draw.rectangle([0, 0, width, bar_h], fill=section_accent_rgb)

    # --- Text rendering ---
    fade_alpha = 1.0  # used by accent underline below

    if _title_mode and title_lines and title_positions and title_font:
        # Title slide rendering: big bold DIN Condensed text, centered,
        # with thick stroke and instant visibility (no fade-in).
        # Text is rendered ALL CAPS for maximum impact.
        txt_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_overlay)

        alpha_int = 255  # instant, no fade

        for i, line in enumerate(title_lines):
            x, y = title_positions[i]
            # Heavy black stroke for readability over image
            txt_draw.text(
                (x, y), line, fill=(0, 0, 0, alpha_int),
                font=title_font,
                stroke_width=5, stroke_fill=(0, 0, 0, alpha_int),
            )
            # White text on top
            txt_draw.text(
                (x, y), line, fill=(255, 255, 255, alpha_int),
                font=title_font,
            )

        img_rgba = img.convert("RGBA")
        img = Image.alpha_composite(img_rgba, txt_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

    elif _cinematic_mode and subtitle_lines and subtitle_positions and subtitle_font:
        # Cinematic subtitle rendering: small white text at bottom with stroke
        cine = cinematic_config or {}
        stroke_w = cine.get("subtitle_stroke_width", 2)

        txt_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_overlay)

        # Fade in subtitles — instant on the opener (slide_index 0)
        # so text is immediately visible for the scroll decision
        if slide_index == 0:
            sub_alpha = 1.0
        else:
            fade_dur = 0.3
            if t < fade_dur:
                sub_alpha = t / fade_dur
                sub_alpha = 1 - (1 - sub_alpha) ** 2
            else:
                sub_alpha = 1.0

        alpha_int = int(255 * sub_alpha)

        # --- Subtitle background pill for readability ---
        pill_cfg = cine.get("subtitle_pill", {})
        if pill_cfg.get("enabled", True) and subtitle_positions:
            pill_color_hex = pill_cfg.get("color", "#000000")
            pill_opacity = pill_cfg.get("opacity", 0.65)
            pill_px = pill_cfg.get("padding_x", 24)
            pill_py = pill_cfg.get("padding_y", 12)
            pill_radius = pill_cfg.get("border_radius", 12)
            pr, pg, pb = _hex_to_rgb(pill_color_hex)
            pill_alpha = int(255 * pill_opacity * sub_alpha)

            # Compute bounding box around all subtitle lines
            _sub_font_size = subtitle_font.size if hasattr(subtitle_font, 'size') else 44
            line_h = int(_sub_font_size * 1.5)
            min_x = min(pos[0] for pos in subtitle_positions) - pill_px
            min_y = subtitle_positions[0][1] - pill_py
            max_y = subtitle_positions[-1][1] + line_h + pill_py
            # Measure widest line for max_x
            max_x = 0
            for i, line in enumerate(subtitle_lines):
                bbox = txt_draw.textbbox((0, 0), line, font=subtitle_font)
                line_w = bbox[2] - bbox[0]
                max_x = max(max_x, subtitle_positions[i][0] + line_w)
            max_x += pill_px

            txt_draw.rounded_rectangle(
                [min_x, min_y, max_x, max_y],
                radius=pill_radius,
                fill=(pr, pg, pb, pill_alpha),
            )

        for i, line in enumerate(subtitle_lines):
            x, y = subtitle_positions[i]
            # Black stroke outline for readability over any background
            txt_draw.text(
                (x, y), line, fill=(0, 0, 0, alpha_int),
                font=subtitle_font,
                stroke_width=stroke_w, stroke_fill=(0, 0, 0, alpha_int),
            )
            # White text on top
            txt_draw.text(
                (x, y), line, fill=(255, 255, 255, alpha_int),
                font=subtitle_font,
            )

        img_rgba = img.convert("RGBA")
        img = Image.alpha_composite(img_rgba, txt_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

    elif (kinetic_config is not None
        and kinetic_config.get("enabled", False)
        and bg_clip is not None
    ):
        # Kinetic text: each line appears with a stagger delay + ease-out fade
        stagger_delay = kinetic_config.get("stagger_delay", 0.25)
        line_fade_dur = kinetic_config.get("line_fade_duration", 0.3)

        txt_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_overlay)
        tr, tg, tb = text_color

        for i, line in enumerate(wrapped_lines):
            line_start = i * stagger_delay
            if t < line_start:
                continue  # line hasn't appeared yet

            elapsed = t - line_start
            if elapsed < line_fade_dur:
                line_alpha = elapsed / line_fade_dur
                line_alpha = 1 - (1 - line_alpha) ** 2  # ease-out
            else:
                line_alpha = 1.0

            alpha_int = int(255 * line_alpha)
            x, y = line_positions[i]

            # Shadow
            if shadow_config.get("enabled", False):
                offset = shadow_config.get("offset", [3, 3])
                shadow_alpha = int(alpha_int * 0.5)
                txt_draw.text(
                    (x + offset[0], y + offset[1]),
                    line, fill=(0, 0, 0, shadow_alpha), font=font,
                )

            txt_draw.text((x, y), line, fill=(tr, tg, tb, alpha_int), font=font)

        img_rgba = img.convert("RGBA")
        img = Image.alpha_composite(img_rgba, txt_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Set fade_alpha for accent underline (based on first line)
        if line_fade_dur > 0 and t < line_fade_dur:
            fade_alpha = 1 - (1 - t / line_fade_dur) ** 2
        else:
            fade_alpha = 1.0
    else:
        # Original single fade-in — skip for Hook Frame (slide_index 0)
        # so text is instantly visible for the 1.7-second scroll decision
        fade_alpha = 1.0
        if text_fade_config.get("enabled", False) and slide_index != 0:
            fade_dur = text_fade_config.get("duration", 0.4)
            if t < fade_dur:
                fade_alpha = t / fade_dur
                fade_alpha = 1 - (1 - fade_alpha) ** 2

        if fade_alpha < 1.0:
            txt_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_overlay)
            alpha_int = int(255 * fade_alpha)

            if shadow_config.get("enabled", False):
                offset = shadow_config.get("offset", [3, 3])
                shadow_alpha = int(alpha_int * 0.5)
                for i, line in enumerate(wrapped_lines):
                    x, y = line_positions[i]
                    txt_draw.text(
                        (x + offset[0], y + offset[1]),
                        line, fill=(0, 0, 0, shadow_alpha), font=font,
                    )

            tr, tg, tb = text_color
            for i, line in enumerate(wrapped_lines):
                x, y = line_positions[i]
                txt_draw.text((x, y), line, fill=(tr, tg, tb, alpha_int), font=font)

            img_rgba = img.convert("RGBA")
            img = Image.alpha_composite(img_rgba, txt_overlay).convert("RGB")
            draw = ImageDraw.Draw(img)
        else:
            for i, line in enumerate(wrapped_lines):
                x, y = line_positions[i]
                if shadow_config.get("enabled", False):
                    offset = shadow_config.get("offset", [3, 3])
                    shadow_hex = shadow_config.get("color", "#000000")
                    shadow_rgb = _hex_to_rgb(shadow_hex[:7])
                    draw.text((x + offset[0], y + offset[1]), line, fill=shadow_rgb, font=font)
                draw.text((x, y), line, fill=text_color, font=font)

    # --- Accent underline (on title/CTA slides, skip in cinematic mode) ---
    if accent_color_hex and not _cinematic_mode:
        padding_x = int(width * 0.1)
        max_text_width = width - 2 * padding_x
        underline_y = y_start + total_text_height + int(font_size * 0.4)
        underline_width = min(max_text_width, int(width * 0.4))
        underline_x = (width - underline_width) // 2
        underline_thickness = max(3, font_size // 12)

        if fade_alpha < 1.0:
            # Faded underline
            ul_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ul_draw = ImageDraw.Draw(ul_overlay)
            ar, ag, ab = section_accent_rgb
            ul_draw.rectangle(
                [underline_x, underline_y,
                 underline_x + underline_width, underline_y + underline_thickness],
                fill=(ar, ag, ab, int(255 * fade_alpha)),
            )
            img_rgba = img.convert("RGBA")
            img = Image.alpha_composite(img_rgba, ul_overlay).convert("RGB")
            draw = ImageDraw.Draw(img)
        else:
            draw.rectangle(
                [underline_x, underline_y,
                 underline_x + underline_width, underline_y + underline_thickness],
                fill=section_accent_rgb,
            )

    # --- Animated progress bar ---
    if progress_config.get("enabled", False):
        bar_height = progress_config.get("height", 4)
        bar_color_hex = progress_config.get("color")
        bar_color = _hex_to_rgb(bar_color_hex) if bar_color_hex else section_accent_rgb

        # Smooth progress: interpolate within the current slide
        base_progress = slide_index / max(total_slides, 1)
        slide_progress = (slide_index + t / slide_duration) / max(total_slides, 1)
        bar_width = int(width * slide_progress)

        y_top = height - bar_height
        draw.rectangle([0, y_top, bar_width, height], fill=bar_color)

    # --- Watermark ---
    if watermark_config.get("enabled", False):
        wm_text = watermark_config.get("text", "THE DOCKET")
        wm_font_size = watermark_config.get("font_size", 24)
        wm_opacity = watermark_config.get("opacity", 0.15)
        wm_margin = watermark_config.get("margin", 40)

        wm_font = _get_font(wm_font_size)
        wm_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        wm_draw = ImageDraw.Draw(wm_overlay)

        bbox = wm_draw.textbbox((0, 0), wm_text, font=wm_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        wx = img.width - tw - wm_margin
        wy = img.height - th - wm_margin
        wm_alpha = int(255 * wm_opacity)
        wm_draw.text((wx, wy), wm_text, fill=(255, 255, 255, wm_alpha), font=wm_font)

        img_rgba = img.convert("RGBA")
        img = Image.alpha_composite(img_rgba, wm_overlay).convert("RGB")

    return np.array(img)


# ---------------------------------------------------------------------------
# Text layout pre-computation
# ---------------------------------------------------------------------------


def _compute_text_layout(
    text: str,
    width: int,
    height: int,
    font_size: int,
    font: ImageFont.FreeTypeFont,
) -> tuple[list[str], list[tuple[int, int]], int, int]:
    """Pre-compute text wrapping and positions (done once per slide, not per frame).

    Respects Instagram safe zones: top 250px, bottom 320px, right 120px.
    Text is centered within the safe area.

    Returns:
        (wrapped_lines, line_positions, total_text_height, y_start)
    """
    # Instagram safe zones (pixels)
    SAFE_TOP = 250
    SAFE_BOTTOM = 320
    SAFE_RIGHT = 120

    padding_left = int(width * 0.1)
    padding_right = max(int(width * 0.1), SAFE_RIGHT)
    max_text_width = width - padding_left - padding_right

    avg_char_width = font_size * 0.55
    chars_per_line = max(1, int(max_text_width / avg_char_width))
    wrapped_lines = textwrap.wrap(text, width=chars_per_line)

    if not wrapped_lines:
        wrapped_lines = [""]

    line_spacing = int(font_size * 1.4)
    total_text_height = len(wrapped_lines) * line_spacing

    # Center vertically within safe area (between SAFE_TOP and height - SAFE_BOTTOM)
    available_height = height - SAFE_TOP - SAFE_BOTTOM
    y_start = SAFE_TOP + (available_height - total_text_height) // 2

    # Use a temp image to measure text widths
    tmp_img = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)

    # Center horizontally within safe area (shifted slightly left for right icons)
    safe_center_x = (width - padding_right + padding_left) // 2

    line_positions = []
    for i, line in enumerate(wrapped_lines):
        bbox = tmp_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = safe_center_x - line_width // 2
        y = y_start + i * line_spacing
        line_positions.append((x, y))

    return wrapped_lines, line_positions, total_text_height, y_start


def _compute_title_layout(
    text: str,
    width: int,
    height: int,
    font_size: int,
    font: ImageFont.FreeTypeFont,
) -> tuple[list[str], list[tuple[int, int]], int, int]:
    """Pre-compute title slide text layout — big, bold, ALL CAPS, centered.

    Positioned in the vertical center of the frame with generous padding.
    Text is converted to uppercase for maximum visual impact.
    Safe zones: top 250px, bottom 320px, right 120px.
    """
    SAFE_TOP = 250
    SAFE_BOTTOM = 320
    SAFE_RIGHT = 120

    # Wider padding for title text (more breathing room)
    padding_left = int(width * 0.08)
    padding_right = max(int(width * 0.08), SAFE_RIGHT)
    max_text_width = width - padding_left - padding_right

    # Convert to ALL CAPS
    text_upper = text.upper()

    # Condensed fonts are narrower, so use a tighter avg_char_width
    avg_char_width = font_size * 0.42  # condensed fonts are ~40% narrower
    chars_per_line = max(1, int(max_text_width / avg_char_width))
    wrapped_lines = textwrap.wrap(text_upper, width=chars_per_line)

    if not wrapped_lines:
        wrapped_lines = [""]

    line_spacing = int(font_size * 1.3)
    total_text_height = len(wrapped_lines) * line_spacing

    # Center vertically in safe area
    available_height = height - SAFE_TOP - SAFE_BOTTOM
    y_start = SAFE_TOP + (available_height - total_text_height) // 2

    tmp_img = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)

    safe_center_x = (width - padding_right + padding_left) // 2

    line_positions = []
    for i, line in enumerate(wrapped_lines):
        bbox = tmp_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = safe_center_x - line_width // 2
        y = y_start + i * line_spacing
        line_positions.append((x, y))

    return wrapped_lines, line_positions, total_text_height, y_start


def _compute_subtitle_layout(
    text: str,
    width: int,
    height: int,
    font_size: int,
    font: ImageFont.FreeTypeFont,
    y_fraction: float = 0.55,
) -> tuple[list[str], list[tuple[int, int]], int, int]:
    """Pre-compute subtitle text layout (center-third of frame, centered).

    Used for Tier 1 cinematic mode — bold text positioned in the center-third
    of the frame for maximum engagement. Safe zones avoid Instagram UI overlays:
    top 250px (username/follow), bottom 320px (like/comment/share), right 120px.
    """
    # Instagram safe zones (pixels)
    SAFE_TOP = 250
    SAFE_BOTTOM = 320
    SAFE_RIGHT = 120

    # Asymmetric horizontal padding: normal left, extra right for IG icons
    padding_left = int(width * 0.08)
    padding_right = max(int(width * 0.08), SAFE_RIGHT)
    max_text_width = width - padding_left - padding_right

    avg_char_width = font_size * 0.55
    chars_per_line = max(1, int(max_text_width / avg_char_width))
    wrapped_lines = textwrap.wrap(text, width=chars_per_line)

    if not wrapped_lines:
        wrapped_lines = [""]

    line_spacing = int(font_size * 1.5)
    total_text_height = len(wrapped_lines) * line_spacing

    # Position subtitles at y_fraction of the frame height (center-third)
    y_start = int(height * y_fraction) - total_text_height // 2

    # Clamp to safe zones
    if y_start < SAFE_TOP:
        y_start = SAFE_TOP
    if y_start + total_text_height > height - SAFE_BOTTOM:
        y_start = height - SAFE_BOTTOM - total_text_height

    tmp_img = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)

    # Center text horizontally within the safe area (shifted slightly left)
    safe_center_x = (width - padding_right + padding_left) // 2

    line_positions = []
    for i, line in enumerate(wrapped_lines):
        bbox = tmp_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = safe_center_x - line_width // 2
        y = y_start + i * line_spacing
        line_positions.append((x, y))

    return wrapped_lines, line_positions, total_text_height, y_start


# ---------------------------------------------------------------------------
# Thumbnail / cover image generation
# ---------------------------------------------------------------------------


def generate_thumbnail(
    script: VideoScript,
    output_path: Path,
    config: dict | None = None,
    ai_image: "Image.Image | None" = None,
) -> Path:
    """Generate a branded thumbnail/cover image for Instagram Reels.

    Design principles:
    - Critical content in center 1080×1080 (profile grid crops to 1:1)
    - Hook text: large, bold, max 8 words
    - Section accent color bar at top
    - "THE DOCKET" watermark
    - Darkened background for text contrast

    Args:
        script: VideoScript with title, hook, body_slides, section.
        output_path: Where to save the PNG thumbnail.
        config: Full video config dict from config.yaml.
        ai_image: Optional PIL Image to use as background.

    Returns:
        Path to the saved thumbnail image.
    """
    config = config or {}
    visual_config = config.get("visual", {})
    section_colors = visual_config.get("section_colors", {})

    width, height = 1080, 1920
    accent_hex = section_colors.get(script.section, section_colors.get("default", "#4ecdc4"))
    accent_rgb = _hex_to_rgb(accent_hex)

    # Background: AI image with vignette (matches title slide treatment)
    if ai_image is not None:
        img = ai_image.copy().resize((width, height), Image.LANCZOS)
        # Radial vignette: lighter center (25%), darker edges (65%)
        import numpy as np
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.sqrt(((x_grid - cx) / cx) ** 2 + ((y_grid - cy) / cy) ** 2)
        vignette = 0.25 + 0.40 * np.clip(dist, 0, 1)
        vignette = vignette[:, :, np.newaxis]
        blended = img_arr.astype(np.float32) * (1 - vignette)
        img = Image.fromarray(blended.astype(np.uint8))
    else:
        bg_hex = config.get("background_color", "#1a1a2e")
        bg_rgb = _hex_to_rgb(bg_hex)
        img = Image.new("RGB", (width, height), bg_rgb)

    draw = ImageDraw.Draw(img)

    # Section accent bar at top (4px)
    draw.rectangle([0, 0, width, 4], fill=accent_rgb)

    # Thumbnail = Title slide: big bold ALL CAPS title using Docket brand font
    # This matches what the viewer sees on slide 1 of the video.
    title_text = (script.title or "").upper()

    # Use _compute_title_layout to get consistent positioning
    thumb_title_font_size = 80
    thumb_title_font = _get_title_font(thumb_title_font_size)
    title_wrapped, title_positions, _, _ = _compute_title_layout(
        script.title or "", width, height, thumb_title_font_size, thumb_title_font,
    )

    # Draw title (white with heavy black stroke, same as video title slide)
    for i, line in enumerate(title_wrapped):
        x, y = title_positions[i]
        # Black stroke outline
        draw.text((x, y), line, fill=(0, 0, 0), font=thumb_title_font,
                  stroke_width=5, stroke_fill=(0, 0, 0))
        # White text
        draw.text((x, y), line, fill=(255, 255, 255), font=thumb_title_font)

    # No separate hook text on thumbnail — title only for clean visual

    # "THE DOCKET" watermark — bottom area, subtle
    wm_font = _get_font(28)
    wm_text = "THE DOCKET"
    wm_bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
    wm_w = wm_bbox[2] - wm_bbox[0]
    wm_x = (width - wm_w) // 2
    wm_y = height - 80

    wm_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(wm_overlay)
    wm_alpha = int(255 * 0.25)
    wm_draw.text((wm_x, wm_y), wm_text, fill=(255, 255, 255, wm_alpha), font=wm_font)
    img_rgba = img.convert("RGBA")
    img = Image.alpha_composite(img_rgba, wm_overlay).convert("RGB")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    console.print(f"[dim]Thumbnail generated: {output_path.name}[/dim]")
    return output_path


# ---------------------------------------------------------------------------
# Main video generation
# ---------------------------------------------------------------------------


def generate_video(
    script: VideoScript,
    voiceover_path: "list[Path | None] | Path | None" = None,
    output_path: Path = None,
    config: dict | None = None,
    stock_clip_path: Path | None = None,
    video_tier: str = "narrative",
    slide_image_paths: list[Path | None] | None = None,
    background_image_path: Path | None = None,
) -> Path:
    """Generate a vertical short-form video from a VideoScript.

    Each slide is an animated VideoClip with floating particles,
    breathing gradient, and text fade-in — rendered per-frame via
    make_frame(t). When a stock clip is provided, it replaces the
    gradient background with looping B-roll footage and enables
    kinetic (line-by-line) text animation.

    Two-tier AI image modes:
        - Tier 1 (cinematic): Per-slide AI images with subtitle captions
          and Ken Burns pan/zoom. slide_image_paths provides one image
          per body slide.
        - Tier 2 (narrative): Single AI background image with standard
          text overlay. background_image_path provides the image.

    Args:
        script: The structured content for the video.
        voiceover_path: Per-slide audio paths (list of Path|None, one per slide),
            a single Path to a combined MP3, or None for silent video.
        output_path: Where to write the final MP4.
        config: Video config dict. Loaded from config.yaml if not provided.
        stock_clip_path: Path to a stock video clip for background, or None.
        video_tier: "cinematic" or "narrative" (default).
        slide_image_paths: List of AI image paths (Tier 1), one per body slide.
        background_image_path: Single AI background image path (Tier 2).

    Returns:
        The path to the exported MP4 file.
    """
    if config is None:
        config = _load_video_config()

    resolution = tuple(config.get("resolution", [1080, 1920]))
    fps = config.get("fps", 30)
    bg_color_hex = config.get("background_color", "#1a1a2e")
    text_color_hex = config.get("text_color", "#e0e0e0")
    accent_color_hex = config.get("accent_color", "#4ecdc4")
    title_font_size = config.get("title_font_size", 64)
    body_font_size = config.get("font_size", 48)
    duration_per_slide = config.get("duration_per_slide", 5)
    transition_duration = config.get("transition_duration", 0.5)

    bg_color = _hex_to_rgb(bg_color_hex)
    text_color = _hex_to_rgb(text_color_hex)

    # Visual config
    visual_config = config.get("visual", {})
    section_colors = visual_config.get("section_colors", {})

    # Resolve section accent color
    section_accent_hex = section_colors.get(
        script.section, section_colors.get("default", accent_color_hex)
    )
    section_accent_rgb = _hex_to_rgb(section_accent_hex)

    width, height = resolution

    # --- Stock B-roll clip ---
    stock_config = visual_config.get("stock_footage", {})
    bg_clip = None
    if (
        stock_clip_path is not None
        and Path(stock_clip_path).exists()
        and stock_config.get("enabled", True)
    ):
        console.print(f"[cyan]Loading stock B-roll:[/] {stock_clip_path}")
        bg_clip = VideoFileClip(str(stock_clip_path), audio=False)
        console.print(
            f"[dim]Stock clip: {bg_clip.w}x{bg_clip.h}, "
            f"{bg_clip.duration:.1f}s[/dim]"
        )
    elif stock_clip_path is not None:
        console.print("[yellow]Stock clip not found or disabled, using gradient fallback.[/yellow]")

    overlay_opacity = stock_config.get("overlay_opacity", 0.55)

    # Kinetic text config (only active when stock footage is used)
    kinetic_config = visual_config.get("kinetic_text", {})

    # --- AI-generated images (Tier 1 / Tier 2) ---
    cinematic_config = config.get("cinematic", {})
    narrative_config = config.get("narrative", {})

    # Load Tier 1 per-slide images
    slide_images: list[Image.Image | None] = []
    if video_tier == "cinematic" and slide_image_paths:
        for img_path in slide_image_paths:
            if img_path and Path(img_path).exists():
                img = Image.open(img_path).convert("RGB").resize((width, height), Image.LANCZOS)
                slide_images.append(img)
            else:
                slide_images.append(None)
        has_slide_images = any(img is not None for img in slide_images)
        if not has_slide_images:
            video_tier = "narrative"  # downgrade if no images loaded
            slide_images = []
    else:
        has_slide_images = False

    # Load Tier 2 single background image
    single_bg_image: Image.Image | None = None
    if video_tier == "narrative" and background_image_path and Path(background_image_path).exists():
        single_bg_image = Image.open(background_image_path).convert("RGB").resize(
            (width, height), Image.LANCZOS
        )
        console.print(f"[cyan]AI background image loaded:[/] {background_image_path}")

    # Narrative overlay opacity (may differ from stock footage overlay)
    narrative_overlay = narrative_config.get("overlay_opacity", 0.55)

    section_label = f" [{script.section}]" if script.section else ""
    if video_tier == "cinematic" and has_slide_images:
        bg_label = " [AI cinematic]"
    elif single_bg_image is not None:
        bg_label = " [AI narrative]"
    elif bg_clip:
        bg_label = " [B-roll]"
    else:
        bg_label = " [gradient]"
    console.print(
        f"[bold cyan]Generating animated video[/] at {width}x{height} @ {fps}fps"
        f"{section_label}{bg_label}"
    )

    # -- Define slide structure with default durations --
    slides: list[dict] = []

    # Image assignment strategy (cinematic mode):
    #   Slide 0 (Title)  → slide_images[0]  (title font, vignette overlay)
    #   Slide 1 (Hook)   → slide_images[1]  (subtitle captions)
    #   Body 1..N         → slide_images[2..N+1] (subtitle captions)
    #   CTA               → last slide_image (dark overlay)
    # Every content slide gets its own unique image — no reuse.
    # The pipeline must generate enough images to cover all slides.

    _cinematic_closer_img = None  # last AI image for CTA slide
    if video_tier == "cinematic" and has_slide_images:
        for _img in reversed(slide_images):
            if _img is not None:
                _cinematic_closer_img = _img
                break

    # Image index cursor — advances through slide_images sequentially
    _img_cursor = 0

    def _next_image() -> "Image.Image | None":
        """Get the next available AI image, advancing the cursor. Returns None if exhausted."""
        nonlocal _img_cursor
        if not has_slide_images:
            return None
        while _img_cursor < len(slide_images):
            img = slide_images[_img_cursor]
            _img_cursor += 1
            if img is not None:
                return img
        return None  # no more images

    # --- Slide 1: Title ---
    # Bold Futura Condensed ExtraBold ALL CAPS on first AI image with
    # light vignette. This IS the thumbnail. VO reads just the title,
    # then a beat, then quick fade to hook.
    _title_img = _next_image()
    slides.append({
        "text": script.title,
        "duration": 3.0,  # default; overridden by voiceover duration
        "font_size": title_font_size,
        "accent": True,
        "label": "Title",
        "ai_image": _title_img,
        "slide_type": "title",
    })

    # --- Slide 2: Hook ---
    # Hook text as body-style subtitle over second AI image.
    # VO picks up after the beat following the title.
    if script.hook:
        _hook_img = _next_image()
        slides.append({
            "text": script.hook,
            "duration": 4.0,  # default; overridden by voiceover duration
            "font_size": body_font_size,
            "accent": False,
            "label": "Hook",
            "ai_image": _hook_img,
            "slide_type": "body",
        })

    # --- Body slides ---
    # Each body slide gets its own unique AI image from the cursor.
    for idx, body_text in enumerate(script.body_slides, 1):
        body_ai_image = _next_image()
        slides.append({
            "text": body_text,
            "duration": float(duration_per_slide),
            "font_size": body_font_size,
            "accent": False,
            "label": f"Body {idx}",
            "ai_image": body_ai_image,
            "slide_type": "body",
        })

    # --- CTA slide ---
    if script.cta:
        slides.append({
            "text": script.cta,
            "duration": 3.0,
            "font_size": body_font_size,
            "accent": True,
            "label": "CTA",
            "ai_image": _cinematic_closer_img,
            "slide_type": "overlay",
        })

    total_slides = len(slides)

    # -- Determine voiceover mode --
    # per_slide_audio: list of AudioFileClip|None (one per slide) — new mode
    # global_audio: single AudioFileClip — legacy mode
    per_slide_audio: list[AudioFileClip | None] | None = None
    global_audio: AudioFileClip | None = None
    has_any_audio = False

    voiceover_padding = config.get("voiceover_padding", 0.3)
    min_slide_duration = config.get("min_slide_duration", 1.5)

    if isinstance(voiceover_path, list):
        # --- Per-slide audio mode ---
        console.print(f"[cyan]Per-slide voiceover mode:[/] {len(voiceover_path)} segments")
        per_slide_audio = []
        for i, ap in enumerate(voiceover_path):
            if ap is not None and Path(ap).exists():
                aclip = AudioFileClip(str(ap))
                per_slide_audio.append(aclip)
                has_any_audio = True
            else:
                per_slide_audio.append(None)

        # Set each slide's duration from its audio
        # Title slide gets extra "beat" padding (0.6s) for the pause before
        # the hook starts — gives the viewer a moment to absorb the title.
        title_beat = 0.6
        for i, slide_info in enumerate(slides):
            if i < len(per_slide_audio) and per_slide_audio[i] is not None:
                audio_dur = per_slide_audio[i].duration
                padding = voiceover_padding
                if slide_info.get("slide_type") == "title":
                    padding += title_beat  # extra beat after title VO
                slide_info["duration"] = max(
                    audio_dur + padding,
                    min_slide_duration,
                )
                console.print(
                    f"  [dim]{slide_info['label']}: audio {audio_dur:.1f}s "
                    f"-> slide {slide_info['duration']:.1f}s[/dim]"
                )
            else:
                # No audio for this slide — keep default duration
                console.print(
                    f"  [dim]{slide_info['label']}: no audio, "
                    f"keeping {slide_info['duration']:.1f}s default[/dim]"
                )

    elif voiceover_path and Path(voiceover_path).exists():
        # --- Legacy single-file mode ---
        console.print(f"[cyan]Loading voiceover:[/] {voiceover_path}")
        global_audio = AudioFileClip(str(voiceover_path))
        has_any_audio = True
        audio_duration = global_audio.duration

        total_default = sum(s["duration"] for s in slides)
        if total_default > 0 and abs(total_default - audio_duration) > 0.5:
            scale = audio_duration / total_default
            console.print(
                f"[yellow]Adjusting slide timing:[/] {total_default:.1f}s -> "
                f"{audio_duration:.1f}s (scale {scale:.2f}x)"
            )
            for s in slides:
                s["duration"] = s["duration"] * scale

    # -- Build animated video clips --
    clips: list[VideoClip] = []
    audio_clips_to_close: list[AudioFileClip] = []
    particle_config = visual_config.get("particles", {})
    particle_count = particle_config.get("count", 35) if particle_config.get("enabled", False) else 0

    # Pre-load subtitle font for cinematic mode
    subtitle_font_size = cinematic_config.get("subtitle_font_size", 34)
    subtitle_y_frac = cinematic_config.get("subtitle_y_fraction", 0.85)
    _subtitle_font = _get_font(subtitle_font_size)

    # Pre-load title font (Futura Condensed ExtraBold)
    _title_font_size = cinematic_config.get("title_font_size", 80)
    _title_font = _get_title_font(_title_font_size)

    for slide_idx, slide_info in enumerate(slides):
        slide_duration = slide_info["duration"]
        console.print(
            f"  [dim]Building animated slide:[/] {slide_info['label']} "
            f"({slide_duration:.1f}s)"
        )

        # Pre-compute text layout once per slide
        slide_font = _get_font(slide_info["font_size"])
        wrapped_lines, line_positions, total_text_height, y_start = _compute_text_layout(
            slide_info["text"], width, height, slide_info["font_size"], slide_font,
        )

        # Pre-compute subtitle layout for cinematic mode
        _sub_lines, _sub_positions, _, _ = _compute_subtitle_layout(
            slide_info["text"], width, height, subtitle_font_size,
            _subtitle_font, y_fraction=subtitle_y_frac,
        )

        # Pre-compute title layout (only used for "title" slide_type)
        _ttl_lines, _ttl_positions, _, _ = _compute_title_layout(
            slide_info["text"], width, height, _title_font_size, _title_font,
        )

        # Generate deterministic particles for this slide
        slide_particles = _generate_particles(
            particle_count, particle_config, seed=slide_idx * 1000
        )

        # Resolve per-slide AI image (Tier 1 cinematic)
        _slide_ai_img = slide_info.get("ai_image")

        # Capture slide-specific variables in closure
        _text = slide_info["text"]
        _fs = slide_info["font_size"]
        _accent = section_accent_hex if slide_info["accent"] else None
        _idx = slide_idx
        _dur = slide_duration
        _font = slide_font
        _lines = wrapped_lines
        _positions = line_positions
        _tth = total_text_height
        _ys = y_start
        _particles = slide_particles
        _bg_clip = bg_clip
        _overlay_opacity = overlay_opacity
        _kinetic_config = kinetic_config
        _s_bg_img = _slide_ai_img
        _single_bg = single_bg_image
        _cine_cfg = cinematic_config
        _narr_overlay = narrative_overlay
        _sub_l = _sub_lines
        _sub_p = _sub_positions
        _sub_f = _subtitle_font
        _stype = slide_info.get("slide_type", "body")
        _ttl_l = _ttl_lines
        _ttl_p = _ttl_positions
        _ttl_f = _title_font

        def make_frame(t, _text=_text, _fs=_fs, _accent=_accent, _idx=_idx,
                       _dur=_dur, _font=_font, _lines=_lines, _positions=_positions,
                       _tth=_tth, _ys=_ys, _particles=_particles,
                       _bg_clip=_bg_clip, _overlay_opacity=_overlay_opacity,
                       _kinetic_config=_kinetic_config,
                       _s_bg_img=_s_bg_img, _single_bg=_single_bg,
                       _cine_cfg=_cine_cfg, _narr_overlay=_narr_overlay,
                       _sub_l=_sub_l, _sub_p=_sub_p, _sub_f=_sub_f,
                       _stype=_stype,
                       _ttl_l=_ttl_l, _ttl_p=_ttl_p, _ttl_f=_ttl_f):
            return _render_animated_frame(
                t,
                text=_text,
                width=width,
                height=height,
                bg_color=bg_color,
                text_color=text_color,
                font_size=_fs,
                accent_color_hex=_accent,
                section_accent_rgb=section_accent_rgb,
                slide_index=_idx,
                total_slides=total_slides,
                slide_duration=_dur,
                visual_config=visual_config,
                particles=_particles,
                font=_font,
                wrapped_lines=_lines,
                line_positions=_positions,
                total_text_height=_tth,
                y_start=_ys,
                bg_clip=_bg_clip,
                overlay_opacity=_overlay_opacity,
                kinetic_config=_kinetic_config,
                slide_bg_image=_s_bg_img,
                single_bg_image=_single_bg,
                cinematic_config=_cine_cfg,
                narrative_overlay=_narr_overlay,
                slide_type=_stype,
                subtitle_lines=_sub_l,
                subtitle_positions=_sub_p,
                subtitle_font=_sub_f,
                title_lines=_ttl_l,
                title_positions=_ttl_p,
                title_font=_ttl_f,
            )

        clip = VideoClip(make_frame, duration=slide_duration)

        # Attach per-slide audio (if available)
        if per_slide_audio is not None and slide_idx < len(per_slide_audio):
            slide_audio = per_slide_audio[slide_idx]
            if slide_audio is not None:
                clip = clip.with_audio(slide_audio)
                audio_clips_to_close.append(slide_audio)

        # Add fade transitions.
        # Rules:
        #   1. No CrossFadeIn on first slide (avoids black first frame)
        #   2. No crossfade between consecutive slides sharing the same AI
        #      image (avoids black flash — Ken Burns + text change is enough)
        #   3. Normal crossfade between slides with different images
        if transition_duration > 0:
            # Check if this slide and the previous share the same AI image
            prev_img = slides[slide_idx - 1].get("ai_image") if slide_idx > 0 else None
            curr_img = slide_info.get("ai_image")
            same_bg = (prev_img is not None and curr_img is not None
                       and prev_img is curr_img)
            # Check if this slide and the next share the same AI image
            next_img = slides[slide_idx + 1].get("ai_image") if slide_idx < len(slides) - 1 else None
            same_bg_next = (curr_img is not None and next_img is not None
                            and curr_img is next_img)

            effects = []
            if slide_idx > 0 and not same_bg:
                effects.append(__import__("moviepy").video.fx.CrossFadeIn(transition_duration))
            if not same_bg_next:
                effects.append(__import__("moviepy").video.fx.CrossFadeOut(transition_duration))
            if effects:
                clip = clip.with_effects(effects)

        clips.append(clip)

    console.print(f"[cyan]Concatenating {len(clips)} animated slides...[/]")

    final_video = concatenate_videoclips(clips, method="compose")

    # Overlay global voiceover audio (legacy single-file mode only)
    if global_audio is not None:
        console.print("[cyan]Overlaying voiceover audio...[/]")
        final_video = final_video.with_audio(global_audio)

    # --- Background music bed ---
    music_config = config.get("background_music", {})
    if music_config.get("enabled", False):
        from moviepy import CompositeAudioClip, concatenate_audioclips
        from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

        music_tracks = music_config.get("tracks", [])
        music_volume = music_config.get("volume", 0.08)
        fade_in = music_config.get("fade_in_seconds", 1.5)
        fade_out = music_config.get("fade_out_seconds", 2.0)

        if music_tracks:
            # Deterministic track selection (hash of title for reproducibility)
            _seed = sum(ord(c) for c in script.title)
            _track_rel = music_tracks[_seed % len(music_tracks)]
            track_path = Path(__file__).resolve().parents[2] / _track_rel

            if track_path.exists():
                music_clip = AudioFileClip(str(track_path))
                video_duration = final_video.duration

                # Loop music to cover full video duration
                if music_clip.duration < video_duration:
                    loops_needed = int(video_duration / music_clip.duration) + 1
                    music_clip = concatenate_audioclips([music_clip] * loops_needed)
                music_clip = music_clip.subclipped(0, video_duration)

                # Apply volume and fades
                music_clip = music_clip.with_volume_scaled(music_volume)
                music_clip = music_clip.with_effects([
                    AudioFadeIn(fade_in),
                    AudioFadeOut(fade_out),
                ])

                # Mix with existing audio
                existing_audio = final_video.audio
                if existing_audio is not None:
                    final_video = final_video.with_audio(
                        CompositeAudioClip([existing_audio, music_clip])
                    )
                else:
                    final_video = final_video.with_audio(music_clip)
                has_any_audio = True

                console.print(
                    f"[cyan]Background music mixed:[/] {track_path.name} "
                    f"at {music_volume * 100:.0f}% vol"
                )
            else:
                console.print(
                    f"[yellow]Music track not found: {track_path}[/yellow]"
                )

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Optimized encoding params ---
    encoding_config = config.get("encoding", {})
    ffmpeg_params = [
        "-preset", encoding_config.get("preset", "slow"),
        "-crf", str(encoding_config.get("crf", 18)),
        "-pix_fmt", encoding_config.get("pixel_format", "yuv420p"),
        "-profile:v", encoding_config.get("profile", "high"),
        "-movflags", "+faststart",  # web-optimized: moov atom at start
    ]
    audio_br = encoding_config.get("audio_bitrate", "192k")

    console.print(f"[bold cyan]Exporting video to:[/] {output_path}")
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac" if has_any_audio else None,
        audio_bitrate=audio_br if has_any_audio else None,
        ffmpeg_params=ffmpeg_params,
        logger="bar",
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]Video exported:[/] {output_path} ({file_size_mb:.1f} MB)")

    # Cleanup
    if global_audio is not None:
        global_audio.close()
    for ac in audio_clips_to_close:
        ac.close()
    if bg_clip is not None:
        bg_clip.close()
    final_video.close()

    return output_path


if __name__ == "__main__":
    console.print("[bold]Docket Social Video Generator - Animated Test[/]\n")

    test_script = VideoScript(
        title="Climate Change Is Reshaping Our Coastlines",
        hook="By 2050, 300 million people could face annual flooding.",
        body_slides=[
            "Sea levels have risen 8 inches since 1900, and the rate is accelerating.",
            "Coastal cities are already spending billions on sea walls and flood barriers.",
            "Mangrove restoration offers a natural, cost-effective defense against storm surges.",
        ],
        cta="Follow @docket for daily climate insights. Link in bio.",
        voiceover_text="",
        section="science",
    )

    output = Path("output") / "test_video_animated.mp4"
    config = _load_video_config()

    console.print("[dim]Config loaded with animation settings[/]\n")

    result = generate_video(test_script, voiceover_path=None, output_path=output, config=config)
    console.print(f"\n[bold]Test complete.[/] Video at: {result}")
