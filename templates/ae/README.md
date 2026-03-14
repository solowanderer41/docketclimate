# After Effects Template Specification — The Docket

This document defines the layer naming conventions, dynamic properties,
and design constraints for AE templates used by the Docket's automated
Instagram Reels pipeline.

The Python pipeline populates these templates via Nexrender's REST API
(self-hosted AE rendering). Each dynamic value maps to a named layer or
property in the AE project. Templates are hosted on R2 and fetched by
the nexrender worker at render time.

## Canvas

- Resolution: **1080 x 1920** (9:16 vertical)
- Frame rate: **30 fps**
- Duration: **variable** (15-60s, driven by voiceover audio lengths)
- Export: **H.264**, CRF 18, yuv420p, AAC 192k

## Templates

| File | Tier | Description |
|------|------|-------------|
| `docket_cinematic.aep` | 1 | Per-slide AI images, Ken Burns zoom, subtitle captions |
| `docket_narrative.aep` | 2 | Single AI background, dark overlay, text card overlay |
| `docket_stock_broll.aep` | — | Stock B-roll video background, kinetic text animation |
| `docket_gradient.aep` | — | Animated gradient background (no external imagery) |

## Dynamic Layers (All Templates)

These layer names are **required** across all four templates. Nexrender
injects values by matching these exact layer names in the AE project.

### Text Layers

| Layer Name | Type | Description |
|------------|------|-------------|
| `title_text` | Text | ALL CAPS headline (Oswald Bold, 64pt) |
| `hook_text` | Text | Hook slide copy (may be empty) |
| `body_text_1` | Text | Body slide 1 |
| `body_text_2` | Text | Body slide 2 |
| `body_text_3` | Text | Body slide 3 |
| `body_text_4` | Text | Body slide 4 |
| `body_text_5` | Text | Body slide 5 |
| `cta_text` | Text | Call-to-action (may be empty — hide layer when empty) |
| `watermark_text` | Text | "THE DOCKET" (bottom-right, 24pt, 15% opacity) |

> **Visibility rule:** Hide `body_text_N` layers where N > `num_body_slides`.
> Hide `hook_text` when the value is empty. Hide `cta_text` when empty.

### Numeric Parameters (Slider Controls)

These are injected via AE Expression Slider Controls on a control null layer.
Each template must have a null object with slider control effects matching
these names.

| Layer Name | Type | Description |
|-----------|------|-------------|
| `num_body_slides` | Slider | How many body slides are active (1-5) |
| `voiceover_padding` | Slider | Seconds of silence after each slide's audio |
| `min_slide_duration` | Slider | Floor duration even if audio is very short |
| `default_slide_duration` | Slider | Fallback duration when no audio is present (5s) |

### Color Parameters (Fill Effects)

Colors are injected as normalized RGBA arrays `[r, g, b, 1.0]` (0.0-1.0)
via `Effects.Fill.Color` properties. Each color layer needs a Fill effect.

| Layer Name | Example (hex) | Description |
|-----------|---------------|-------------|
| `section_color` | `#4ecdc4` → `[0.306, 0.804, 0.769, 1.0]` | Section accent |
| `progress_bar_color` | `#4ecdc4` | Progress bar fill |
| `background_color` | `#1a1a2e` | Base background |
| `text_color` | `#e0e0e0` | Primary text color |

### Audio Layers

Nexrender replaces these layers with audio files fetched from R2 URLs.

| Layer Name | Nexrender Type | Description |
|------------|---------------|-------------|
| `audio_slide_0` | `audio` | Title slide voiceover |
| `audio_slide_1` | `audio` | Hook or Body 1 voiceover |
| `audio_slide_2` | `audio` | Body 2 voiceover |
| ... | ... | (up to ~7 slides) |

Audio files are MP3, 44.1kHz, 128kbps. Slide duration should be:
`max(min_slide_duration, audio_duration + voiceover_padding)`

## Cinematic Template (`docket_cinematic.aep`)

Additional dynamic layers:

| Layer/Param | Type | Description |
|-------------|------|-------------|
| `slide_image_0` | URL → Image | Title slide AI image (1088x1920 PNG) |
| `slide_image_1` | URL → Image | Hook/Body 1 AI image |
| `slide_image_2` | URL → Image | Body 2 AI image |
| ... | ... | One image per slide |
| `ken_burns_zoom` | Float | Zoom factor per slide (default 1.12) |
| `ken_burns_pan_px` | Integer | Pan distance in pixels (default 50) |
| `crossfade_duration` | Float | Seconds of crossfade between slides (0.5) |
| `subtitle_font_size` | Integer | Caption font size (default 44pt) |
| `subtitle_y_fraction` | Float | Vertical position 0-1 (0.72 = lower third) |

**Design notes:**
- Each slide shows a full-bleed AI image with slow Ken Burns zoom + pan
- Text appears as lower-third subtitles in a dark pill background
- Subtitle pill: `#000000` at 65% opacity, 24px horizontal / 12px vertical padding, 12px border radius
- Crossfade between slides (not hard cuts)

## Narrative Template (`docket_narrative.aep`)

Additional dynamic layers:

| Layer/Param | Type | Description |
|-------------|------|-------------|
| `background_image` | URL → Image | Single AI background (1088x1920 PNG) |
| `overlay_opacity` | Float | Dark overlay intensity (default 0.55) |

**Design notes:**
- Single background image holds throughout the video
- Dark overlay sits between background and text layers
- Text cards animate in/out per slide (fade or slide transitions)

## Stock B-Roll Template (`docket_stock_broll.aep`)

Additional dynamic layers:

| Layer/Param | Type | Description |
|-------------|------|-------------|
| `stock_footage` | URL → Video | Background B-roll clip (MP4) |
| `overlay_opacity` | Float | Dark overlay intensity (default 0.55) |

**Design notes:**
- Stock video loops behind a dark overlay
- Text uses kinetic line-by-line animation (stagger 0.25s, fade 0.3s)

## Gradient Template (`docket_gradient.aep`)

No additional dynamic layers beyond the shared set.

**Design notes:**
- Animated breathing gradient using `background_color` + `section_color`
- Gradient angle: 160 degrees (top-left to bottom-right)
- Breathing animation: 0.4 cycles/sec, 8% amplitude shift
- Floating particle field: 35 particles, bokeh-style, section_color at 12% opacity

## Section Color Reference

| Section ID | Color | Name |
|------------|-------|------|
| `lived` | `#e07a5f` | Warm coral |
| `systems` | `#f2cc8f` | Amber |
| `science` | `#4ecdc4` | Teal |
| `futures` | `#7b68ee` | Slate blue |
| `archive` | `#a0937d` | Warm taupe |
| `lab` | `#81b29a` | Sage green |
| `default` | `#4ecdc4` | Teal |

## Instagram Safe Zones

Keep essential text within these margins (Instagram UI overlays):
- **Top:** 250px (username, camera icon)
- **Bottom:** 400px (caption, buttons, navigation bar)
- **Left/Right:** 60px each

## Font Stack

- **Primary:** Oswald Bold (titles), Oswald Regular (body)
- **Fallback:** Arial Bold / Arial (system fonts)
- Ensure Oswald is bundled in the AE project or available on the render machine.

## Encoding

Nexrender uses `@nexrender/action-encode` (ffmpeg) as a postrender step.
Output settings are configured in `config.yaml` under `video.aftereffects.encoding`:
- Codec: H.264 (`libx264`), preset `slow`
- CRF: 18 (visually lossless — Instagram re-encodes)
- Pixel format: yuv420p
- Audio: AAC 192kbps
- Container: MP4
