"""Tests for the After Effects cloud rendering module (nexrender backend)."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# VideoScript uses Python 3.10+ type hints in generator.py which breaks
# on 3.9. Import only the ae_renderer functions we're testing, and define
# a compatible VideoScript for test use.
from src.video.ae_renderer import (
    NexrenderClient,
    _hex_to_rgba,
    build_nexrender_job,
    select_template,
    upload_assets_for_render,
)


@dataclass
class VideoScript:
    """Test-local copy of VideoScript (compatible with Python 3.9)."""

    title: str = ""
    hook: str = ""
    body_slides: list = field(default_factory=list)
    cta: str = ""
    voiceover_text: str = ""
    section: str = ""
    video_tier: str = "narrative"
    image_prompts: list = field(default_factory=list)
    background_prompt: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_script():
    """A minimal VideoScript for testing."""
    return VideoScript(
        title="Rising Seas Threaten Pacific Islands",
        hook="A new study reveals alarming data.",
        body_slides=[
            "Sea levels have risen 3 inches in the past decade.",
            "Pacific island nations face existential threats.",
            "Experts call for immediate climate action.",
        ],
        cta="Read the full story at The Docket.",
        voiceover_text="",
        section="science",
        video_tier="cinematic",
    )


@pytest.fixture
def sample_templates():
    """Template URL mapping matching config.yaml structure."""
    return {
        "cinematic": "https://r2.example.com/templates/docket_cinematic.aep",
        "narrative": "https://r2.example.com/templates/docket_narrative.aep",
        "stock_broll": "https://r2.example.com/templates/docket_stock_broll.aep",
        "gradient": "https://r2.example.com/templates/docket_gradient.aep",
    }


@pytest.fixture
def sample_video_config():
    """Minimal video config dict for testing."""
    return {
        "background_color": "#1a1a2e",
        "text_color": "#e0e0e0",
        "duration_per_slide": 5,
        "voiceover_padding": 0.3,
        "min_slide_duration": 1.5,
        "visual": {
            "section_colors": {
                "lived": "#e07a5f",
                "systems": "#f2cc8f",
                "science": "#4ecdc4",
                "futures": "#7b68ee",
                "archive": "#a0937d",
                "lab": "#81b29a",
                "default": "#4ecdc4",
            },
            "watermark": {"text": "THE DOCKET"},
            "stock_footage": {"overlay_opacity": 0.55},
        },
        "cinematic": {
            "ken_burns_zoom": 1.12,
            "ken_burns_pan_px": 50,
            "crossfade_duration": 0.5,
            "subtitle_font_size": 44,
            "subtitle_y_fraction": 0.72,
        },
        "narrative": {"overlay_opacity": 0.55},
        "aftereffects": {
            "templates": {
                "cinematic": "https://r2.example.com/templates/docket_cinematic.aep",
                "narrative": "https://r2.example.com/templates/docket_narrative.aep",
                "stock_broll": "https://r2.example.com/templates/docket_stock_broll.aep",
                "gradient": "https://r2.example.com/templates/docket_gradient.aep",
            },
            "composition": "main",
            "poll_interval_seconds": 1,
            "render_timeout_seconds": 10,
            "encoding": {
                "preset": "slow",
                "crf": 18,
                "pixel_format": "yuv420p",
                "audio_bitrate": "192k",
            },
        },
    }


@pytest.fixture
def sample_asset_urls():
    """Staged asset URLs as returned by upload_assets_for_render."""
    return {
        "voiceover_urls": [
            "https://r2.dev/ae_staging/audio/slide_0.mp3",
            "https://r2.dev/ae_staging/audio/slide_1.mp3",
            "https://r2.dev/ae_staging/audio/slide_2.mp3",
        ],
        "slide_image_urls": [
            "https://r2.dev/ae_staging/images/slide_0.png",
            "https://r2.dev/ae_staging/images/slide_1.png",
            "https://r2.dev/ae_staging/images/slide_2.png",
        ],
        "background_image_url": None,
        "stock_clip_url": None,
    }


# ---------------------------------------------------------------------------
# Hex-to-RGBA helper tests
# ---------------------------------------------------------------------------


class TestHexToRgba:
    """Tests for _hex_to_rgba() color conversion."""

    def test_basic_conversion(self):
        result = _hex_to_rgba("#4ecdc4")
        assert len(result) == 4
        assert result[0] == round(0x4e / 255.0, 3)  # 0.306
        assert result[1] == round(0xcd / 255.0, 3)  # 0.804
        assert result[2] == round(0xc4 / 255.0, 3)  # 0.769
        assert result[3] == 1.0

    def test_black(self):
        assert _hex_to_rgba("#000000") == [0.0, 0.0, 0.0, 1.0]

    def test_white(self):
        assert _hex_to_rgba("#ffffff") == [1.0, 1.0, 1.0, 1.0]

    def test_without_hash(self):
        result = _hex_to_rgba("e07a5f")
        assert len(result) == 4
        assert result[3] == 1.0

    def test_invalid_hex_raises(self):
        with pytest.raises(ValueError, match="Invalid hex color"):
            _hex_to_rgba("#abc")


# ---------------------------------------------------------------------------
# Template selection tests
# ---------------------------------------------------------------------------


class TestSelectTemplate:
    """Tests for select_template()."""

    def test_cinematic_with_images(self, sample_templates):
        result = select_template(
            video_tier="cinematic",
            has_slide_images=True,
            has_background_image=False,
            has_stock_clip=False,
            templates=sample_templates,
        )
        assert result == "https://r2.example.com/templates/docket_cinematic.aep"

    def test_cinematic_without_images_falls_to_gradient(self, sample_templates):
        result = select_template(
            video_tier="cinematic",
            has_slide_images=False,
            has_background_image=False,
            has_stock_clip=False,
            templates=sample_templates,
        )
        assert result == "https://r2.example.com/templates/docket_gradient.aep"

    def test_narrative_with_background(self, sample_templates):
        result = select_template(
            video_tier="narrative",
            has_slide_images=False,
            has_background_image=True,
            has_stock_clip=False,
            templates=sample_templates,
        )
        assert result == "https://r2.example.com/templates/docket_narrative.aep"

    def test_narrative_with_stock_fallback(self, sample_templates):
        result = select_template(
            video_tier="narrative",
            has_slide_images=False,
            has_background_image=False,
            has_stock_clip=True,
            templates=sample_templates,
        )
        assert result == "https://r2.example.com/templates/docket_stock_broll.aep"

    def test_gradient_fallback(self, sample_templates):
        result = select_template(
            video_tier="narrative",
            has_slide_images=False,
            has_background_image=False,
            has_stock_clip=False,
            templates=sample_templates,
        )
        assert result == "https://r2.example.com/templates/docket_gradient.aep"

    def test_missing_template_raises(self):
        with pytest.raises(ValueError, match="No template URL configured"):
            select_template(
                video_tier="cinematic",
                has_slide_images=True,
                has_background_image=False,
                has_stock_clip=False,
                templates={},  # empty — no templates configured
            )

    def test_stock_preferred_over_gradient(self, sample_templates):
        """When no AI images but stock is available, prefer stock_broll."""
        result = select_template(
            video_tier="cinematic",
            has_slide_images=False,
            has_background_image=False,
            has_stock_clip=True,
            templates=sample_templates,
        )
        assert result == "https://r2.example.com/templates/docket_stock_broll.aep"


# ---------------------------------------------------------------------------
# Nexrender job builder tests
# ---------------------------------------------------------------------------


def _find_asset(assets, layer_name, asset_type=None):
    """Helper: find an asset by layerName (and optionally type) in a job's assets list."""
    for a in assets:
        if a.get("layerName") == layer_name:
            if asset_type is None or a.get("type") == asset_type:
                return a
    return None


class TestBuildNexrenderJob:
    """Tests for build_nexrender_job()."""

    def test_job_structure(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/templates/docket_cinematic.aep",
        )
        assert "template" in job
        assert "assets" in job
        assert "actions" in job
        assert job["template"]["src"] == "https://r2.example.com/templates/docket_cinematic.aep"
        assert job["template"]["composition"] == "main"

    def test_text_layers_populated(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        title = _find_asset(assets, "title_text")
        assert title is not None
        assert title["type"] == "data"
        assert title["property"] == "Source Text"
        assert title["value"] == "Rising Seas Threaten Pacific Islands"

        hook = _find_asset(assets, "hook_text")
        assert hook["value"] == "A new study reveals alarming data."

        body1 = _find_asset(assets, "body_text_1")
        assert body1["value"] == "Sea levels have risen 3 inches in the past decade."

        cta = _find_asset(assets, "cta_text")
        assert cta["value"] == "Read the full story at The Docket."

    def test_unused_body_slots_empty(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        num_slides = _find_asset(assets, "num_body_slides")
        assert num_slides["value"] == 3

        body4 = _find_asset(assets, "body_text_4")
        assert body4["value"] == ""
        body5 = _find_asset(assets, "body_text_5")
        assert body5["value"] == ""

    def test_section_color_as_rgba(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        color_asset = _find_asset(assets, "section_color")
        assert color_asset["type"] == "data"
        assert color_asset["property"] == "Effects.Fill.Color"
        # science = #4ecdc4
        assert isinstance(color_asset["value"], list)
        assert len(color_asset["value"]) == 4
        assert color_asset["value"][3] == 1.0

    def test_cinematic_params_included(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        zoom = _find_asset(assets, "ken_burns_zoom")
        assert zoom is not None
        assert zoom["value"] == 1.12

        font_size = _find_asset(assets, "subtitle_font_size")
        assert font_size["value"] == 44

        # Check slide images are type "image"
        img0 = _find_asset(assets, "slide_image_0", "image")
        assert img0 is not None
        assert img0["src"] == "https://r2.dev/ae_staging/images/slide_0.png"

    def test_narrative_params_included(
        self, sample_script, sample_video_config
    ):
        asset_urls = {
            "voiceover_urls": [],
            "slide_image_urls": [],
            "background_image_url": "https://r2.dev/bg.png",
            "stock_clip_url": None,
        }
        job = build_nexrender_job(
            sample_script, asset_urls, sample_video_config, "narrative",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        opacity = _find_asset(assets, "overlay_opacity")
        assert opacity is not None
        assert opacity["value"] == 0.55

        bg = _find_asset(assets, "background_image", "image")
        assert bg is not None
        assert bg["src"] == "https://r2.dev/bg.png"

        # cinematic params should NOT be present
        assert _find_asset(assets, "ken_burns_zoom") is None

    def test_voiceover_urls_as_audio_assets(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        audio0 = _find_asset(assets, "audio_slide_0", "audio")
        assert audio0 is not None
        assert audio0["src"] == "https://r2.dev/ae_staging/audio/slide_0.mp3"

        audio1 = _find_asset(assets, "audio_slide_1", "audio")
        assert audio1["src"] == "https://r2.dev/ae_staging/audio/slide_1.mp3"

    def test_unknown_section_uses_default_color(
        self, sample_asset_urls, sample_video_config
    ):
        script = VideoScript(
            title="Test", hook="", body_slides=["Slide 1"],
            section="unknown_section",
        )
        job = build_nexrender_job(
            script, sample_asset_urls, sample_video_config, "narrative",
            template_url="https://r2.example.com/test.aep",
        )
        color_asset = _find_asset(job["assets"], "section_color")
        # default = #4ecdc4
        assert color_asset["value"] == _hex_to_rgba("#4ecdc4")

    def test_empty_hook_and_cta(
        self, sample_asset_urls, sample_video_config
    ):
        script = VideoScript(
            title="Test", hook="", body_slides=["Content"], cta="",
            section="lab",
        )
        job = build_nexrender_job(
            script, sample_asset_urls, sample_video_config, "narrative",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        hook = _find_asset(assets, "hook_text")
        assert hook["value"] == ""
        cta = _find_asset(assets, "cta_text")
        assert cta["value"] == ""

    def test_postrender_encoding_params(
        self, sample_script, sample_asset_urls, sample_video_config
    ):
        job = build_nexrender_job(
            sample_script, sample_asset_urls, sample_video_config, "cinematic",
            template_url="https://r2.example.com/test.aep",
        )
        postrender = job["actions"]["postrender"]
        assert len(postrender) == 1
        action = postrender[0]
        assert action["module"] == "@nexrender/action-encode"
        assert action["output"] == "output.mp4"
        assert action["params"]["-c:v"] == "libx264"
        assert action["params"]["-preset"] == "slow"
        assert action["params"]["-crf"] == "18"
        assert action["params"]["-pix_fmt"] == "yuv420p"

    def test_stock_footage_as_video_asset(
        self, sample_script, sample_video_config
    ):
        asset_urls = {
            "voiceover_urls": [],
            "slide_image_urls": [],
            "background_image_url": None,
            "stock_clip_url": "https://r2.dev/stock/broll.mp4",
        }
        job = build_nexrender_job(
            sample_script, asset_urls, sample_video_config, "narrative",
            template_url="https://r2.example.com/test.aep",
        )
        assets = job["assets"]

        stock = _find_asset(assets, "stock_footage", "video")
        assert stock is not None
        assert stock["src"] == "https://r2.dev/stock/broll.mp4"


# ---------------------------------------------------------------------------
# NexrenderClient tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestNexrenderClient:
    """Tests for the NexrenderClient REST wrapper."""

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
        "NEXRENDER_SECRET": "test_secret",
    })
    def test_init_with_env(self):
        client = NexrenderClient()
        assert client.server_url == "http://localhost:3000"
        assert client.secret == "test_secret"
        client.close()

    def test_init_missing_url_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="NEXRENDER_SERVER_URL"):
                NexrenderClient()

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
        "NEXRENDER_SECRET": "secret",
    })
    def test_create_job(self):
        client = NexrenderClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"uid": "job_001", "state": "queued"}
        mock_response.raise_for_status = MagicMock()
        client._client.post = MagicMock(return_value=mock_response)

        result = client.create_job({"template": {"src": "test.aep"}})

        assert result["uid"] == "job_001"
        client._client.post.assert_called_once_with(
            "/api/v1/jobs",
            json={"template": {"src": "test.aep"}},
        )
        client.close()

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
        "NEXRENDER_SECRET": "secret",
    })
    def test_get_job(self):
        client = NexrenderClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "uid": "job_001",
            "state": "finished",
            "output": "https://r2.dev/output.mp4",
        }
        mock_response.raise_for_status = MagicMock()
        client._client.get = MagicMock(return_value=mock_response)

        result = client.get_job("job_001")

        assert result["state"] == "finished"
        assert "output" in result
        client.close()

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
        "NEXRENDER_SECRET": "secret",
    })
    def test_wait_for_job_success(self):
        client = NexrenderClient()

        # Simulate: queued → started → finished
        responses = [
            {"uid": "j1", "state": "queued"},
            {"uid": "j1", "state": "started"},
            {"uid": "j1", "state": "finished", "output": "https://out.mp4"},
        ]
        call_count = 0

        def mock_get_job(job_uid):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        client.get_job = mock_get_job

        result = client.wait_for_job("j1", poll_interval=0, timeout=5)
        assert result["state"] == "finished"
        client.close()

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
        "NEXRENDER_SECRET": "secret",
    })
    def test_wait_for_job_failure(self):
        client = NexrenderClient()
        client.get_job = lambda _: {
            "uid": "j1",
            "state": "error",
            "errorMessage": "Template missing",
        }

        with pytest.raises(RuntimeError, match="Template missing"):
            client.wait_for_job("j1", poll_interval=0, timeout=5)
        client.close()

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
        "NEXRENDER_SECRET": "secret",
    })
    def test_wait_for_job_timeout(self):
        client = NexrenderClient()
        client.get_job = lambda _: {"uid": "j1", "state": "started"}

        with pytest.raises(TimeoutError, match="timed out"):
            client.wait_for_job("j1", poll_interval=0, timeout=0)
        client.close()

    @patch.dict("os.environ", {
        "NEXRENDER_SERVER_URL": "http://localhost:3000",
    })
    def test_init_without_secret(self):
        """Client should work without a secret (nexrender allows it)."""
        client = NexrenderClient()
        assert client.secret == ""
        client.close()


# ---------------------------------------------------------------------------
# Asset staging tests (mocked uploader)
# ---------------------------------------------------------------------------


class TestUploadAssets:
    """Tests for upload_assets_for_render() with mocked R2 uploads."""

    @patch("src.video.uploader.upload_video")
    def test_stages_voiceover_list(self, mock_upload, tmp_path):
        audio_0 = tmp_path / "slide_0.mp3"
        audio_1 = tmp_path / "slide_1.mp3"
        audio_0.write_bytes(b"fake audio 0")
        audio_1.write_bytes(b"fake audio 1")

        mock_upload.side_effect = lambda p, k: f"https://r2.dev/{k}"

        result = upload_assets_for_render(
            voiceover_paths=[audio_0, audio_1],
            slide_image_paths=None,
            background_image_path=None,
            stock_clip_path=None,
        )

        assert len(result["voiceover_urls"]) == 2
        assert "slide_0.mp3" in result["voiceover_urls"][0]
        assert mock_upload.call_count == 2

    @patch("src.video.uploader.upload_video")
    def test_stages_slide_images(self, mock_upload, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"fake png")

        mock_upload.side_effect = lambda p, k: f"https://r2.dev/{k}"

        result = upload_assets_for_render(
            voiceover_paths=None,
            slide_image_paths=[img, None, img],
            background_image_path=None,
            stock_clip_path=None,
        )

        assert len(result["slide_image_urls"]) == 3
        assert result["slide_image_urls"][0] is not None
        assert result["slide_image_urls"][1] is None
        assert result["slide_image_urls"][2] is not None

    @patch("src.video.uploader.upload_video")
    def test_stages_background_image(self, mock_upload, tmp_path):
        bg = tmp_path / "bg.png"
        bg.write_bytes(b"fake bg")

        mock_upload.side_effect = lambda p, k: f"https://r2.dev/{k}"

        result = upload_assets_for_render(
            voiceover_paths=None,
            slide_image_paths=None,
            background_image_path=bg,
            stock_clip_path=None,
        )

        assert result["background_image_url"] is not None
        assert "background.png" in result["background_image_url"]

    @patch("src.video.uploader.upload_video")
    def test_stages_stock_clip(self, mock_upload, tmp_path):
        stock = tmp_path / "broll.mp4"
        stock.write_bytes(b"fake video")

        mock_upload.side_effect = lambda p, k: f"https://r2.dev/{k}"

        result = upload_assets_for_render(
            voiceover_paths=None,
            slide_image_paths=None,
            background_image_path=None,
            stock_clip_path=stock,
        )

        assert result["stock_clip_url"] is not None
        assert "broll.mp4" in result["stock_clip_url"]

    @patch("src.video.uploader.upload_video")
    def test_handles_none_paths(self, mock_upload):
        result = upload_assets_for_render(
            voiceover_paths=None,
            slide_image_paths=None,
            background_image_path=None,
            stock_clip_path=None,
        )

        assert result["voiceover_urls"] == []
        assert result["slide_image_urls"] == []
        assert result["background_image_url"] is None
        assert result["stock_clip_url"] is None
        mock_upload.assert_not_called()
