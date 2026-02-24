"""ElevenLabs text-to-speech integration for Docket Social video pipeline."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from rich.console import Console

load_dotenv()

console = Console()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


def _get_client() -> ElevenLabs:
    """Create and return an authenticated ElevenLabs client."""
    if not ELEVENLABS_API_KEY:
        console.print("[bold red]Error:[/] ELEVENLABS_API_KEY not set in environment.")
        raise EnvironmentError("ELEVENLABS_API_KEY is required. Set it in your .env file.")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


def generate_voiceover(text: str, output_path: Path) -> Path:
    """Generate speech audio from text using ElevenLabs API.

    Args:
        text: The text to convert to speech.
        output_path: Where to save the resulting MP3 file.

    Returns:
        The path to the saved MP3 file.
    """
    if not ELEVENLABS_VOICE_ID:
        console.print("[bold red]Error:[/] ELEVENLABS_VOICE_ID not set in environment.")
        raise EnvironmentError(
            "ELEVENLABS_VOICE_ID is required. Set it in your .env file. "
            "Run list_voices() to see available voice IDs."
        )

    client = _get_client()

    console.print(f"[bold cyan]Generating voiceover[/] ({len(text)} chars) with voice [yellow]{ELEVENLABS_VOICE_ID}[/]")

    # Load voiceover settings from environment or use defaults matching config.yaml
    model_id = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    stability = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
    similarity_boost = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))

    audio_iterator = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id=model_id,
        output_format=output_format,
        voice_settings={
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in audio_iterator:
            f.write(chunk)

    file_size_kb = output_path.stat().st_size / 1024
    console.print(f"[bold green]Voiceover saved:[/] {output_path} ({file_size_kb:.1f} KB)")
    return output_path


def generate_voiceover_per_slide(
    slide_texts: list[str],
    output_dir: Path,
    slide_labels: list[str] | None = None,
) -> list[Path | None]:
    """Generate separate voiceover audio for each slide.

    Calls ElevenLabs once per slide text, saving individual MP3 files.
    Empty texts produce None entries. Individual failures are logged
    and skipped so other slides still get audio.

    Args:
        slide_texts: Text to speak for each slide (in order).
        output_dir: Directory to save audio files.
        slide_labels: Optional labels for filenames (e.g., ["title", "hook"]).

    Returns:
        List of Paths to generated MP3 files (None for skipped/failed slides).
    """
    if not slide_texts:
        return []

    if slide_labels is None:
        slide_labels = [f"slide_{i}" for i in range(len(slide_texts))]

    if not ELEVENLABS_VOICE_ID:
        raise EnvironmentError("ELEVENLABS_VOICE_ID is required.")

    client = _get_client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    stability = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
    similarity_boost = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))

    console.print(
        f"[bold cyan]Generating {len(slide_texts)} voiceover segments[/] "
        f"with voice [yellow]{ELEVENLABS_VOICE_ID}[/]"
    )

    audio_paths: list[Path | None] = []

    for text, label in zip(slide_texts, slide_labels):
        if not text or not text.strip():
            audio_paths.append(None)
            continue

        try:
            audio_iterator = client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text,
                model_id=model_id,
                output_format=output_format,
                voice_settings={
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                },
            )

            out_path = output_dir / f"{label}.mp3"
            with open(out_path, "wb") as f:
                for chunk in audio_iterator:
                    f.write(chunk)

            size_kb = out_path.stat().st_size / 1024
            console.print(f"  [green]{label}:[/] {size_kb:.0f} KB  [dim]{text[:50]}{'...' if len(text)>50 else ''}[/dim]")
            audio_paths.append(out_path)

        except Exception as e:
            console.print(f"  [red]{label}: failed ({e})[/red]")
            audio_paths.append(None)

    ok = sum(1 for p in audio_paths if p)
    console.print(f"[bold green]Voiceover: {ok}/{len(slide_texts)} segments generated[/]")
    return audio_paths


def list_voices() -> list[dict]:
    """Fetch and return available ElevenLabs voices.

    Returns:
        A list of dicts with voice id, name, category, and description.
    """
    client = _get_client()

    console.print("[bold cyan]Fetching available voices...[/]")
    response = client.voices.get_all()

    voices = []
    for voice in response.voices:
        entry = {
            "voice_id": voice.voice_id,
            "name": voice.name,
            "category": getattr(voice, "category", "unknown"),
            "description": getattr(voice, "description", ""),
        }
        voices.append(entry)

    console.print(f"[bold green]Found {len(voices)} voices:[/]")
    for v in voices:
        console.print(f"  [yellow]{v['voice_id']}[/] - {v['name']} ({v['category']})")

    return voices


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[bold yellow]Usage:[/] python -m src.video.voiceover \"Text to speak\"")
        console.print("\nAvailable voices:")
        list_voices()
        sys.exit(0)

    text = " ".join(sys.argv[1:])
    output = Path("output") / "test_voiceover.mp3"
    generate_voiceover(text, output)
    console.print(f"\n[bold]Test complete.[/] Play the file at: {output}")
