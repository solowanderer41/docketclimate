"""
Cloud storage uploader for video files (S3-compatible).

Uploads generated MP4 video files to an S3-compatible storage service
(Cloudflare R2, AWS S3, Backblaze B2) and returns a publicly accessible
HTTPS URL that the Instagram Reels API can fetch from.

Required environment variables (in .env):
    STORAGE_ENDPOINT_URL  - S3-compatible API endpoint
                            (e.g. https://<account-id>.r2.cloudflarestorage.com)
    STORAGE_ACCESS_KEY    - Access key ID
    STORAGE_SECRET_KEY    - Secret access key
    STORAGE_BUCKET        - Bucket name (e.g. docket-videos)
    STORAGE_PUBLIC_URL    - Public base URL for accessing files
                            (e.g. https://pub-<hash>.r2.dev)
"""

import os
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

_REQUIRED_VARS = [
    "STORAGE_ENDPOINT_URL",
    "STORAGE_ACCESS_KEY",
    "STORAGE_SECRET_KEY",
    "STORAGE_BUCKET",
    "STORAGE_PUBLIC_URL",
]


def _validate_env():
    """Check that all required environment variables are set."""
    missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing cloud storage env vars: {', '.join(missing)}. "
            f"Set them in .env — see .env.example for details."
        )


def _get_client():
    """Create and return a boto3 S3 client for the configured endpoint."""
    _validate_env()
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("STORAGE_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("STORAGE_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("STORAGE_SECRET_KEY"),
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def upload_video(local_path: Path, key: str) -> str:
    """
    Upload a video file to S3-compatible cloud storage.

    Args:
        local_path: Path to the local MP4 file.
        key: S3 object key (e.g. "videos/18_mon_046.mp4").

    Returns:
        Public HTTPS URL where the video is accessible.
    """
    client = _get_client()
    bucket = os.getenv("STORAGE_BUCKET")
    public_base = os.getenv("STORAGE_PUBLIC_URL", "").rstrip("/")

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Video file not found: {local_path}")

    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    console.print(
        f"[cyan]Uploading {local_path.name} "
        f"({file_size_mb:.1f} MB) to {bucket}/{key}[/cyan]"
    )

    client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    public_url = f"{public_base}/{key}"
    console.print(f"[green]Uploaded: {public_url}[/green]")
    return public_url


def test_storage_connection() -> bool:
    """
    Test connectivity to the configured storage bucket.

    Returns True if successful, raises on failure.
    """
    _validate_env()
    client = _get_client()
    bucket = os.getenv("STORAGE_BUCKET")

    console.print(f"[cyan]Testing connection to bucket '{bucket}'...[/cyan]")
    client.head_bucket(Bucket=bucket)
    console.print(f"[green]Storage connection OK ({bucket})[/green]")
    return True
