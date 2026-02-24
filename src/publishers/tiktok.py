"""
TikTok Publisher Module for Docket Social
==========================================

Publishes videos to TikTok via the Content Posting API.

The TikTok Content Posting API uses a multi-step flow:
    1. Initialize the upload (POST /v2/post/publish/inbox/video/init/)
       - TikTok returns an upload_url
    2. Upload the video binary to that upload_url via PUT
    3. TikTok processes the video and delivers it to the creator's inbox
       in the TikTok app, where they confirm and publish

NOTE: TikTok's "direct post" API posts videos to the creator's inbox, not
directly to their public profile. The creator must open TikTok and tap
"Post" to finalize. This is a TikTok platform requirement for third-party apps.

Required API Credentials (in .env):
    TIKTOK_CLIENT_KEY     - Your TikTok app's Client Key
    TIKTOK_CLIENT_SECRET  - Your TikTok app's Client Secret
    TIKTOK_ACCESS_TOKEN   - OAuth 2.0 access token for the user

How to obtain credentials:
    1. Register as a TikTok developer at https://developers.tiktok.com/
    2. Create an App in the developer portal
    3. Request the "Content Posting API" scope (video.publish)
    4. Implement OAuth 2.0 authorization flow to obtain a user access token
    5. The Client Key and Secret are found in your app's settings page
    6. Access tokens must be refreshed periodically using the refresh token
"""

import os
import time

import requests
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

TIKTOK_API_BASE = "https://open.tiktokapis.com"
INIT_ENDPOINT = f"{TIKTOK_API_BASE}/v2/post/publish/inbox/video/init/"
STATUS_ENDPOINT = f"{TIKTOK_API_BASE}/v2/post/publish/status/fetch/"

# Chunk size for video upload (default 10 MB)
UPLOAD_CHUNK_SIZE = 10 * 1024 * 1024
STATUS_POLL_INTERVAL = 5  # seconds
STATUS_POLL_TIMEOUT = 300  # max seconds to wait


class TikTokPublisher:
    """Client for publishing videos to TikTok via the Content Posting API."""

    def __init__(self):
        self.client_key = os.getenv("TIKTOK_CLIENT_KEY")
        self.client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
        self.access_token = os.getenv("TIKTOK_ACCESS_TOKEN")

        missing = []
        if not self.client_key:
            missing.append("TIKTOK_CLIENT_KEY")
        if not self.client_secret:
            missing.append("TIKTOK_CLIENT_SECRET")
        if not self.access_token:
            missing.append("TIKTOK_ACCESS_TOKEN")

        if missing:
            console.print(
                f"[bold red]Missing TikTok credentials in .env: "
                f"{', '.join(missing)}[/bold red]"
            )

    def _auth_headers(self) -> dict:
        """Return authorization headers for TikTok API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    def login(self):
        """
        Verify the access token by fetching user info.

        TikTok does not have a dedicated login endpoint for content posting;
        this calls the user info endpoint to validate the token.
        """
        try:
            response = requests.get(
                f"{TIKTOK_API_BASE}/v2/user/info/",
                headers=self._auth_headers(),
                params={"fields": "open_id,display_name"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("error", {}).get("code", 0) != 0:
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"TikTok API error: {error_msg}")

            user_data = data.get("data", {}).get("user", {})
            display_name = user_data.get("display_name", "Unknown")
            console.print(
                f"[bold green]TikTok token valid. User: {display_name}[/bold green]"
            )
            return user_data
        except requests.RequestException as e:
            console.print(
                f"[bold red]TikTok token validation failed: {e}[/bold red]"
            )
            raise

    def _get_video_size(self, video_path: str) -> int:
        """Get the file size in bytes."""
        return os.path.getsize(video_path)

    def _init_upload(self, video_size: int, caption: str) -> dict:
        """
        Initialize a video upload with TikTok.

        Args:
            video_size: Size of the video file in bytes.
            caption: Caption/title for the video.

        Returns:
            dict containing 'publish_id' and 'upload_url'.
        """
        payload = {
            "post_info": {
                "title": caption[:150] if caption else "",
                "privacy_level": "SELF_ONLY",  # Default to private; user confirms in app
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": min(UPLOAD_CHUNK_SIZE, video_size),
                "total_chunk_count": max(
                    1, (video_size + UPLOAD_CHUNK_SIZE - 1) // UPLOAD_CHUNK_SIZE
                ),
            },
        }

        response = requests.post(
            INIT_ENDPOINT,
            headers=self._auth_headers(),
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error", {}).get("code", 0) != 0:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            raise RuntimeError(f"TikTok upload init failed: {error_msg}")

        upload_data = data.get("data", {})
        publish_id = upload_data.get("publish_id", "")
        upload_url = upload_data.get("upload_url", "")

        console.print(f"[dim]TikTok upload initialized: {publish_id}[/dim]")
        return {"publish_id": publish_id, "upload_url": upload_url}

    def _upload_video_file(self, upload_url: str, video_path: str, video_size: int):
        """
        Upload the video binary to TikTok's upload URL.

        For small files (under chunk size), uploads in a single PUT request.
        For larger files, uploads in chunks.

        Args:
            upload_url: The URL provided by TikTok for the upload.
            video_path: Local path to the video file.
            video_size: Size of the video in bytes.
        """
        if video_size <= UPLOAD_CHUNK_SIZE:
            # Single-chunk upload
            with open(video_path, "rb") as f:
                headers = {
                    "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
                    "Content-Type": "video/mp4",
                }
                response = requests.put(
                    upload_url,
                    headers=headers,
                    data=f,
                    timeout=120,
                )
                response.raise_for_status()
            console.print("[dim]Video uploaded to TikTok (single chunk).[/dim]")
        else:
            # Multi-chunk upload
            chunk_index = 0
            with open(video_path, "rb") as f:
                while True:
                    chunk = f.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break

                    offset = chunk_index * UPLOAD_CHUNK_SIZE
                    end = offset + len(chunk) - 1
                    headers = {
                        "Content-Range": f"bytes {offset}-{end}/{video_size}",
                        "Content-Type": "video/mp4",
                    }
                    response = requests.put(
                        upload_url,
                        headers=headers,
                        data=chunk,
                        timeout=120,
                    )
                    response.raise_for_status()

                    chunk_index += 1
                    console.print(
                        f"[dim]Uploaded chunk {chunk_index} "
                        f"({end + 1}/{video_size} bytes)[/dim]"
                    )

            console.print(
                f"[dim]Video uploaded to TikTok ({chunk_index} chunks).[/dim]"
            )

    def _check_publish_status(self, publish_id: str) -> dict:
        """
        Poll TikTok for the publish status of an uploaded video.

        Args:
            publish_id: The publish ID from the init step.

        Returns:
            dict with status information.
        """
        payload = {"publish_id": publish_id}
        start_time = time.time()

        while time.time() - start_time < STATUS_POLL_TIMEOUT:
            try:
                response = requests.post(
                    STATUS_ENDPOINT,
                    headers=self._auth_headers(),
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

                status = data.get("data", {}).get("status", "PROCESSING")

                if status == "PUBLISH_COMPLETE":
                    console.print(
                        "[dim]TikTok video delivered to inbox.[/dim]"
                    )
                    return data.get("data", {})
                elif status in ("FAILED", "PUBLISH_FAILED"):
                    fail_reason = data.get("data", {}).get("fail_reason", "Unknown")
                    raise RuntimeError(
                        f"TikTok publish failed: {fail_reason}"
                    )
                else:
                    elapsed = int(time.time() - start_time)
                    console.print(
                        f"[dim]TikTok processing... status={status} "
                        f"({elapsed}s elapsed)[/dim]"
                    )
            except requests.RequestException as e:
                console.print(
                    f"[bold yellow]Status check error (will retry): {e}[/bold yellow]"
                )

            time.sleep(STATUS_POLL_INTERVAL)

        console.print(
            f"[bold yellow]TikTok status polling timed out after "
            f"{STATUS_POLL_TIMEOUT}s. The video may still be processing. "
            f"Check your TikTok inbox.[/bold yellow]"
        )
        return {"status": "TIMEOUT"}

    def publish_video(self, video_path: str, caption: str) -> dict:
        """
        Publish a video to TikTok.

        The video is uploaded to TikTok and delivered to the creator's inbox.
        The creator must open the TikTok app and confirm the post.

        Args:
            video_path: Local file path to the video (MP4 recommended).
            caption:    Caption/title for the video (max 150 characters).

        Returns:
            dict with 'publish_id' and processing status.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_size = self._get_video_size(video_path)
        console.print(
            f"[dim]Preparing to upload {video_path} "
            f"({video_size / (1024 * 1024):.1f} MB)[/dim]"
        )

        try:
            # Step 1: Initialize the upload
            init_data = self._init_upload(video_size, caption)
            publish_id = init_data["publish_id"]
            upload_url = init_data["upload_url"]

            # Step 2: Upload the video file
            self._upload_video_file(upload_url, video_path, video_size)

            # Step 3: Check publish status
            status_data = self._check_publish_status(publish_id)

            result = {
                "publish_id": publish_id,
                "status": status_data.get("status", "UNKNOWN"),
            }
            console.print(
                f"[bold green]TikTok video uploaded.[/bold green] "
                f"Publish ID: {publish_id}. "
                f"Open TikTok to confirm and post."
            )
            return result

        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            console.print(
                f"[bold red]TikTok API error: {e} {error_detail}[/bold red]"
            )
            raise
        except Exception as e:
            console.print(
                f"[bold red]Failed to publish to TikTok: {e}[/bold red]"
            )
            raise

    def publish(self, post: dict) -> dict:
        """
        Publish a post using a standardized post dict.

        Args:
            post: dict with 'video_path' and optionally 'text' (used as caption).

        Returns:
            dict with publish ID and status.
        """
        video_path = post.get("video_path", "")
        caption = post.get("text", post.get("caption", ""))
        if not video_path:
            raise ValueError(
                "post dict must include 'video_path' for TikTok publishing."
            )
        return self.publish_video(video_path, caption)


def publish(post: dict) -> dict:
    """
    Module-level convenience function to publish a video to TikTok.

    Args:
        post: dict with 'video_path' and optionally 'text' (used as caption).

    Returns:
        dict with publish ID and status.
    """
    publisher = TikTokPublisher()
    return publisher.publish(post)


def test_connection() -> bool:
    """
    Verify that TikTok credentials are valid by checking the access token.

    Returns:
        True if the token is valid, False otherwise.
    """
    try:
        publisher = TikTokPublisher()
        publisher.login()
        console.print("[bold green]TikTok connection test passed.[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]TikTok connection test failed: {e}[/bold red]")
        return False
