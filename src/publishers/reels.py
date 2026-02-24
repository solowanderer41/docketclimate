"""
Instagram Reels Publisher Module for Docket Social
===================================================

Publishes video Reels to Instagram via the Instagram Graph API (HTTP).

The Instagram Graph API uses a two-step publishing flow:
    1. Create a media container (POST to /{ig_user_id}/media) with media_type=REELS
    2. Publish the container (POST to /{ig_user_id}/media_publish)

IMPORTANT: The video must be hosted at a publicly accessible URL. Instagram's
servers will fetch the video from that URL during container creation. You can use
services like AWS S3, Google Cloud Storage, or any CDN to host the video file.

Video requirements for Reels:
    - Format: MP4 (H.264 codec recommended)
    - Aspect ratio: 9:16 (vertical) recommended
    - Duration: 3 seconds to 15 minutes
    - Max file size: 1 GB
    - Resolution: 1080x1920 recommended

Required API Credentials (in .env):
    META_INSTAGRAM_ACCOUNT_ID - Your Instagram Business/Creator account ID (numeric)
    META_ACCESS_TOKEN         - A valid Meta access token with instagram_content_publish scope

How to obtain credentials:
    1. Create a Meta Developer account at https://developers.facebook.com/
    2. Create an App and add the "Instagram Graph API" product
    3. Link your Instagram Business or Creator account to a Facebook Page
    4. Generate a User Access Token with these permissions:
       - instagram_basic
       - instagram_content_publish
       - pages_read_engagement
    5. Your Instagram account ID can be found via:
       GET /me/accounts -> page_id -> GET /{page_id}?fields=instagram_business_account
    6. For production, exchange the short-lived token for a long-lived token
       (valid for 60 days) via the token exchange endpoint
"""

import os
import time

import requests
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

BASE_URL = "https://graph.instagram.com/v21.0"
PUBLISH_STATUS_CHECK_INTERVAL = 5  # seconds between status checks
PUBLISH_TIMEOUT = 300  # max seconds to wait (video processing can be slow)


class ReelsPublisher:
    """Client for publishing Reels to Instagram via the Graph API."""

    def __init__(self):
        self.ig_user_id = os.getenv("META_INSTAGRAM_ACCOUNT_ID")
        self.access_token = os.getenv("META_INSTAGRAM_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")

        missing = []
        if not self.ig_user_id:
            missing.append("META_INSTAGRAM_ACCOUNT_ID")
        if not self.access_token:
            missing.append("META_ACCESS_TOKEN")

        if missing:
            console.print(
                f"[bold red]Missing Instagram credentials in .env: "
                f"{', '.join(missing)}[/bold red]"
            )

    def login(self):
        """
        Validate the access token by fetching the Instagram account info.

        The Graph API does not have a separate login step; this verifies
        that the stored token and account ID are valid.
        """
        try:
            response = requests.get(
                f"{BASE_URL}/{self.ig_user_id}",
                params={
                    "fields": "id,username",
                    "access_token": self.access_token,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            console.print(
                f"[bold green]Instagram token valid. "
                f"Account: {data.get('username', data.get('id'))}[/bold green]"
            )
            return data
        except requests.RequestException as e:
            console.print(
                f"[bold red]Instagram token validation failed: {e}[/bold red]"
            )
            raise

    def _create_container(
        self, video_url: str, caption: str, cover_url: str | None = None,
    ) -> str:
        """
        Create a Reels media container.

        The video_url must be a publicly accessible URL pointing to an MP4 file.
        Instagram's servers will download the video from this URL.

        Args:
            video_url: Public URL to the video file.
            caption: Caption text for the Reel.
            cover_url: Optional public URL to a custom cover/thumbnail image.
                       If provided, Instagram uses this as the Reel's cover
                       instead of auto-selecting a frame.

        Returns:
            The container/creation ID.
        """
        url = f"{BASE_URL}/{self.ig_user_id}/media"
        payload = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": self.access_token,
        }
        if cover_url:
            payload["cover_url"] = cover_url

        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        container_id = data["id"]
        console.print(f"[dim]Reels container created: {container_id}[/dim]")
        return container_id

    def _wait_for_container_ready(self, container_id: str) -> bool:
        """
        Poll the container status until it is ready for publishing.

        Instagram needs time to process the uploaded video. This method
        polls the container status endpoint until it reports FINISHED
        or a timeout is reached.

        Args:
            container_id: The media container ID to check.

        Returns:
            True if the container is ready, False if it timed out or errored.
        """
        url = f"{BASE_URL}/{container_id}"
        start_time = time.time()

        while time.time() - start_time < PUBLISH_TIMEOUT:
            try:
                response = requests.get(
                    url,
                    params={
                        "fields": "status_code,status",
                        "access_token": self.access_token,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                status_code = data.get("status_code", "")

                if status_code == "FINISHED":
                    console.print("[dim]Reels container ready for publishing.[/dim]")
                    return True
                elif status_code == "ERROR":
                    error_msg = data.get("status", "Unknown error")
                    console.print(
                        f"[bold red]Reels container processing failed: "
                        f"{error_msg}[/bold red]"
                    )
                    return False
                else:
                    elapsed = int(time.time() - start_time)
                    console.print(
                        f"[dim]Reels processing... status={status_code} "
                        f"({elapsed}s elapsed)[/dim]"
                    )

            except requests.RequestException as e:
                console.print(
                    f"[bold yellow]Status check error (will retry): {e}[/bold yellow]"
                )

            time.sleep(PUBLISH_STATUS_CHECK_INTERVAL)

        console.print(
            f"[bold red]Reels container processing timed out after "
            f"{PUBLISH_TIMEOUT}s.[/bold red]"
        )
        return False

    def _publish_container(self, container_id: str) -> str:
        """
        Publish a ready media container.

        Args:
            container_id: The ID returned from the container creation step.

        Returns:
            The published media ID.
        """
        url = f"{BASE_URL}/{self.ig_user_id}/media_publish"
        payload = {
            "creation_id": container_id,
            "access_token": self.access_token,
        }

        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["id"]

    def publish_video(
        self, video_url: str, caption: str, cover_url: str | None = None,
    ) -> dict:
        """
        Publish a video as an Instagram Reel.

        NOTE: The video must be hosted at a publicly accessible URL. Instagram
        fetches the video from that URL during processing. Local file paths
        are not supported. Use a service like AWS S3, GCS, or a CDN.

        Args:
            video_url: Public URL to the video file (MP4, H.264 recommended).
            caption:   Caption text for the Reel.
            cover_url: Optional public URL to a custom thumbnail/cover image.

        Returns:
            dict with 'media_id' of the published Reel.
        """
        try:
            # Step 1: Create the media container
            container_id = self._create_container(video_url, caption, cover_url=cover_url)

            # Step 2: Wait for video processing to complete
            if not self._wait_for_container_ready(container_id):
                raise RuntimeError(
                    "Reels container did not become ready for publishing."
                )

            # Step 3: Publish the container
            media_id = self._publish_container(container_id)
            result = {"media_id": media_id}
            console.print(
                f"[bold green]Published Instagram Reel:[/bold green] {media_id}"
            )
            return result

        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            console.print(
                f"[bold red]Instagram API error: {e} {error_detail}[/bold red]"
            )
            raise
        except Exception as e:
            console.print(
                f"[bold red]Failed to publish Instagram Reel: {e}[/bold red]"
            )
            raise

    def delete_media(self, media_id: str) -> bool:
        """
        Delete an Instagram media object (Reel, post, story) via the Graph API.

        Args:
            media_id: The Instagram media ID to delete.
                      Accepts raw ID ("18071476904405728") or
                      prefixed format ("reels:18071476904405728").

        Returns:
            True if deletion was successful.
        """
        # Strip "reels:" prefix if present
        if media_id.startswith("reels:"):
            media_id = media_id[len("reels:"):]

        url = f"{BASE_URL}/{media_id}"
        params = {"access_token": self.access_token}

        try:
            response = requests.delete(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                console.print(
                    f"[bold green]Deleted Instagram media: {media_id}[/bold green]"
                )
                return True
            else:
                console.print(
                    f"[bold red]Delete response was not success: {data}[/bold red]"
                )
                return False
        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            console.print(
                f"[bold red]Failed to delete media {media_id}: "
                f"{e} {error_detail}[/bold red]"
            )
            raise

    def publish(self, post: dict) -> dict:
        """
        Publish a post using a standardized post dict.

        Args:
            post: dict with 'video_url' and optionally 'text' (used as caption).

        Returns:
            dict with media ID.
        """
        video_url = post.get("video_url", "")
        caption = post.get("text", post.get("caption", ""))
        cover_url = post.get("cover_url")
        if not video_url:
            raise ValueError(
                "post dict must include 'video_url' for Instagram Reels publishing."
            )
        return self.publish_video(video_url, caption, cover_url=cover_url)


def publish(post: dict) -> dict:
    """
    Module-level convenience function to publish a Reel to Instagram.

    Args:
        post: dict with 'video_url' and optionally 'text' (used as caption).

    Returns:
        dict with media ID.
    """
    publisher = ReelsPublisher()
    return publisher.publish(post)


def delete_media(media_id: str) -> bool:
    """
    Module-level convenience function to delete an Instagram media object.

    Args:
        media_id: The Instagram media ID (raw or "reels:"-prefixed).

    Returns:
        True if deletion was successful.
    """
    publisher = ReelsPublisher()
    return publisher.delete_media(media_id)


def test_connection() -> bool:
    """
    Verify that Instagram credentials are valid by checking the access token.

    Returns:
        True if the token and account ID are valid, False otherwise.
    """
    try:
        publisher = ReelsPublisher()
        publisher.login()
        console.print(
            "[bold green]Instagram Reels connection test passed.[/bold green]"
        )
        return True
    except Exception as e:
        console.print(
            f"[bold red]Instagram Reels connection test failed: {e}[/bold red]"
        )
        return False
