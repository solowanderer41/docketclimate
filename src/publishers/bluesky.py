"""
Bluesky Publisher Module for Docket Social
===========================================

Publishes text posts to Bluesky via the AT Protocol.

Required API Credentials (in .env):
    BLUESKY_HANDLE      - Your Bluesky handle (e.g. yourname.bsky.social)
    BLUESKY_APP_PASSWORD - An App Password generated from Bluesky Settings > App Passwords

How to obtain credentials:
    1. Go to https://bsky.app/settings/app-passwords
    2. Click "Add App Password"
    3. Name it (e.g. "docket-social") and copy the generated password
    4. Your handle is your username (e.g. yourname.bsky.social)
"""

import io
import os
import re
from datetime import datetime, timezone

import requests
from atproto import Client, models
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
from rich.console import Console

load_dotenv()
console = Console()


class BlueskyPublisher:
    """Client for publishing posts to Bluesky via the AT Protocol."""

    def __init__(self):
        self.handle = os.getenv("BLUESKY_HANDLE")
        self.app_password = os.getenv("BLUESKY_APP_PASSWORD")
        self.client = Client()
        self._logged_in = False

        if not self.handle or not self.app_password:
            console.print(
                "[bold red]Missing BLUESKY_HANDLE or BLUESKY_APP_PASSWORD in .env[/bold red]"
            )

    def login(self):
        """Authenticate with the Bluesky API using handle and app password."""
        try:
            self.client.login(self.handle, self.app_password)
            self._logged_in = True
            console.print(
                f"[bold green]Logged in to Bluesky as {self.handle}[/bold green]"
            )
        except Exception as e:
            self._logged_in = False
            console.print(f"[bold red]Bluesky login failed: {e}[/bold red]")
            raise

    def _ensure_logged_in(self):
        """Log in if not already authenticated."""
        if not self._logged_in:
            self.login()

    def _parse_facets(self, text: str) -> list:
        """
        Parse text for links and hashtags and return AT Protocol facets.

        Facets annotate byte ranges of the text with rich features like links
        and hashtags so they render properly in Bluesky clients.
        """
        facets = []
        text_bytes = text.encode("utf-8")

        # Detect URLs
        url_pattern = re.compile(
            r"https?://[^\s\)\]\}>\"']+"
        )
        for match in url_pattern.finditer(text):
            url = match.group(0)
            byte_start = len(text[: match.start()].encode("utf-8"))
            byte_end = byte_start + len(url.encode("utf-8"))
            facets.append(
                {
                    "index": {"byteStart": byte_start, "byteEnd": byte_end},
                    "features": [
                        {
                            "$type": "app.bsky.richtext.facet#link",
                            "uri": url,
                        }
                    ],
                }
            )

        # Detect hashtags
        hashtag_pattern = re.compile(r"(?:^|\s)(#\w+)", re.UNICODE)
        for match in hashtag_pattern.finditer(text):
            tag = match.group(1)  # Just the #hashtag, without leading whitespace
            byte_start = len(text[: match.start(1)].encode("utf-8"))
            byte_end = byte_start + len(tag.encode("utf-8"))
            facets.append(
                {
                    "index": {"byteStart": byte_start, "byteEnd": byte_end},
                    "features": [
                        {
                            "$type": "app.bsky.richtext.facet#tag",
                            "tag": match.group(1),
                        }
                    ],
                }
            )

        return facets if facets else None

    def _fetch_link_card(self, url: str):
        """
        Fetch Open Graph metadata from a URL and return an external embed.

        Fetches the page, extracts og:title, og:description, og:image,
        uploads the thumbnail to Bluesky, and returns an AppBskyEmbedExternal.Main.
        """
        try:
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "Docket-Social/1.0 (link-card preview)"
            })
            resp.raise_for_status()
        except Exception as e:
            console.print(f"[dim]Could not fetch link card for {url}: {e}[/dim]")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract Open Graph metadata
        og_title = ""
        og_desc = ""
        og_image = ""

        tag = soup.find("meta", property="og:title")
        if tag:
            og_title = tag.get("content", "")
        if not og_title:
            title_tag = soup.find("title")
            og_title = title_tag.string if title_tag else url

        tag = soup.find("meta", property="og:description")
        if tag:
            og_desc = tag.get("content", "")
        if not og_desc:
            tag = soup.find("meta", attrs={"name": "description"})
            if tag:
                og_desc = tag.get("content", "")

        tag = soup.find("meta", property="og:image")
        if tag:
            og_image = tag.get("content", "")

        # Upload thumbnail if available (Bluesky max: 976KB)
        thumb_blob = None
        if og_image:
            try:
                img_resp = requests.get(og_image, timeout=10)
                img_resp.raise_for_status()
                img_data = img_resp.content

                # Resize/compress to fit under 976KB
                img = Image.open(io.BytesIO(img_data))
                img = img.convert("RGB")  # ensure JPEG-compatible mode

                # Scale down if very large
                max_dim = 1200
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.LANCZOS)

                # Compress to JPEG under 950KB
                quality = 85
                while quality >= 30:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=quality)
                    if buf.tell() <= 950_000:
                        break
                    quality -= 10

                img_bytes = buf.getvalue()
                thumb_blob = self.client.upload_blob(img_bytes).blob
            except Exception as e:
                console.print(f"[dim]Could not upload thumbnail: {e}[/dim]")

        external = models.AppBskyEmbedExternal.External(
            uri=url,
            title=og_title or "",
            description=og_desc or "",
            thumb=thumb_blob,
        )
        return models.AppBskyEmbedExternal.Main(external=external)

    def _extract_first_url(self, text: str) -> str | None:
        """Extract the first URL from post text."""
        match = re.search(r"https?://[^\s\)\]\}>\"']+", text)
        return match.group(0) if match else None

    def publish_text(self, text: str) -> dict:
        """
        Publish a text post to Bluesky.

        Args:
            text: The post content (max 300 characters).

        Returns:
            dict with 'uri' and 'cid' of the created post.
        """
        self._ensure_logged_in()

        if len(text) > 300:
            console.print(
                "[bold yellow]Warning: Bluesky posts are limited to 300 characters. "
                "Text will be truncated.[/bold yellow]"
            )
            text = text[:300]

        facets = self._parse_facets(text)

        # Build link card embed from first URL in post
        embed = None
        first_url = self._extract_first_url(text)
        if first_url:
            embed = self._fetch_link_card(first_url)

        try:
            response = self.client.send_post(text=text, facets=facets, embed=embed)
            result = {"uri": response.uri, "cid": response.cid}
            console.print(
                f"[bold green]Published to Bluesky:[/bold green] {response.uri}"
            )
            return result
        except Exception as e:
            console.print(f"[bold red]Failed to publish to Bluesky: {e}[/bold red]")
            raise

    def publish(self, post) -> dict:
        """
        Publish a post using a standardized post object or dict.

        Args:
            post: object with a 'text' attribute, or dict with a 'text' key.

        Returns:
            dict with post URI and CID.
        """
        text = getattr(post, "text", None) or post.get("text", "")
        return self.publish_text(text)

    def test_connection(self) -> bool:
        """Verify credentials are valid."""
        self._ensure_logged_in()
        console.print("[bold green]Bluesky connection test passed.[/bold green]")
        return True


def publish(post: dict) -> dict:
    """
    Module-level convenience function to publish a post to Bluesky.

    Args:
        post: dict with at least a 'text' key.

    Returns:
        dict with post URI and CID.
    """
    publisher = BlueskyPublisher()
    return publisher.publish(post)


def test_connection() -> bool:
    """
    Verify that Bluesky credentials are valid by attempting to log in.

    Returns:
        True if login succeeds, False otherwise.
    """
    try:
        publisher = BlueskyPublisher()
        publisher.login()
        console.print("[bold green]Bluesky connection test passed.[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Bluesky connection test failed: {e}[/bold red]")
        return False
