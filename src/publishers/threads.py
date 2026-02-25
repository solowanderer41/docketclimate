"""
Threads Publisher Module for Docket Social
==========================================

Publishes text posts to Meta Threads via the Threads API (HTTP).

The Threads API uses a three-step publishing flow:
    1. Create a media container (POST to /{user_id}/threads)
    2. Poll container status (GET /{container_id}?fields=status) until FINISHED
    3. Publish the container (POST to /{user_id}/threads_publish)

Required API Credentials (in .env):
    META_THREADS_USER_ID - Your Threads/Instagram user ID (numeric)
    META_ACCESS_TOKEN    - A valid Meta access token with threads_publish scope

How to obtain credentials:
    1. Create a Meta Developer account at https://developers.facebook.com/
    2. Create an App and add the "Threads API" product
    3. In the Threads API settings, generate an access token
    4. Your user ID is shown in the API settings or can be retrieved via
       GET /me?fields=id on the Threads API
    5. Ensure the token has threads_basic and threads_content_publish permissions
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

BASE_URL = "https://graph.threads.net/v1.0"
PUBLISH_POLL_INTERVAL = 2  # seconds between status checks
PUBLISH_TIMEOUT = 30  # max seconds to wait for publishing


class ThreadsPublisher:
    """Client for publishing posts to Threads via the Meta Threads API."""

    def __init__(self):
        self.user_id = os.getenv("META_THREADS_USER_ID")
        self.access_token = os.getenv("META_ACCESS_TOKEN")

        missing = []
        if not self.user_id:
            missing.append("META_THREADS_USER_ID")
        if not self.access_token:
            missing.append("META_ACCESS_TOKEN")

        if missing:
            console.print(
                f"[bold red]Missing Threads credentials in .env: "
                f"{', '.join(missing)}[/bold red]"
            )

    def login(self):
        """
        Validate the access token by fetching the user profile.

        The Threads API does not have a separate login step; this verifies
        that the stored token is still valid.
        """
        try:
            response = requests.get(
                f"{BASE_URL}/me",
                params={
                    "fields": "id,username",
                    "access_token": self.access_token,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            console.print(
                f"[bold green]Threads token valid. "
                f"User: {data.get('username', data.get('id'))}[/bold green]"
            )
            return data
        except requests.RequestException as e:
            console.print(f"[bold red]Threads token validation failed: {e}[/bold red]")
            raise

    def _create_container(self, text: str) -> str:
        """
        Create a media container for a text post.

        Args:
            text: The post content (max 500 characters).

        Returns:
            The container/creation ID.
        """
        url = f"{BASE_URL}/{self.user_id}/threads"
        payload = {
            "media_type": "TEXT",
            "text": text,
            "access_token": self.access_token,
        }

        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        container_id = data["id"]
        console.print(
            f"[dim]Threads container created: {container_id}[/dim]"
        )
        return container_id

    def _wait_for_container(self, container_id: str) -> str:
        """
        Poll the container status until it is FINISHED or an error occurs.

        The Threads API requires containers to finish processing before they
        can be published. Skipping this step causes intermittent 400 errors
        ("The requested resource does not exist") on the publish endpoint.

        Args:
            container_id: The ID returned from the container creation step.

        Returns:
            The final status string ("FINISHED").

        Raises:
            RuntimeError: If the container enters an ERROR state.
            TimeoutError: If the container is not ready within PUBLISH_TIMEOUT.
        """
        url = f"{BASE_URL}/{container_id}"
        params = {
            "fields": "status,error_message",
            "access_token": self.access_token,
        }
        deadline = time.time() + PUBLISH_TIMEOUT

        while time.time() < deadline:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status == "FINISHED":
                console.print(f"[dim]Container {container_id} ready.[/dim]")
                return status

            if status == "ERROR":
                error_msg = data.get("error_message", "unknown error")
                raise RuntimeError(
                    f"Threads container {container_id} failed: {error_msg}"
                )

            console.print(
                f"[dim]Container status: {status}, waiting {PUBLISH_POLL_INTERVAL}s...[/dim]"
            )
            time.sleep(PUBLISH_POLL_INTERVAL)

        raise TimeoutError(
            f"Threads container {container_id} not ready after {PUBLISH_TIMEOUT}s"
        )

    def _publish_container(self, container_id: str) -> str:
        """
        Publish a previously created media container.

        Args:
            container_id: The ID returned from the container creation step.

        Returns:
            The published thread/post ID.
        """
        url = f"{BASE_URL}/{self.user_id}/threads_publish"
        payload = {
            "creation_id": container_id,
            "access_token": self.access_token,
        }

        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data["id"]

    def publish_text(self, text: str) -> dict:
        """
        Publish a text post to Threads using the two-step flow.

        Args:
            text: The post content (max 500 characters).

        Returns:
            dict with 'thread_id' of the published post.
        """
        if len(text) > 500:
            console.print(
                "[bold yellow]Warning: Threads posts are limited to 500 characters. "
                "Text will be truncated.[/bold yellow]"
            )
            text = text[:500]

        try:
            container_id = self._create_container(text)
            self._wait_for_container(container_id)
            thread_id = self._publish_container(container_id)
            result = {"thread_id": thread_id}
            console.print(
                f"[bold green]Published to Threads:[/bold green] {thread_id}"
            )
            return result
        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            console.print(
                f"[bold red]Threads API error: {e} {error_detail}[/bold red]"
            )
            raise
        except Exception as e:
            console.print(f"[bold red]Failed to publish to Threads: {e}[/bold red]")
            raise

    def publish(self, post: dict) -> dict:
        """
        Publish a post using a standardized post dict.

        Args:
            post: dict with at least a 'text' key.

        Returns:
            dict with thread ID.
        """
        text = getattr(post, "text", None) or post.get("text", "")
        return self.publish_text(text)


def publish(post: dict) -> dict:
    """
    Module-level convenience function to publish a post to Threads.

    Args:
        post: dict with at least a 'text' key.

    Returns:
        dict with thread ID.
    """
    publisher = ThreadsPublisher()
    return publisher.publish(post)


def test_connection() -> bool:
    """
    Verify that Threads credentials are valid by checking the access token.

    Returns:
        True if the token is valid, False otherwise.
    """
    try:
        publisher = ThreadsPublisher()
        publisher.login()
        console.print("[bold green]Threads connection test passed.[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]Threads connection test failed: {e}[/bold red]")
        return False
