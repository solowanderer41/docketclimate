"""
Twitter/X Publisher Module for Docket Social
=============================================

Publishes tweets via the Twitter API v2 using OAuth 1.0a User Context.

Required API Credentials (in .env):
    TWITTER_API_KEY             - Consumer / API Key
    TWITTER_API_SECRET          - Consumer / API Secret
    TWITTER_ACCESS_TOKEN        - OAuth 1.0a Access Token
    TWITTER_ACCESS_TOKEN_SECRET - OAuth 1.0a Access Token Secret

How to obtain credentials:
    1. Go to https://developer.twitter.com/en/portal/dashboard
    2. Create a Project and an App (or use an existing one)
    3. In your App settings, go to "Keys and tokens"
    4. Generate or regenerate your API Key & Secret (Consumer Keys)
    5. Generate or regenerate your Access Token & Secret
    6. Ensure your App has "Read and Write" permissions
    7. Your developer account must have at least Basic access tier for
       posting via API v2
"""

import os
import time

import tweepy
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# Twitter API v2 rate limits for tweet creation (per 15-min window)
RATE_LIMIT_WINDOW = 15 * 60  # 15 minutes in seconds
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds


class TwitterPublisher:
    """Client for publishing tweets via the Twitter API v2."""

    def __init__(self):
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        self.client = None

        missing = []
        if not self.api_key:
            missing.append("TWITTER_API_KEY")
        if not self.api_secret:
            missing.append("TWITTER_API_SECRET")
        if not self.access_token:
            missing.append("TWITTER_ACCESS_TOKEN")
        if not self.access_token_secret:
            missing.append("TWITTER_ACCESS_TOKEN_SECRET")

        if missing:
            console.print(
                f"[bold red]Missing Twitter credentials in .env: "
                f"{', '.join(missing)}[/bold red]"
            )

    def login(self):
        """Initialize the Tweepy Client with OAuth 1.0a credentials."""
        try:
            self.client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True,
            )
            console.print("[bold green]Twitter client initialized.[/bold green]")
        except Exception as e:
            console.print(
                f"[bold red]Failed to initialize Twitter client: {e}[/bold red]"
            )
            raise

    def _ensure_client(self):
        """Initialize the client if not already done."""
        if self.client is None:
            self.login()

    def publish_text(self, text: str) -> dict:
        """
        Publish a text tweet.

        Handles rate limiting with exponential back-off and retries.

        Args:
            text: Tweet content (max 280 characters).

        Returns:
            dict with 'tweet_id' of the created tweet.
        """
        self._ensure_client()

        if len(text) > 280:
            console.print(
                "[bold yellow]Warning: Tweet exceeds 280 characters. "
                "Text will be truncated.[/bold yellow]"
            )
            text = text[:280]

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.create_tweet(text=text)
                tweet_id = response.data["id"]
                result = {"tweet_id": tweet_id}
                console.print(
                    f"[bold green]Published tweet:[/bold green] "
                    f"https://twitter.com/i/status/{tweet_id}"
                )
                return result

            except tweepy.TooManyRequests as e:
                wait_time = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                console.print(
                    f"[bold yellow]Rate limited. Waiting {wait_time}s before "
                    f"retry {attempt}/{MAX_RETRIES}...[/bold yellow]"
                )
                time.sleep(wait_time)
                last_error = e

            except tweepy.TweepyException as e:
                console.print(
                    f"[bold red]Twitter API error on attempt "
                    f"{attempt}/{MAX_RETRIES}: {e}[/bold red]"
                )
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY)

        console.print(
            f"[bold red]Failed to publish tweet after {MAX_RETRIES} attempts.[/bold red]"
        )
        raise last_error

    def publish_with_reply(self, main_text: str, reply_text: str) -> dict:
        """
        Publish a tweet then immediately reply with a second tweet.

        Used to keep external links out of the main tweet (links penalize
        reach 30-50% on X). The hook goes in the main tweet; the article
        link goes in the reply.

        Args:
            main_text: The primary tweet (hook text, no link). Max 280 chars.
            reply_text: The reply tweet (article link + optional context).

        Returns:
            dict with 'tweet_id' (main) and 'reply_id'.
        """
        result = self.publish_text(main_text)
        tweet_id = result["tweet_id"]

        # Post reply with the link
        self._ensure_client()
        try:
            reply_response = self.client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=tweet_id,
            )
            reply_id = reply_response.data["id"]
            result["reply_id"] = reply_id
            console.print(
                f"[green]Reply with link posted:[/green] "
                f"https://twitter.com/i/status/{reply_id}"
            )
        except Exception as e:
            console.print(
                f"[yellow]Reply tweet failed ({e}), "
                f"main tweet is still live.[/yellow]"
            )

        return result

    def publish(self, post: dict) -> dict:
        """
        Publish a post using a standardized post dict.

        Args:
            post: dict with at least a 'text' key.

        Returns:
            dict with tweet ID.
        """
        text = getattr(post, "text", None) or post.get("text", "")
        return self.publish_text(text)


def publish(post: dict) -> dict:
    """
    Module-level convenience function to publish a tweet.

    Args:
        post: dict with at least a 'text' key.

    Returns:
        dict with tweet ID.
    """
    publisher = TwitterPublisher()
    return publisher.publish(post)


def test_connection() -> bool:
    """
    Verify that Twitter credentials are valid by fetching the authenticated user.

    Returns:
        True if authentication succeeds, False otherwise.
    """
    try:
        publisher = TwitterPublisher()
        publisher.login()
        me = publisher.client.get_me()
        if me and me.data:
            console.print(
                f"[bold green]Twitter connection test passed. "
                f"Authenticated as @{me.data.username}[/bold green]"
            )
            return True
        else:
            console.print(
                "[bold red]Twitter connection test failed: "
                "Could not retrieve user info.[/bold red]"
            )
            return False
    except Exception as e:
        console.print(f"[bold red]Twitter connection test failed: {e}[/bold red]")
        return False
