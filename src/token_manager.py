"""
Meta Token Manager for Docket Social
=====================================

Handles long-lived token refresh for Meta APIs (Threads + Instagram).

Meta access tokens have a lifecycle:
    1. Short-lived token (~1 hour) — generated in Meta Developer portal
    2. Long-lived token (~60 days) — exchanged from short-lived token
    3. Refreshed long-lived token (~60 days) — refreshed before expiry

Each Meta platform has its OWN refresh endpoint:
    - Threads: GET https://graph.threads.net/refresh_access_token
              grant_type=th_refresh_token
    - Instagram: GET https://graph.instagram.com/refresh_access_token
                 grant_type=ig_refresh_token

The refresh endpoint extends the token for another 60 days as long as:
    - The token hasn't already expired
    - The token is at least 24 hours old

This module provides:
    - refresh_threads_token()   — refresh a Threads long-lived token
    - refresh_instagram_token() — refresh an Instagram long-lived token
    - refresh_all_meta_tokens() — orchestrate full refresh + .env update
    - check_token_health()      — quick expiry check for status display

Required .env variables:
    META_ACCESS_TOKEN               — Current Threads access token
    META_INSTAGRAM_ACCESS_TOKEN     — Current Instagram access token
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
TOKEN_LOG_DIR = PROJECT_ROOT / "logs"

# Platform-specific API URLs
THREADS_API_URL = "https://graph.threads.net"
INSTAGRAM_API_URL = "https://graph.instagram.com"


def refresh_threads_token(current_token: str) -> dict:
    """
    Refresh a Threads long-lived token for another ~60 days.

    Uses the Threads-specific refresh endpoint:
        GET https://graph.threads.net/refresh_access_token
        ?grant_type=th_refresh_token&access_token=<current_token>

    Args:
        current_token: The current long-lived Threads access token.

    Returns:
        dict with 'access_token' and 'expires_in' (seconds).
    """
    response = requests.get(
        f"{THREADS_API_URL}/refresh_access_token",
        params={
            "grant_type": "th_refresh_token",
            "access_token": current_token,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if "access_token" not in data:
        raise ValueError(f"Threads token refresh failed: {data}")

    return data


def refresh_instagram_token(current_token: str) -> dict:
    """
    Refresh an Instagram long-lived token for another ~60 days.

    Uses the Instagram-specific refresh endpoint:
        GET https://graph.instagram.com/refresh_access_token
        ?grant_type=ig_refresh_token&access_token=<current_token>

    Args:
        current_token: The current long-lived Instagram access token.

    Returns:
        dict with 'access_token', 'expires_in' (seconds), and 'permissions'.
    """
    response = requests.get(
        f"{INSTAGRAM_API_URL}/refresh_access_token",
        params={
            "grant_type": "ig_refresh_token",
            "access_token": current_token,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if "access_token" not in data:
        raise ValueError(f"Instagram token refresh failed: {data}")

    return data


def _update_env_var(var_name: str, new_value: str):
    """
    Update a single variable in the .env file.

    Reads the entire file, replaces the matching line, writes back.
    Preserves comments and formatting.
    """
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env file not found at {ENV_PATH}")

    content = ENV_PATH.read_text()
    pattern = rf"^{re.escape(var_name)}=.*$"

    if re.search(pattern, content, re.MULTILINE):
        new_content = re.sub(
            pattern, f"{var_name}={new_value}", content, flags=re.MULTILINE
        )
    else:
        # Variable doesn't exist yet — append it
        new_content = content.rstrip() + f"\n{var_name}={new_value}\n"

    ENV_PATH.write_text(new_content)


def _log_token_refresh(token_name: str, success: bool, details: str):
    """Log token refresh activity."""
    TOKEN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = TOKEN_LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "token-refresh",
        "token": token_name,
        "success": success,
        "details": details,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def refresh_all_meta_tokens(dry_run: bool = False) -> dict:
    """
    Orchestrate refresh of all Meta tokens (Threads + Instagram).

    For each token:
        1. Check if it exists and is non-empty
        2. Validate the current token via the platform API
        3. Refresh via the platform-specific refresh endpoint
        4. Validate the new token works
        5. Update .env with the new token

    Args:
        dry_run: If True, validate tokens but don't refresh or update .env.

    Returns:
        dict with results for each token.
    """
    results = {}

    tokens_to_refresh = [
        {
            "name": "META_ACCESS_TOKEN",
            "label": "Threads",
            "refresh_fn": refresh_threads_token,
            "validate_url": f"{THREADS_API_URL}/v1.0/me",
            "validate_params": {"fields": "id,username"},
            "user_field": "username",
        },
        {
            "name": "META_INSTAGRAM_ACCESS_TOKEN",
            "label": "Instagram/Reels",
            "refresh_fn": refresh_instagram_token,
            "validate_url": None,  # Uses IG account ID
            "validate_params": {"fields": "id,username"},
            "user_field": "username",
        },
    ]

    for token_info in tokens_to_refresh:
        token_name = token_info["name"]
        label = token_info["label"]
        current_token = os.getenv(token_name, "")

        console.print(f"\n[bold cyan]{'─' * 50}[/bold cyan]")
        console.print(f"[bold]{label}[/bold] ({token_name})")

        if not current_token:
            msg = "No token set — skipping"
            console.print(f"  [dim]{msg}[/dim]")
            results[token_name] = {"status": "skipped", "reason": msg}
            continue

        # Step 1: Validate current token is still working
        validation_ok, username = _validate_token(token_info, current_token)

        if validation_ok:
            console.print(
                f"  [dim]Current token:[/dim] [green]Valid[/green]"
                f"  [dim](user: {username})[/dim]"
            )
        else:
            console.print(f"  [red]Current token is INVALID or EXPIRED[/red]")
            console.print(
                f"  [yellow]You need to generate a new token from the "
                f"Meta Developer portal and update .env manually.[/yellow]"
            )
            results[token_name] = {
                "status": "expired",
                "reason": "Token invalid/expired — manual renewal required",
            }
            _log_token_refresh(token_name, False, "Token expired/invalid")
            continue

        if dry_run:
            console.print(f"  [cyan]DRY RUN — token is valid, would refresh[/cyan]")
            results[token_name] = {"status": "dry_run", "valid": True}
            continue

        # Step 2: Refresh the token
        refresh_fn = token_info["refresh_fn"]
        try:
            console.print(f"  [dim]Refreshing token...[/dim]")
            refresh_result = refresh_fn(current_token)
            new_token = refresh_result["access_token"]
            expires_in = refresh_result.get("expires_in", 0)
            new_days = expires_in // 86400 if expires_in else "unknown"

            console.print(
                f"  [green]✓ New token obtained "
                f"(valid for ~{new_days} days)[/green]"
            )

        except requests.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                pass
            msg = f"Refresh failed: {e} {error_detail}"
            console.print(f"  [red]{msg}[/red]")
            results[token_name] = {"status": "failed", "reason": msg}
            _log_token_refresh(token_name, False, msg)
            continue

        except Exception as e:
            msg = f"Refresh failed: {e}"
            console.print(f"  [red]{msg}[/red]")
            results[token_name] = {"status": "failed", "reason": msg}
            _log_token_refresh(token_name, False, msg)
            continue

        # Step 3: Validate new token works
        new_valid, new_username = _validate_token(token_info, new_token)
        if new_valid:
            console.print(
                f"  [green]✓ New token validated "
                f"(user: {new_username})[/green]"
            )
        else:
            console.print(
                f"  [yellow]Warning: New token validation failed[/yellow]"
            )
            console.print(
                f"  [yellow]Skipping .env update for safety.[/yellow]"
            )
            results[token_name] = {
                "status": "failed",
                "reason": "New token validation failed",
            }
            _log_token_refresh(token_name, False, "New token validation failed")
            continue

        # Step 4: Update .env
        try:
            _update_env_var(token_name, new_token)
            console.print(f"  [green]✓ .env updated with new token[/green]")
            results[token_name] = {
                "status": "refreshed",
                "expires_in_days": new_days,
            }
            _log_token_refresh(
                token_name, True, f"Refreshed — valid for ~{new_days} days"
            )
        except Exception as e:
            msg = f"Failed to update .env: {e}"
            console.print(f"  [red]{msg}[/red]")
            console.print(
                f"  [yellow]New token (copy manually):[/yellow]\n"
                f"  {new_token[:20]}...{new_token[-10:]}"
            )
            results[token_name] = {"status": "env_update_failed", "reason": msg}
            _log_token_refresh(token_name, False, msg)

    return results


def _validate_token(token_info: dict, token: str) -> tuple[bool, str]:
    """
    Validate a token by calling its platform API.

    Returns (is_valid, username_or_id).
    """
    validate_url = token_info.get("validate_url")

    if validate_url:
        # Direct validation (Threads)
        try:
            resp = requests.get(
                validate_url,
                params={**token_info["validate_params"], "access_token": token},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            username = data.get("username", data.get("id", "unknown"))
            return True, username
        except Exception:
            return False, ""
    else:
        # Instagram — validate via account ID
        ig_account_id = os.getenv("META_INSTAGRAM_ACCOUNT_ID")
        if not ig_account_id:
            return False, ""
        try:
            resp = requests.get(
                f"{INSTAGRAM_API_URL}/v24.0/{ig_account_id}",
                params={"fields": "id,username", "access_token": token},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            username = data.get("username", data.get("id", "unknown"))
            return True, username
        except Exception:
            return False, ""


def check_token_health() -> dict:
    """
    Quick health check of all Meta tokens.

    Validates each token against its platform API and reports status.
    Used by the weekly report and token-status command.
    """
    health = {}

    checks = [
        {
            "name": "META_ACCESS_TOKEN",
            "label": "Threads",
            "validate_url": f"{THREADS_API_URL}/v1.0/me",
            "validate_params": {"fields": "id,username"},
        },
        {
            "name": "META_INSTAGRAM_ACCESS_TOKEN",
            "label": "Instagram/Reels",
            "validate_url": None,
            "validate_params": {"fields": "id,username"},
        },
    ]

    for check in checks:
        token_name = check["name"]
        label = check["label"]
        token = os.getenv(token_name, "")

        if not token:
            health[token_name] = {
                "label": label,
                "status": "missing",
                "days_remaining": None,
            }
            continue

        is_valid, username = _validate_token(check, token)

        if is_valid:
            # Try a refresh (dry) to get expiry info
            try:
                if token_name == "META_ACCESS_TOKEN":
                    # Threads token — peek at expiry via refresh response
                    # (We don't actually use the new token)
                    resp = requests.get(
                        f"{THREADS_API_URL}/refresh_access_token",
                        params={
                            "grant_type": "th_refresh_token",
                            "access_token": token,
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    expires_in = resp.json().get("expires_in", 0)
                else:
                    resp = requests.get(
                        f"{INSTAGRAM_API_URL}/refresh_access_token",
                        params={
                            "grant_type": "ig_refresh_token",
                            "access_token": token,
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    expires_in = resp.json().get("expires_in", 0)

                days_remaining = expires_in // 86400 if expires_in else None
                health[token_name] = {
                    "label": label,
                    "status": "valid",
                    "days_remaining": days_remaining,
                    "username": username,
                }
            except Exception:
                # Token valid but couldn't get expiry
                health[token_name] = {
                    "label": label,
                    "status": "valid",
                    "days_remaining": None,
                    "username": username,
                }
        else:
            health[token_name] = {
                "label": label,
                "status": "expired",
                "days_remaining": 0,
            }

    return health
