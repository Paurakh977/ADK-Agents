"""
GitHub Copilot OAuth Device Flow handler.

Implements the OAuth 2.0 Device Authorization Grant for GitHub Copilot.
This is how the official OpenCode app authenticates with GitHub Copilot.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

# GitHub Copilot uses the VS Code client ID for device flow
GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"


class CopilotOAuthError(Exception):
    pass


class CopilotAuthCancelled(CopilotOAuthError):
    pass


class CopilotAuthTimeout(CopilotOAuthError):
    pass


async def start_device_flow() -> dict[str, Any]:
    """Start the OAuth device flow and return device code info."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_DEVICE_CODE_URL,
            data={
                "client_id": GITHUB_COPILOT_CLIENT_ID,
                "scope": "user:email",
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


async def poll_for_token(
    device_code: str, interval: float = 5.0, timeout: float = 300.0
) -> str:
    """Poll GitHub for the OAuth access token until user authorizes or timeout."""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": GITHUB_COPILOT_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            if "access_token" in data:
                return data["access_token"]

            error = data.get("error", "")
            if error == "authorization_pending":
                await asyncio.sleep(interval)
            elif error == "slow_down":
                interval += 5
                await asyncio.sleep(interval)
            elif error == "expired_token":
                raise CopilotOAuthError("Device code expired. Please try again.")
            elif error == "access_denied":
                raise CopilotAuthCancelled("Authorization was denied by user.")
            else:
                raise CopilotOAuthError(f"Unexpected error: {error}")

    raise CopilotAuthTimeout("Timed out waiting for authorization.")


async def get_copilot_token(github_token: str) -> str:
    """Exchange GitHub token for a Copilot API token."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            GITHUB_COPILOT_TOKEN_URL,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/json",
                "Editor-Version": "opencode/1.0",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("token")
        if not token:
            raise CopilotOAuthError(f"Failed to get Copilot token: {data}")
        return token


async def complete_copilot_auth() -> dict[str, Any]:
    """Run the full device flow and return credentials dict for storage."""
    flow = await start_device_flow()
    device_code = flow["device_code"]
    user_code = flow["user_code"]
    verification_uri = flow["verification_uri"]
    expires_in = flow.get("expires_in", 900)
    interval = flow.get("interval", 5)

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "expires_in": expires_in,
        "interval": interval,
    }


async def finish_copilot_auth(
    device_code: str, interval: float = 5.0
) -> dict[str, Any]:
    """Complete the device flow and return credentials for storage."""
    github_token = await poll_for_token(device_code, interval=interval)
    copilot_token = await get_copilot_token(github_token)
    return {
        "github_token": github_token,
        "copilot_token": copilot_token,
        "auth_type": "oauth",
    }
