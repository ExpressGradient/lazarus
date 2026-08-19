from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAIError
from openai.types.shared.reasoning import Reasoning

from kosong.chat_provider.openai_common import convert_error
from kosong.contrib.chat_provider.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesStreamedMessage,
    _convert_tool,
)
from kosong.message import Message
from kosong.tooling import Tool


# Public OAuth client ID used by the official Codex client; not a secret.
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_DUMMY_KEY = "codex-subscription"


def _claims(token: str) -> Mapping[str, Any]:
    try:
        encoded = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        return claims if isinstance(claims, Mapping) else {}
    except (IndexError, ValueError, binascii.Error, json.JSONDecodeError):
        return {}


class CodexAuth:
    def __init__(self, auth_file: Path | None = None) -> None:
        if auth_file is None:
            codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
            auth_file = codex_home / "auth.json"
        try:
            payload = json.loads(auth_file.read_text())
            tokens = payload["tokens"]
            self._access = tokens["access_token"]
            self._refresh = tokens["refresh_token"]
            self._account_id = tokens.get("account_id") or self._account_from_token(
                tokens.get("id_token", self._access)
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Codex login not found; run `codex login` first.") from exc
        self._lock = asyncio.Lock()

    async def credentials(self) -> tuple[str, str]:
        if self._expires_soon():
            async with self._lock:
                if self._expires_soon():
                    await self._refresh_access()
        if not self._account_id:
            raise ValueError(
                "Codex login has no ChatGPT account ID; run `codex login` again."
            )
        return self._access, self._account_id

    def _expires_soon(self) -> bool:
        expires = _claims(self._access).get("exp")
        return isinstance(expires, (int, float)) and expires <= time.time() + 60

    async def _refresh_access(self) -> None:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh,
                    "client_id": _CLIENT_ID,
                },
            )
            response.raise_for_status()
        tokens = response.json()
        self._access = tokens["access_token"]
        self._refresh = tokens.get("refresh_token", self._refresh)
        self._account_id = self._account_from_token(self._access) or self._account_id

    @staticmethod
    def _account_from_token(token: str) -> str | None:
        claims = _claims(token)
        auth = claims.get("https://api.openai.com/auth")
        if isinstance(auth, Mapping):
            account_id = auth.get("chatgpt_account_id")
            if isinstance(account_id, str):
                return account_id
        return None


class CodexChatGPT(OpenAIResponses):
    """OpenAI Responses provider backed by Codex subscription usage."""

    name = "codex"

    def __init__(self, *, model: str, auth_file: Path | None = None) -> None:
        self._codex_auth = CodexAuth(auth_file)
        self._session_id = str(uuid.uuid4())
        http_client = httpx.AsyncClient(
            event_hooks={"request": [self._prepare_request]},
            timeout=httpx.Timeout(600, connect=30),
        )
        super().__init__(
            model=model,
            api_key=_DUMMY_KEY,
            stream=True,
            http_client=http_client,
        )

    async def _prepare_request(self, request: httpx.Request) -> None:
        access, account_id = await self._codex_auth.credentials()
        for header in tuple(request.headers):
            if header.lower().startswith("x-stainless-"):
                del request.headers[header]
        request.url = httpx.URL(_RESPONSES_URL)
        request.headers["host"] = "chatgpt.com"
        request.headers["authorization"] = f"Bearer {access}"
        request.headers["ChatGPT-Account-Id"] = account_id
        request.headers["accept"] = "text/event-stream"
        request.headers["originator"] = "lazarus"
        request.headers["session-id"] = self._session_id
        request.headers["user-agent"] = "lazarus"

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> OpenAIResponsesStreamedMessage:
        inputs = [
            item for message in history for item in self._convert_message(message)
        ]
        generation_kwargs: dict[str, Any] = dict(self._generation_kwargs)
        reasoning_effort = generation_kwargs.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            generation_kwargs["reasoning"] = Reasoning(
                effort=reasoning_effort, summary="auto"
            )
            generation_kwargs["include"] = ["reasoning.encrypted_content"]
        try:
            response = await self._client.responses.create(
                stream=True,
                model=self._model,
                instructions=system_prompt,
                input=inputs,
                tools=[_convert_tool(tool) for tool in tools],
                store=False,
                **generation_kwargs,
            )
            return OpenAIResponsesStreamedMessage(response)
        except (OpenAIError, httpx.HTTPError) as exc:
            raise convert_error(exc) from exc
