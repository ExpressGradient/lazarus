"""Responses adapter for the official ChatGPT plan OAuth flow."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast
from uuid import uuid4

import httpx
from openai import AsyncStream, OpenAIError
from openai.types.responses import ResponseInputParam, ResponseStreamEvent, ToolParam
from kosong.chat_provider import ChatProviderError
from kosong.contrib.chat_provider.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesStreamedMessage,
    _convert_tool,
)
from kosong.message import Message
from kosong.tooling import Tool

from lazarus.chatgpt_auth import ChatGPTAuth, RESOURCE
from lazarus.chatgpt_errors import (
    ChatGPTError,
    response_error,
    retry_chatgpt,
    transport_error,
)


class ChatGPT(OpenAIResponses):
    name = "chatgpt"

    def __init__(self, *, model: str = "", account: str = "default"):
        self.auth = ChatGPTAuth(account)
        # ChatGPT uses session-id for cache affinity across requests.
        self._session_id = str(uuid4())
        super().__init__(
            model=model, api_key="oauth", base_url=RESOURCE, max_retries=0, stream=True
        )

    async def models(self) -> list[dict]:
        async def request():
            token = await asyncio.to_thread(self.auth.access_token)
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(
                        f"{RESOURCE}/models",
                        headers={"Authorization": f"Bearer {token}"},
                    )
            except httpx.HTTPError as error:
                raise transport_error("model discovery", error) from error
            if response.is_error:
                raise response_error("model discovery", response)
            return [
                m for m in response.json()["models"] if m.get("visibility") == "list"
            ]

        return await retry_chatgpt(request)

    async def prepare(self) -> None:
        if not self._model:
            models = await self.models()
            if not models:
                raise ChatProviderError("No models available for this ChatGPT account.")
            self._model = models[0]["slug"]

    async def close(self) -> None:
        await self._client.close()

    async def generate(
        self, system_prompt: str, tools: Sequence[Tool], history: Sequence[Message]
    ):
        await self.prepare()
        token = await asyncio.to_thread(self.auth.access_token)
        inputs = [
            dict(item) for message in history for item in self._convert_message(message)
        ]
        for item in inputs:
            if item.get("role") == "system":
                item["role"] = "developer"
            if item.get("type") == "function_call":
                item["namespace"] = "lazarus"
        kwargs: dict[str, Any] = {}
        effort = self._generation_kwargs.get("reasoning_effort")
        if effort:
            kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=cast(ResponseInputParam, inputs),
                # ChatGPT namespaces are not yet represented in the SDK types.
                tools=cast(
                    list[ToolParam],
                    [
                        {
                            "type": "namespace",
                            "name": "lazarus",
                            "description": "Lazarus local tools",
                            "tools": [_convert_tool(t) for t in tools],
                        }
                    ]
                    if tools
                    else [],
                ),
                stream=True,
                store=False,
                include=["reasoning.encrypted_content"],
                extra_headers={
                    "Authorization": f"Bearer {token}",
                    "session-id": self._session_id,
                },
                **kwargs,
            )
            return ChatGPTStream(response)
        except (OpenAIError, httpx.HTTPError) as exc:
            raise transport_error("inference", exc) from exc


class ChatGPTStream(OpenAIResponsesStreamedMessage):
    async def _convert_stream_response(self, response):
        request_id = response.response.headers.get("x-request-id")

        async def checked():
            completed = False
            try:
                async for event in response:
                    if event.type in (
                        "response.failed",
                        "response.incomplete",
                        "error",
                    ):
                        result = getattr(event, "response", None)
                        error = getattr(result, "error", None)
                        details = getattr(result, "incomplete_details", None)
                        code = (
                            getattr(error, "code", None)
                            or getattr(event, "code", None)
                            or getattr(details, "reason", None)
                            or event.type
                        )
                        raise ChatGPTError(
                            "inference", code=code, request_id=request_id
                        )
                    if (
                        event.type == "response.output_item.added"
                        and event.item.type == "function_call"
                    ):
                        if getattr(event.item, "namespace", None) not in (
                            None,
                            "lazarus",
                        ):
                            raise ChatProviderError(
                                "ChatGPT returned an unknown tool namespace."
                            )
                    if event.type == "response.completed":
                        completed = True
                    yield event
                if not completed:
                    raise ChatGPTError(
                        "inference", code="stream_interrupted", request_id=request_id
                    )
            except (OpenAIError, httpx.HTTPError) as exc:
                raise transport_error("inference", exc) from exc

        try:
            # The upstream converter only iterates; it does not need stream methods.
            async for part in super()._convert_stream_response(
                cast(AsyncStream[ResponseStreamEvent], checked())
            ):
                yield part
        finally:
            await response.close()
