"""Safe ChatGPT diagnostics and bounded retries; never retry tool execution."""

import asyncio
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
import math
import random
import re
import sys
import time
from typing import TypeVar

import httpx
from kosong.chat_provider import ChatProviderError
from openai import APIConnectionError, OpenAIError, APIStatusError

REFRESH_REJECTED = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}
TRANSIENT = {
    "server_error",
    "internal_error",
    "rate_limit_exceeded",
    "stream_interrupted",
    "connection_error",
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
}
TERMINAL = REFRESH_REJECTED | {
    "invalid_client",
    "invalid_token",
    "invalid_api_key",
    "model_not_found",
    "invalid_request_error",
    "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_user_not_eligible",
    "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user",
    "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
}


def _identifier(value: object) -> str | None:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[\w.:-]{1,128}", value)
        else None
    )


class ChatGPTError(ChatProviderError, ValueError):
    def __init__(
        self,
        operation: str,
        *,
        code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        transport_type: str | None = None,
    ):
        self.operation = operation
        self.code = _identifier(code)
        self.status = status
        self.request_id = _identifier(request_id)
        self.retry_after = retry_after
        self.transport_type = _identifier(transport_type)
        self.retryable = self.code not in TERMINAL and (
            self.code in TRANSIENT
            or status in (408, 429)
            or (status is not None and status >= 500)
        )
        detail = f"HTTP {status}" if status is not None else "request failed"
        if self.code:
            detail += f", {self.code}"
        if self.transport_type:
            detail += f", {self.transport_type}"
        if self.request_id:
            detail += f", request {self.request_id}"
        hint = ""
        if operation == "token refresh" and self.code in REFRESH_REJECTED:
            hint = "; run `lazarus auth login` again"
        elif self.code == "subscription_sharing_usage_limit_exceeded":
            hint = "; check https://chatgpt.com/settings/usage"
        elif self.code == "invalid_client":
            hint = "; check the saved client registration"
        elif self.code == "refresh_outcome_unknown":
            hint = "; refresh may have reached the server; saved credentials kept"
        super().__init__(f"ChatGPT {operation} failed ({detail}){hint}.")


def response_error(operation: str, response: httpx.Response) -> ChatGPTError:
    try:
        body = response.json()
    except ValueError:
        body = None
    code = None
    if isinstance(body, dict):
        error = body.get("error", body)
        code = error.get("code") if isinstance(error, dict) else error
    retry_after = None
    header = response.headers.get("retry-after")
    if header:
        try:
            retry_after = float(header)
        except ValueError:
            try:
                retry_after = parsedate_to_datetime(header).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                pass
        if retry_after is not None and not math.isfinite(retry_after):
            retry_after = None
    return ChatGPTError(
        operation,
        code=code,
        status=response.status_code,
        request_id=response.headers.get("x-request-id"),
        retry_after=retry_after,
    )


def transport_error(
    operation: str, error: OpenAIError | httpx.HTTPError
) -> ChatGPTError:
    if isinstance(error, (APIStatusError, httpx.HTTPStatusError)):
        return response_error(operation, error.response)
    if isinstance(error, (APIConnectionError, httpx.TransportError)):
        cause = error.__cause__ if isinstance(error, APIConnectionError) else error
        code = "connection_error"
        # A lost refresh response may contain the only valid replacement token.
        # Only retry transport failures known to happen before the request is sent.
        if operation == "token refresh" and not isinstance(
            cause, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
        ):
            code = "refresh_outcome_unknown"
        return ChatGPTError(
            operation, code=code, transport_type=type(cause or error).__name__
        )
    body = getattr(error, "body", None)
    code = body.get("code") if isinstance(body, dict) else None
    return ChatGPTError(operation, code=code)


def retry_delay(error: ChatGPTError, attempt: int) -> float | None:
    # Two retries total. Never shorten a server-directed wait to our cap.
    if not error.retryable or attempt >= 2 or (error.retry_after or 0) > 60:
        return None
    delay = max(error.retry_after or 0, 2**attempt + random.uniform(0, 0.25))
    print(f"{error} Retrying {attempt + 1}/2 in {delay:.1f}s.", file=sys.stderr)
    return delay


T = TypeVar("T")


async def retry_chatgpt(operation: Callable[[], Awaitable[T]]) -> T:
    for attempt in range(3):
        try:
            return await operation()
        except ChatGPTError as error:
            delay = retry_delay(error, attempt)
            if delay is None:
                # An enclosing operation must not retry it again.
                error.retryable = False
                raise
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
