from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlencode, urlsplit

from cryptography.hazmat.primitives.asymmetric import rsa
import httpx
import jwt
from kosong.chat_provider import ChatProviderError
from kosong.message import Message, ToolCall
from kosong.tooling import Tool, ToolOk, ToolResult

from lazarus.chatgpt_auth import (
    ChatGPTAuth,
    ISSUER,
    RESOURCE,
    SCOPES,
    _save,
    verify_identity,
)
from lazarus.chatgpt import ChatGPT
from lazarus.cli import (
    build_parser,
    create_chat_provider,
    run,
    run_request,
    TokenTotals,
)
from lazarus.chatgpt_errors import (
    ChatGPTError,
    response_error,
    retry_chatgpt,
    retry_delay,
)


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(cls.key.public_key()))
        cls.jwk["kid"] = "test"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.auth = ChatGPTAuth(directory=Path(self.temp.name))

    def identity(self, **overrides):
        claims = dict(
            sub="user",
            email="user@example.test",
            iss=ISSUER,
            aud="issued",
            nonce="nonce",
            iat=int(time.time()),
            exp=int(time.time()) + 300,
        )
        claims.update(overrides)
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "test"})

    def tokens(self, **overrides):
        return dict(
            access_token="access",
            refresh_token="rotated",
            token_type="Bearer",
            expires_in=3600,
            scope=SCOPES,
            **overrides,
        )

    def seed(self):
        with self.auth.locked() as data:
            data["accounts"]["default"] = dict(
                client_id="issued",
                subject="user",
                refresh_token="old",
                access_token="expired",
                expires_at=0,
                scope=SCOPES,
            )
            _save(self.auth.path, data)

    def test_identity_validation(self):
        self.assertEqual(
            "user",
            verify_identity(self.identity(), {"keys": [self.jwk]}, "issued", "nonce")[
                "sub"
            ],
        )
        for change in (
            {"nonce": "wrong"},
            {"aud": "wrong"},
            {"iss": "wrong"},
            {"exp": 1},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "validation failed"),
            ):
                verify_identity(
                    self.identity(**change), {"keys": [self.jwk]}, "issued", "nonce"
                )
        wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with patch.object(self, "key", wrong), self.assertRaises(ValueError):
            verify_identity(self.identity(), {"keys": [self.jwk]}, "issued", "nonce")

    def test_refresh_is_serialized_and_survives_restart(self):
        self.seed()

        def refresh(url, data):
            self.assertEqual("old", data["refresh_token"])
            self.assertEqual(RESOURCE, data["resource"])
            time.sleep(0.02)
            return httpx.Response(200, json=self.tokens())

        with patch("lazarus.chatgpt_auth.httpx.Client") as client:
            post = client.return_value.__enter__.return_value.post
            post.side_effect = refresh
            with ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(lambda _: self.auth.access_token(), range(2)))
            self.assertEqual(["access", "access"], values)
            self.assertEqual(1, post.call_count)
        self.assertEqual(
            "access", ChatGPTAuth(directory=self.auth.directory).access_token()
        )
        self.assertEqual(
            "rotated",
            json.loads(self.auth.path.read_text())["accounts"]["default"][
                "refresh_token"
            ],
        )
        self.assertEqual(0o600, self.auth.path.stat().st_mode & 0o777)

    def test_failed_refresh_preserves_credentials_and_redacts_body(self):
        self.seed()
        before = self.auth.path.read_text()
        with patch("lazarus.chatgpt_auth.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value = (
                httpx.Response(400, json={"secret": "DO-NOT-PRINT"})
            )
            with self.assertRaisesRegex(ValueError, "HTTP 400") as error:
                self.auth.access_token()
            self.assertNotIn("DO-NOT-PRINT", str(error.exception))
        self.assertEqual(before, self.auth.path.read_text())

    def test_refresh_retries_transient_failures_and_saves_rotation(self):
        self.seed()
        with (
            patch("lazarus.chatgpt_auth.httpx.Client") as client,
            patch("lazarus.chatgpt_auth.time.sleep") as sleep,
            redirect_stderr(StringIO()),
        ):
            post = client.return_value.__enter__.return_value.post
            post.side_effect = [
                httpx.ConnectError("DO-NOT-PRINT"),
                httpx.Response(500, text="DO-NOT-PRINT"),
                httpx.Response(200, json=self.tokens()),
            ]
            self.assertEqual(self.auth.access_token(), "access")
            self.assertEqual(post.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        self.assertEqual(
            json.loads(self.auth.path.read_text())["accounts"]["default"][
                "refresh_token"
            ],
            "rotated",
        )

    def test_refresh_exhaustion_preserves_credentials_without_outer_retry(self):
        self.seed()
        before = self.auth.path.read_text()
        with (
            patch("lazarus.chatgpt_auth.httpx.Client") as client,
            patch("lazarus.chatgpt_auth.time.sleep"),
            redirect_stderr(StringIO()) as output,
        ):
            post = client.return_value.__enter__.return_value.post
            post.return_value = httpx.Response(
                500, text="DO-NOT-PRINT", headers={"x-request-id": "req-test"}
            )

            async def request():
                return self.auth.access_token()

            with self.assertRaises(ChatGPTError) as raised:
                asyncio.run(retry_chatgpt(request))
            self.assertEqual(post.call_count, 3)
            self.assertIn("req-test", str(raised.exception))
            self.assertNotIn("login", str(raised.exception))
            self.assertNotIn("DO-NOT-PRINT", output.getvalue())
        self.assertEqual(before, self.auth.path.read_text())

    def test_terminal_refresh_rejection_clears_only_tokens(self):
        self.seed()
        with patch("lazarus.chatgpt_auth.httpx.Client") as client:
            post = client.return_value.__enter__.return_value.post
            post.return_value = httpx.Response(400, json={"error": "invalid_grant"})
            with self.assertRaisesRegex(ChatGPTError, "auth login"):
                self.auth.access_token()
            self.assertEqual(post.call_count, 1)
        record = json.loads(self.auth.path.read_text())["accounts"]["default"]
        self.assertNotIn("refresh_token", record)
        self.assertEqual(record["client_id"], "issued")
        self.assertEqual(record["subject"], "user")

    def test_ambiguous_refresh_does_not_retry_or_clear_credentials(self):
        for failure in (
            httpx.ReadTimeout,
            httpx.ReadError,
            httpx.WriteTimeout,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ):
            with self.subTest(failure=failure.__name__):
                self.seed()
                before = self.auth.path.read_bytes()
                with (
                    patch("lazarus.chatgpt_auth.httpx.Client") as client,
                    patch(
                        "lazarus.chatgpt_errors.asyncio.sleep", new_callable=AsyncMock
                    ) as sleep,
                ):
                    post = client.return_value.__enter__.return_value.post
                    post.side_effect = [
                        failure("DO-NOT-PRINT"),
                        httpx.Response(400, json={"error": "invalid_grant"}),
                    ]

                    async def request():
                        return await asyncio.to_thread(self.auth.access_token)

                    with self.assertRaisesRegex(
                        ChatGPTError, "refresh_outcome_unknown"
                    ) as raised:
                        asyncio.run(retry_chatgpt(request))
                    self.assertIn(failure.__name__, str(raised.exception))
                    self.assertNotIn("DO-NOT-PRINT", str(raised.exception))
                    post.assert_called_once()
                    sleep.assert_not_awaited()
                self.assertEqual(before, self.auth.path.read_bytes())

    def test_earliest_refresh_keeps_valid_token_and_blocks_expired_token(self):
        self.seed()
        with self.auth.locked() as data:
            data["accounts"]["default"].update(
                access_token="valid",
                expires_at=time.time() + 30,
                earliest_refresh_at=time.time() + 120,
            )
            _save(self.auth.path, data)
        with patch("lazarus.chatgpt_auth.httpx.Client") as client:
            self.assertEqual(self.auth.access_token(), "valid")
            with self.auth.locked() as data:
                data["accounts"]["default"]["expires_at"] = 0
                _save(self.auth.path, data)
            with self.assertRaisesRegex(ChatGPTError, "refresh_not_ready"):
                self.auth.access_token()
            client.assert_not_called()

    def test_login_callback_pkce_and_registration(self):
        captured = {}
        owner = self

        class Server:
            server_port = 54321

            def __init__(self, address, handler):
                owner.assertEqual(("127.0.0.1", 0), address)
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def handle_request(self):
                handler = object.__new__(self.handler)
                handler.wfile = BytesIO()
                handler.send_response = lambda code: captured.update(status=code)
                handler.send_header = lambda *args: None
                handler.end_headers = lambda: None
                for state in ("wrong", captured["state"][0]):
                    handler.path = "/auth/callback?" + urlencode(
                        dict(state=state, code="code", client_id="issued")
                    )
                    handler.do_GET()
                    owner.assertEqual(
                        400 if state == "wrong" else 200, captured["status"]
                    )

        def browser(url):
            captured.update(parse_qs(urlsplit(url).query))

        def post(url, data):
            import base64
            import hashlib

            challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(data["code_verifier"].encode()).digest()
                )
                .decode()
                .rstrip("=")
            )
            self.assertEqual(captured["code_challenge"][0], challenge)
            self.assertEqual("issued", data["client_id"])
            self.assertEqual(captured["redirect_uri"][0], data["redirect_uri"])
            return httpx.Response(
                200,
                json=self.tokens(id_token=self.identity(nonce=captured["nonce"][0])),
            )

        with (
            patch("lazarus.chatgpt_auth.HTTPServer", Server),
            patch("lazarus.chatgpt_auth.webbrowser.open", browser),
            patch("lazarus.chatgpt_auth.httpx.Client") as client,
            redirect_stdout(StringIO()),
        ):
            http = client.return_value.__enter__.return_value
            http.post.side_effect = post
            http.get.side_effect = [
                httpx.Response(200, json={"jwks_uri": ISSUER + "/jwks"}),
                httpx.Response(200, json={"keys": [self.jwk]}),
            ]
            self.auth.login()
        self.assertEqual(["dynamic_agent_client"], captured["client_id"])
        self.assertEqual(["Lazarus"], captured["agent_name_hint"])
        self.assertEqual("access", self.auth.access_token())

    def test_missing_plan_permission_blocks_inference(self):
        self.seed()
        with self.auth.locked() as data:
            data["accounts"]["default"].update(
                scope="openid email", expires_at=time.time() + 3600
            )
            _save(self.auth.path, data)
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.auth.access_token()

    def test_accounts_remain_separate(self):
        self.seed()
        work = ChatGPTAuth("work", self.auth.directory)
        with work.locked() as data:
            data["accounts"]["work"] = dict(
                client_id="work-client",
                subject="user",
                **self.tokens(),
                expires_at=time.time() + 3600,
            )
            original = data["accounts"]["default"].copy()
            host_id = data["host_id"]
            _save(work.path, data)
        self.assertEqual("access", work.access_token())
        stored = json.loads(work.path.read_text())
        self.assertEqual(original, stored["accounts"]["default"])
        self.assertEqual(host_id, stored["host_id"])

    def test_logout_clears_tokens_but_keeps_registration(self):
        self.seed()
        with (
            patch("lazarus.chatgpt_auth.httpx.Client") as client,
            redirect_stdout(StringIO()) as output,
        ):
            client.return_value.__enter__.return_value.get.side_effect = (
                httpx.ConnectError("offline")
            )
            self.auth.logout()
        self.assertIn("revocation was not confirmed", output.getvalue())
        record = json.loads(self.auth.path.read_text())["accounts"]["default"]
        self.assertEqual({"client_id": "issued", "subject": "user"}, record)
        with self.assertRaisesRegex(ValueError, "signed out"):
            self.auth.access_token()


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli(self):
        args = build_parser().parse_args(["--provider", "chatgpt", "--account", "work"])
        provider = create_chat_provider(args)
        self.assertEqual("work", provider.auth.account)
        self.assertEqual("chatgpt", provider.name)
        await provider._client.close()
        self.assertEqual("login", build_parser().parse_args(["auth", "login"]).action)

    async def test_account_options_before_and_after_auth(self):
        for argv in (
            ["--account", "work", "auth", "login"],
            ["auth", "login", "--account", "work"],
        ):
            self.assertEqual("work", build_parser().parse_args(argv).account)

    async def test_model_catalog_preserves_order_and_filters_hidden_models(self):
        provider = ChatGPT()
        provider.auth = SimpleNamespace(access_token=lambda: "oauth-token")
        self.addAsyncCleanup(provider.close)
        real_client = httpx.AsyncClient

        def handler(request):
            self.assertEqual(RESOURCE + "/models", str(request.url))
            self.assertEqual("Bearer oauth-token", request.headers["authorization"])
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"slug": "hidden", "visibility": "hide"},
                        {"slug": "first", "visibility": "list"},
                        {"slug": "second", "visibility": "list"},
                    ]
                },
            )

        with patch(
            "lazarus.chatgpt.httpx.AsyncClient",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        ):
            await provider.prepare()
        self.assertEqual("first", provider.model_name)

    async def provider(self, events, requests):
        def handler(request):
            requests.append(request)
            if callable(events):
                return events(request, len(requests))
            body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=body
            )

        from openai import AsyncOpenAI

        provider = ChatGPT(model="test")
        await provider._client.close()
        provider._client = AsyncOpenAI(
            api_key="unused",
            base_url=RESOURCE,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        provider.auth = SimpleNamespace(access_token=lambda: "oauth-token")
        self.addAsyncCleanup(provider._client.close)
        return provider

    async def test_tool_roundtrip_and_request_contract(self):
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "name": "python",
                    "namespace": "lazarus",
                    "call_id": "call",
                    "id": "item",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"code":"1+1"}',
            },
            {
                "type": "response.completed",
                "response": {"id": "response", "usage": None},
            },
        ]
        requests = []
        provider = await self.provider(events, requests)
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://wrong.invalid"}):
            stream = await provider.generate(
                "instructions",
                [Tool(name="python", description="run", parameters={"type": "object"})],
                [
                    Message(role="system", content="system guidance"),
                    Message(role="user", content="hello"),
                ],
            )
            parts = [part async for part in stream]
        self.assertIsInstance(parts[0], ToolCall)
        self.assertEqual("python", parts[0].function.name)
        request = requests[0]
        self.assertEqual(RESOURCE + "/responses", str(request.url))
        self.assertEqual("Bearer oauth-token", request.headers["authorization"])
        body = json.loads(request.content)
        self.assertTrue(body["stream"])
        self.assertFalse(body["store"])
        self.assertEqual("developer", body["input"][0]["role"])
        self.assertEqual("namespace", body["tools"][0]["type"])
        call = ToolCall(
            id="call", function=ToolCall.FunctionBody(name="python", arguments="{}")
        )
        stream = await provider.generate(
            "instructions",
            [],
            [
                Message(role="assistant", content=[], tool_calls=[call]),
                Message(role="tool", tool_call_id="call", content="2"),
            ],
        )
        _ = [part async for part in stream]
        body = json.loads(requests[1].content)
        self.assertEqual("lazarus", body["input"][0]["namespace"])
        self.assertEqual("call", body["input"][1]["call_id"])

    async def test_session_id_survives_appends_configuration_and_token_refresh(self):
        events = [
            {
                "type": "response.completed",
                "response": {"id": "response", "usage": None},
            }
        ]
        requests = []
        provider = await self.provider(events, requests)
        history = [Message(role="user", content="hello")]
        stream = await provider.generate("instructions", [], history)
        _ = [part async for part in stream]

        provider = provider.with_generation_kwargs(reasoning_effort="high")
        provider.auth = SimpleNamespace(access_token=lambda: "refreshed-token")
        history.extend(
            [
                Message(role="assistant", content="hello"),
                Message(role="user", content="continue"),
            ]
        )
        stream = await provider.generate("instructions", [], history)
        _ = [part async for part in stream]

        session_id = requests[0].headers["session-id"]
        self.assertTrue(session_id)
        self.assertEqual(session_id, requests[1].headers["session-id"])
        self.assertEqual("Bearer refreshed-token", requests[1].headers["authorization"])

    async def test_separate_providers_have_distinct_session_ids(self):
        events = [
            {
                "type": "response.completed",
                "response": {"id": "response", "usage": None},
            }
        ]
        requests = []
        for _ in range(2):
            provider = await self.provider(events, requests)
            stream = await provider.generate("instructions", [], [])
            _ = [part async for part in stream]

        self.assertNotEqual(
            requests[0].headers["session-id"], requests[1].headers["session-id"]
        )

    async def test_failed_incomplete_and_truncated_streams_raise(self):
        for terminal in (None, "response.failed", "response.incomplete"):
            with self.subTest(terminal=terminal):
                events = [{"type": "response.output_text.delta", "delta": "partial"}]
                if terminal:
                    events.append(
                        {
                            "type": terminal,
                            "response": {
                                "error": {
                                    "code": "subscription_sharing_usage_limit_exceeded"
                                }
                            },
                        }
                    )
                provider = await self.provider(events, [])
                stream = await provider.generate("instructions", [], [])
                with self.assertRaises(ChatProviderError):
                    _ = [part async for part in stream]

    async def test_retry_boundary_discards_partial_tools_and_preserves_history(self):
        completed = {
            "type": "response.completed",
            "response": {"id": "response", "usage": None},
        }
        tool = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "name": "python",
                "namespace": "lazarus",
                "call_id": "call",
                "id": "item",
                "arguments": '{"code":"work"}',
            },
        }

        def sse(events):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join("data: " + json.dumps(e) + "\n\n" for e in events),
            )

        class BrokenStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield ("data: " + json.dumps(tool) + "\n\n").encode()
                raise httpx.ReadError("DO-NOT-PRINT")

            async def aclose(self):
                self.closed = True

        for failure in (
            "http",
            "disconnect",
            "stream_disconnect",
            "truncated",
            "server_event",
        ):
            with self.subTest(failure=failure):
                requests, dispatched = [], []
                broken = BrokenStream()

                def handler(request, attempt):
                    if attempt == 1:
                        if failure == "http":
                            return httpx.Response(
                                503, json={"error": {"code": "server_error"}}
                            )
                        if failure == "disconnect":
                            raise httpx.ReadError("DO-NOT-PRINT")
                        if failure == "stream_disconnect":
                            return httpx.Response(
                                200,
                                headers={"content-type": "text/event-stream"},
                                stream=broken,
                            )
                        events = [
                            tool,
                            {
                                "type": "response.output_text.delta",
                                "delta": "discard me",
                            },
                        ]
                        if failure == "server_event":
                            events.append(
                                {
                                    "type": "response.failed",
                                    "response": {"error": {"code": "server_error"}},
                                }
                            )
                        return sse(events)
                    return sse(
                        [tool, completed]
                        if attempt == 2
                        else [
                            {"type": "response.output_text.delta", "delta": "done"},
                            completed,
                        ]
                    )

                provider = await self.provider(handler, requests)

                def handle(call):
                    dispatched.append(call)
                    return ToolResult(
                        tool_call_id=call.id, return_value=ToolOk(output="result")
                    )

                history = []
                with (
                    patch(
                        "lazarus.chatgpt_errors.asyncio.sleep", new_callable=AsyncMock
                    ) as sleep,
                    redirect_stdout(StringIO()),
                    redirect_stderr(StringIO()),
                ):
                    await run_request(
                        provider,
                        SimpleNamespace(tools=[], handle=handle),
                        SimpleNamespace(cwd="/app"),
                        history,
                        "task",
                        TokenTotals(),
                        200000,
                        system_prompt="fixed",
                    )
                self.assertEqual(len(dispatched), 1)
                self.assertEqual(len(requests), 3)
                self.assertEqual(requests[0].content, requests[1].content)
                self.assertEqual(len({r.headers["session-id"] for r in requests}), 1)
                self.assertNotIn("discard me", str(history))
                sleep.assert_awaited_once()
                if failure == "stream_disconnect":
                    self.assertTrue(broken.closed)

    async def test_retry_limit_terminal_errors_and_cancellation(self):
        for error, attempts in (
            (ChatGPTError("inference", status=500), 3),
            (
                ChatGPTError(
                    "inference",
                    status=429,
                    code="subscription_sharing_usage_limit_exceeded",
                ),
                1,
            ),
            (ChatGPTError("inference", status=401), 1),
            (ChatGPTError("inference", code="max_output_tokens"), 1),
            (asyncio.CancelledError(), 1),
        ):
            with (
                self.subTest(error=str(error)),
                patch("lazarus.chatgpt_errors.asyncio.sleep", new_callable=AsyncMock),
                redirect_stderr(StringIO()),
            ):
                operation = AsyncMock(side_effect=error)
                with self.assertRaises(type(error)):
                    await retry_chatgpt(operation)
                self.assertEqual(operation.await_count, attempts)
        with (
            patch(
                "lazarus.chatgpt_errors.asyncio.sleep", new_callable=AsyncMock
            ) as sleep,
            redirect_stderr(StringIO()),
        ):
            sleep.side_effect = asyncio.CancelledError
            operation = AsyncMock(side_effect=ChatGPTError("inference", status=500))
            with self.assertRaises(asyncio.CancelledError):
                await retry_chatgpt(operation)
            operation.assert_awaited_once()

    def test_retry_after_and_safe_diagnostics(self):
        with redirect_stderr(StringIO()):
            for header, expected in (
                ("10", 10),
                ("120", None),
                ("Thu, 01 Jan 1970 00:00:10 GMT", 10),
            ):
                with patch("lazarus.chatgpt_errors.time.time", return_value=0):
                    error = response_error(
                        "inference",
                        httpx.Response(
                            429,
                            json={"detail": "DO-NOT-PRINT"},
                            headers={"retry-after": header},
                        ),
                    )
                    self.assertEqual(retry_delay(error, 0), expected)
                    self.assertNotIn("DO-NOT-PRINT", str(error))

    async def test_session_id_survives_cli_resume_and_old_journals(self):
        events = [
            {"type": "response.output_text.delta", "delta": "done"},
            {
                "type": "response.completed",
                "response": {"id": "response", "usage": None},
            },
        ]
        requests = []
        with (
            tempfile.TemporaryDirectory() as directory,
            redirect_stdout(StringIO()),
            patch("lazarus.cli._system_prompt", return_value="fixed"),
        ):
            path = str(Path(directory) / "session")
            provider = await self.provider(events, requests)
            await run(provider, "first", 200000, 48, session_dir=path)
            provider = await self.provider(events, requests)
            await run(provider, "second", 200000, 48, resume=path)
            self.assertEqual(
                requests[0].headers["session-id"], requests[1].headers["session-id"]
            )
            journal = Path(path) / "journal.jsonl"
            journal.write_text(
                "\n".join(
                    line
                    for line in journal.read_text().splitlines()
                    if json.loads(line)["event"] != "chatgpt_session"
                )
                + "\n"
            )
            for prompt in ("third", "fourth"):
                provider = await self.provider(events, requests)
                await run(provider, prompt, 200000, 48, resume=path)
            self.assertNotEqual(
                requests[1].headers["session-id"], requests[2].headers["session-id"]
            )
            self.assertEqual(
                requests[2].headers["session-id"], requests[3].headers["session-id"]
            )


if __name__ == "__main__":
    unittest.main()
