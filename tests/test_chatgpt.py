from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

from cryptography.hazmat.primitives.asymmetric import rsa
import httpx
import jwt
from kosong.chat_provider import ChatProviderError
from kosong.message import Message, ToolCall
from kosong.tooling import Tool

from lazarus.chatgpt_auth import (
    ChatGPTAuth,
    ISSUER,
    RESOURCE,
    SCOPES,
    _save,
    verify_identity,
)
from lazarus.chatgpt import ChatGPT
from lazarus.cli import build_parser, create_chat_provider


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
                [Message(role="user", content="hello")],
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


if __name__ == "__main__":
    unittest.main()
