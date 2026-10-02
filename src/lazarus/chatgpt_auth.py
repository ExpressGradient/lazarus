"""Lazarus-owned OAuth credentials for ChatGPT plan usage."""

from __future__ import annotations

import base64
from datetime import datetime
from contextlib import contextmanager
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid
import webbrowser

import httpx
import jwt

from lazarus.chatgpt_errors import (
    ChatGPTError,
    REFRESH_REJECTED,
    response_error,
    retry_delay,
    transport_error,
)

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"


def _save(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(response: httpx.Response, operation: str = "authentication") -> dict:
    if response.is_error:
        raise response_error(operation, response)
    return response.json()


class ChatGPTAuth:
    def __init__(self, account: str = "default", directory: Path | None = None):
        self.account = account
        self.directory = (
            directory
            or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "lazarus"
        )
        self.path = self.directory / "chatgpt.json"

    @contextmanager
    def locked(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.directory / "chatgpt.lock", os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = (
                    json.loads(self.path.read_text())
                    if self.path.exists()
                    else {
                        "host_id": f"urn:uuid:{uuid.uuid4()}",
                        "accounts": {},
                    }
                )
                yield data
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _record(self, data: dict) -> dict:
        return data["accounts"].get(self.account, {})

    @staticmethod
    def _tokens(tokens: dict) -> dict:
        if not all(
            isinstance(tokens.get(k), str) and tokens[k]
            for k in ("access_token", "refresh_token")
        ):
            raise ValueError("ChatGPT returned incomplete credentials; sign in again.")
        if tokens.get("token_type", "").lower() != "bearer":
            raise ValueError("ChatGPT returned an unsupported token type.")
        lifetime = float(tokens["expires_in"])
        if not math.isfinite(lifetime) or lifetime <= 0:
            raise ValueError("ChatGPT returned an invalid token lifetime.")
        return {**tokens, "expires_at": time.time() + lifetime}

    def access_token(self) -> str:
        # The lock covers read/refresh/write, including other Lazarus processes.
        with self.locked() as data:
            record = self._record(data)
            if not record.get("refresh_token"):
                raise ValueError(
                    f"ChatGPT account {self.account!r} is signed out; run `lazarus auth login`."
                )
            expires = record.get("expires_at", 0)
            earliest = record.get("earliest_refresh_at", 0)
            try:
                earliest = float(earliest)
            except (TypeError, ValueError):
                try:
                    earliest = datetime.fromisoformat(
                        str(earliest).replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    earliest = 0
            if not math.isfinite(earliest):
                earliest = 0
            if expires <= time.time() + 60 and earliest <= time.time():
                try:
                    tokens = self._refresh(record)
                except ChatGPTError as error:
                    if error.code in REFRESH_REJECTED:
                        for key in (
                            "access_token",
                            "refresh_token",
                            "id_token",
                            "expires_at",
                            "earliest_refresh_at",
                        ):
                            record.pop(key, None)
                        _save(self.path, data)
                    raise
                updated = self._tokens(tokens)
                updated.setdefault("scope", record.get("scope", ""))
                record.pop("earliest_refresh_at", None)
                record.update(updated)
                _save(self.path, data)
            elif expires <= time.time():
                raise ChatGPTError("token refresh", code="refresh_not_ready")
            if "chatgpt.tokens.use.direct" not in record.get("scope", "").split():
                raise ValueError(
                    "ChatGPT plan usage was not authorized; run `lazarus auth login`."
                )
            return record["access_token"]

    def _refresh(self, record: dict) -> dict:
        # The caller holds the credential lock through rotation and atomic save.
        with httpx.Client(timeout=30) as client:
            for attempt in range(3):
                try:
                    try:
                        response = client.post(
                            TOKEN_URL,
                            data={
                                "grant_type": "refresh_token",
                                "client_id": record["client_id"],
                                "refresh_token": record["refresh_token"],
                                "resource": RESOURCE,
                            },
                        )
                    except httpx.HTTPError as error:
                        raise transport_error("token refresh", error) from error
                    return _json(response, "token refresh")
                except ChatGPTError as error:
                    delay = retry_delay(error, attempt)
                    if delay is None:
                        error.retryable = False
                        raise
                    time.sleep(delay)
        raise AssertionError("unreachable")

    def login(self) -> None:
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        result: dict[str, str] = {}

        class Callback(BaseHTTPRequestHandler):
            def setup(self):
                self.request.settimeout(5)
                super().setup()

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self):
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query)
                valid = (
                    parsed.path == "/auth/callback"
                    and query.get("state") == [state]
                    and all(len(values) == 1 for values in query.values())
                )
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(
                    b"Return to Lazarus to finish sign-in."
                    if valid
                    else b"Invalid callback."
                )
                if valid:
                    result.update({key: values[0] for key, values in query.items()})

        # Keep one login/refresh/logout transaction active per credential store.
        with self.locked() as data, HTTPServer(("127.0.0.1", 0), Callback) as server:
            _save(self.path, data)  # Persist host ID even if sign-in is cancelled.
            record = self._record(data)
            client_id = record.get("client_id", "dynamic_agent_client")
            redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
            params = dict(
                client_id=client_id,
                ext_agent_host_id=data["host_id"],
                response_type="code",
                redirect_uri=redirect,
                scope=SCOPES,
                resource=RESOURCE,
                state=state,
                nonce=nonce,
                code_challenge_method="S256",
                code_challenge=challenge,
            )
            if client_id == "dynamic_agent_client":
                params["agent_name_hint"] = "Lazarus"
            elif record.get("email"):
                params["login_hint"] = record["email"]
            url = f"{ISSUER}/api/accounts/authorize?{urlencode(params)}"
            print(
                "Continue with ChatGPT in your browser. If it does not open, visit:\n"
                + url
            )
            webbrowser.open(url)
            deadline = time.monotonic() + 180
            server.timeout = 1
            while not result and time.monotonic() < deadline:
                server.handle_request()
            if not result:
                raise ValueError("ChatGPT sign-in timed out.")
            if "error" in result:
                raise ValueError("ChatGPT sign-in was declined or cancelled.")
            issued = result.get("client_id", client_id)
            if not issued or issued == "dynamic_agent_client" or not result.get("code"):
                raise ValueError("ChatGPT registration was incomplete.")
            if client_id != "dynamic_agent_client" and issued != client_id:
                raise ValueError("ChatGPT returned a different client registration.")
            if client_id == "dynamic_agent_client":
                record = {"client_id": issued}
                data["accounts"][self.account] = record
                _save(self.path, data)  # Reuse registration if code exchange fails.
            with httpx.Client(timeout=30) as client:
                tokens = _json(
                    client.post(
                        TOKEN_URL,
                        data=dict(
                            grant_type="authorization_code",
                            client_id=issued,
                            code=result["code"],
                            code_verifier=verifier,
                            redirect_uri=redirect,
                            resource=RESOURCE,
                        ),
                    )
                )
                discovery = _json(
                    client.get(f"{ISSUER}/.well-known/openid-configuration")
                )
                jwks = _json(client.get(discovery["jwks_uri"]))
            identity = verify_identity(tokens.get("id_token", ""), jwks, issued, nonce)
            if record.get("subject") and record["subject"] != identity["sub"]:
                raise ValueError(
                    "ChatGPT identity changed; use a different --account label."
                )
            updated = self._tokens(tokens)
            updated.setdefault("scope", "")
            record.update(
                updated, subject=identity["sub"], email=identity.get("email", "")
            )
            data["accounts"][self.account] = record
            _save(self.path, data)
            if "chatgpt.tokens.use.direct" not in record.get("scope", "").split():
                raise ValueError(
                    "Signed in, but ChatGPT plan usage was not authorized."
                )
            print(f"Connected ChatGPT account {self.account!r} ({record['email']}).")

    def status(self) -> None:
        with self.locked() as data:
            for label, record in data["accounts"].items():
                status = "connected" if record.get("refresh_token") else "signed out"
                print(f"{label}: {status} {record.get('email', '')}")
            if not data["accounts"]:
                print("No ChatGPT accounts. Run `lazarus auth login`.")

    def logout(self) -> None:
        with self.locked() as data:
            record = self._record(data)
            if record.get("refresh_token"):
                try:
                    with httpx.Client(timeout=30) as client:
                        discovery = _json(
                            client.get(f"{ISSUER}/.well-known/openid-configuration")
                        )
                        response = client.post(
                            discovery["revocation_endpoint"],
                            data={
                                "token": record["refresh_token"],
                                "token_type_hint": "refresh_token",
                                "client_id": record["client_id"],
                            },
                        )
                        response.raise_for_status()
                except (httpx.HTTPError, ValueError, KeyError):
                    print(
                        "Remote revocation was not confirmed; disconnect Lazarus in ChatGPT settings."
                    )
                data["accounts"][self.account] = {
                    k: record[k]
                    for k in ("client_id", "subject", "email")
                    if k in record
                }
                _save(self.path, data)
            print(f"Signed out of {self.account!r}.")


def verify_identity(token: str, jwks: dict, client_id: str, nonce: str) -> dict:
    try:
        header = jwt.get_unverified_header(token)
        key = next(key for key in jwks["keys"] if key.get("kid") == header.get("kid"))
        claims = jwt.decode(
            token,
            jwt.PyJWK.from_dict(key).key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "nonce"]},
        )
        if not secrets.compare_digest(claims["nonce"], nonce):
            raise ValueError("Nonce mismatch")
        return claims
    except (jwt.PyJWTError, StopIteration, KeyError, TypeError, ValueError) as exc:
        raise ValueError("ChatGPT identity validation failed.") from exc
