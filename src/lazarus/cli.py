import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from typing import cast

import kosong
from kosong.chat_provider import ChatProvider, ThinkingEffort, TokenUsage
from kosong.message import Message, ToolCall
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolResult, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel


SYSTEM_PROMPT = """You are Lazarus, a coding agent working in {cwd}.

You have two tools:

`python` runs an IPython cell in one long-lived interpreter. Names, imports,
functions, objects, and IPython state survive every tool call and every new
loop. Use it for all computer work: inspect and edit files, run shell commands,
run tests, and keep useful state. It is your persistent workspace, so consider
building your own helpers and functions when they would make repeated work
easier. Calls time out after 300 seconds by default; set `timeout` when needed.
Print only what you need to see.

`start_new_loop` runs one last IPython cell and then replaces the earlier chat
history with that call and its result. You decide when a fresh context would
help.

The `start_new_loop` cell is a free-form handoff to your next loop. There is no
required structure. Use normal Python: comments, variables, functions, cached
file slices, or anything else that will help. Preserve the main ask, what you
did and learned, relevant changes and test results, what remains, the next
action, and work that should not be repeated. Keep large useful values in the
interpreter instead of printing them.

Work carefully and autonomously. Inspect before editing, preserve unrelated
user changes, keep changes focused, check the diff, and run relevant tests.
Finish with a concise account of the result and any verification limits.
"""

PROVIDERS = ("anthropic", "codex", "google", "kimi", "openai", "openai-legacy")
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "codex": "gpt-5.6-sol",
    "google": "gemini-3.7-flash",
    "kimi": "kimi-k3",
    "openai": "gpt-5.6-sol",
}
THINKING_EFFORTS = ("off", "low", "medium", "high", "xhigh", "max")
PYTHON_TOOL = "python"
NEW_LOOP_TOOL = "start_new_loop"
TOKEN_USAGE_PREFIX = "LAZARUS_TOKEN_USAGE "
DEFAULT_LOOP_TOKEN_LIMIT = 250_000
DEFAULT_CELL_TIMEOUT = 300.0
DEFAULT_TOOL_OUTPUT_LIMIT_KIB = 48


@dataclass
class TokenTotals:
    input_other: int = 0
    input_cache_read: int = 0
    input_cache_creation: int = 0
    output: int = 0
    loops_started: int = 0
    loop_context_tokens: int = 0
    loop_steer_sent: bool = False

    def add(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.input_other += usage.input_other
        self.input_cache_read += usage.input_cache_read
        self.input_cache_creation += usage.input_cache_creation
        self.output += usage.output
        self.loop_context_tokens = (
            usage.input_other
            + usage.input_cache_read
            + usage.input_cache_creation
            + usage.output
        )

    @property
    def input(self) -> int:
        return self.input_other + self.input_cache_read + self.input_cache_creation

    @property
    def total(self) -> int:
        return self.input + self.output

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input,
            "input_other": self.input_other,
            "input_cache_read": self.input_cache_read,
            "input_cache_creation": self.input_cache_creation,
            "output": self.output,
            "total": self.total,
            "loops_started": self.loops_started,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A coding agent with persistent IPython state."
    )
    parser.add_argument("--provider", choices=PROVIDERS, default="kimi")
    parser.add_argument(
        "--model",
        help="Model ID; required for openai-legacy, otherwise uses the provider default.",
    )
    parser.add_argument("--thinking-effort", choices=THINKING_EFFORTS)
    parser.add_argument(
        "--loop-token-limit",
        type=int,
        default=DEFAULT_LOOP_TOKEN_LIMIT,
        metavar="TOKENS",
        help="Steer the model to start a new loop at this context size (default: 250k).",
    )
    parser.add_argument(
        "--tool-output-limit-kib",
        type=int,
        default=DEFAULT_TOOL_OUTPUT_LIMIT_KIB,
        metavar="KIB",
        help="Maximum tool output kept in context (default: 48 KiB).",
    )
    parser.add_argument("--prompt", help="Run one request and exit.")
    return parser


def create_chat_provider(args: argparse.Namespace) -> ChatProvider:
    provider = args.provider
    if args.model:
        model = str(args.model)
    elif provider == "openai-legacy":
        raise ValueError("--model is required for the openai-legacy provider")
    else:
        model = DEFAULT_MODELS[provider]

    match provider:
        case "codex":
            from lazarus.codex_chatgpt import CodexChatGPT

            chat = CodexChatGPT(model=model)
        case "kimi":
            from kosong.chat_provider.kimi import Kimi

            chat: ChatProvider = Kimi(model=model, stream=False)
        case "openai":
            from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

            chat = OpenAIResponses(model=model, stream=False)
        case "openai-legacy":
            from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for the openai-legacy provider"
                )
            chat = OpenAILegacy(
                model=model,
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
                stream=False,
                reasoning_key=os.getenv("OPENAI_REASONING_KEY"),
            )
        case "anthropic":
            from kosong.contrib.chat_provider.anthropic import Anthropic

            chat = Anthropic(model=model, stream=False, default_max_tokens=8192)
        case "google":
            from kosong.contrib.chat_provider.google_genai import GoogleGenAI

            chat = GoogleGenAI(model=model, stream=False)
        case _:
            raise ValueError(f"Unsupported provider: {args.provider}")

    if args.thinking_effort:
        return chat.with_thinking(cast(ThinkingEffort, args.thinking_effort))
    return chat


class CellParams(BaseModel):
    code: str
    timeout: float = DEFAULT_CELL_TIMEOUT


class PythonRuntime:
    def __init__(
        self, tool_output_limit_kib: int = DEFAULT_TOOL_OUTPUT_LIMIT_KIB
    ) -> None:
        if tool_output_limit_kib <= 0:
            raise ValueError("tool output limit must be positive")
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._reader_transport: asyncio.ReadTransport | None = None
        self._lock = asyncio.Lock()
        self._tool_output_limit_bytes = tool_output_limit_kib * 1024
        self._tool_output_dir = tempfile.TemporaryDirectory(
            prefix="lazarus-tool-output-"
        )
        self.cwd = os.getcwd()

    async def run(
        self,
        code: str,
        display_name: str | None = None,
        timeout: float = DEFAULT_CELL_TIMEOUT,
    ) -> ToolReturnValue:
        async with self._lock:
            if display_name is not None:
                _print_cell(display_name, code)
            try:
                result = await asyncio.wait_for(self._run_locked(code), timeout)
            except TimeoutError:
                await self._forget_worker()
                result = ToolError(
                    message=f"Cell exceeded the {timeout:g}s timeout; interpreter state was lost.",
                    output="",
                    brief="Cell timed out",
                )
            if display_name is not None:
                _print_result(result)
            return result

    async def _run_locked(self, code: str) -> ToolReturnValue:
        try:
            await self._ensure_worker()
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._reader is not None

            request = json.dumps({"code": code}, ensure_ascii=False) + "\n"
            self._process.stdin.write(request.encode())
            await self._process.stdin.drain()
            raw_response = await self._reader.readline()
            if not raw_response:
                await self._forget_worker()
                return ToolError(
                    message="The IPython worker exited; its in-memory state was lost.",
                    output="",
                    brief="Worker exited",
                )
            response = json.loads(raw_response)
        except (BrokenPipeError, ConnectionResetError, json.JSONDecodeError) as exc:
            await self._forget_worker()
            return ToolError(
                message=f"The IPython worker protocol failed: {exc}",
                output="",
                brief="Worker failed",
            )

        if isinstance(response.get("cwd"), str):
            self.cwd = response["cwd"]
        output = _cell_output(response)
        if response.get("ok"):
            return ToolOk(output=output or "(no output)")
        return ToolError(
            message=str(response.get("error", "IPython cell failed")),
            output=output,
            brief="Cell failed",
        )

    async def _ensure_worker(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return

        await self._forget_worker()
        read_fd, write_fd = os.pipe()
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "lazarus.python_worker",
            str(write_fd),
            str(self._tool_output_limit_bytes),
            self._tool_output_dir.name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            pass_fds=(write_fd,),
            start_new_session=os.name == "posix",
        )
        os.close(write_fd)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_running_loop()
        transport, _ = await loop.connect_read_pipe(
            lambda: protocol,
            os.fdopen(read_fd, "rb", buffering=0),
        )
        self._reader = reader
        self._reader_transport = transport

    async def _forget_worker(self) -> None:
        if self._reader_transport is not None:
            self._reader_transport.close()
        self._reader = None
        self._reader_transport = None

        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=0.5)
            except TimeoutError:
                self._signal_worker(process, force=False)
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.5)
                except TimeoutError:
                    self._signal_worker(process, force=True)
                    await process.wait()
        if os.name == "posix":
            self._signal_worker(process, force=True)

    @staticmethod
    def _signal_worker(process: asyncio.subprocess.Process, *, force: bool) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif force:
            process.kill()
        else:
            process.terminate()

    async def close(self) -> None:
        async with self._lock:
            await self._forget_worker()
            self._tool_output_dir.cleanup()


class CellTool(CallableTool2[CellParams]):
    params = CellParams

    def __init__(self, runtime: PythonRuntime, name: str, description: str) -> None:
        super().__init__(name=name, description=description)
        self.runtime = runtime

    async def __call__(self, params: CellParams) -> ToolReturnValue:
        return await self.runtime.run(params.code, self.name, params.timeout)


def _cell_output(response: dict[str, object]) -> str:
    parts = []
    if stdout := response.get("stdout"):
        parts.append(str(stdout).rstrip())
    if stderr := response.get("stderr"):
        parts.append(f"[stderr]\n{str(stderr).rstrip()}")
    if output_path := response.get("output_path"):
        parts.append(
            f"[full output: {output_path}; inspect targeted sections only]"
        )
    return "\n".join(parts)


def _tool_message(result: ToolResult) -> Message:
    return Message(
        role="tool",
        tool_call_id=result.tool_call_id,
        content=_result_text(result.return_value),
    )


def _new_loop_history(
    task: str, tool_calls: list[ToolCall], results: list[ToolResult]
) -> list[Message] | None:
    for call, result in reversed(list(zip(tool_calls, results, strict=True))):
        if call.function.name == NEW_LOOP_TOOL and not result.return_value.is_error:
            return [
                Message(role="user", content=task),
                Message(role="assistant", content=[], tool_calls=[call]),
                _tool_message(result),
            ]
    return None


def _print_cell(name: str, code: str) -> None:
    print(f"\n[{name}]\n{code}")


def _print_result(value: ToolReturnValue) -> None:
    label = "error" if value.is_error else "output"
    print(f"\n[{label}]\n{_result_text(value)}")


def _result_text(value: ToolReturnValue) -> str:
    parts = [str(value.output)] if value.output else []
    if value.message:
        parts.append(value.message)
    return "\n".join(parts).rstrip() or "(no output)"


def _print_token_usage(totals: TokenTotals) -> None:
    print(f"{TOKEN_USAGE_PREFIX}{json.dumps(totals.as_dict(), separators=(',', ':'))}")


def _system_prompt(cwd: str) -> str:
    return SYSTEM_PROMPT.format(cwd=cwd)


async def run_request(
    chat: ChatProvider,
    toolset: SimpleToolset,
    runtime: PythonRuntime,
    history: list[Message],
    user_input: str,
    token_totals: TokenTotals,
    loop_token_limit: int,
) -> list[Message]:
    history.append(Message(role="user", content=user_input))

    while True:
        step = await kosong.step(
            chat_provider=chat,
            toolset=toolset,
            history=history,
            system_prompt=_system_prompt(runtime.cwd),
        )
        token_totals.add(step.usage)
        _print_token_usage(token_totals)
        history.append(step.message)
        if text := step.message.extract_text():
            print(f"\n[assistant]\n{text}")
        results = await step.tool_results()
        result_messages = [_tool_message(result) for result in results]
        history.extend(result_messages)

        if new_history := _new_loop_history(user_input, step.tool_calls, results):
            token_totals.loops_started += 1
            token_totals.loop_context_tokens = 0
            token_totals.loop_steer_sent = False
            history = new_history
            print("\n[new loop]\nPrevious chat history was replaced.")
            continue

        if (
            results
            and not token_totals.loop_steer_sent
            and token_totals.loop_context_tokens >= loop_token_limit
        ):
            steer_message = (
                f"This loop's current context has reached at least {loop_token_limit} "
                "tokens. Compact the useful state into a concise handoff and call "
                "`start_new_loop` now."
            )
            history.append(Message(role="user", content=steer_message))
            token_totals.loop_steer_sent = True
            print(f"\n[steer]\n{steer_message}")

        if not results:
            return history


async def run(
    chat: ChatProvider,
    prompt: str | None,
    loop_token_limit: int,
    tool_output_limit_kib: int,
) -> None:
    runtime = PythonRuntime(tool_output_limit_kib)
    toolset = SimpleToolset(
        [
            CellTool(
                runtime,
                PYTHON_TOOL,
                "Run a persistent IPython cell. State survives calls and new loops.",
            ),
            CellTool(
                runtime,
                NEW_LOOP_TOOL,
                "Run a free-form handoff cell, replace chat history, and continue.",
            ),
        ]
    )
    history: list[Message] = []
    token_totals = TokenTotals()

    print(f"Lazarus · {chat.name} · {chat.model_name}")
    try:
        if prompt is not None:
            await run_request(
                chat,
                toolset,
                runtime,
                history,
                prompt,
                token_totals,
                loop_token_limit,
            )
            return

        while True:
            try:
                user_input = input("\n> ")
            except EOFError:
                break
            if user_input.strip() == "/quit":
                break
            history = await run_request(
                chat,
                toolset,
                runtime,
                history,
                user_input,
                token_totals,
                loop_token_limit,
            )
    finally:
        await runtime.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.tool_output_limit_kib <= 0:
        parser.error("--tool-output-limit-kib must be positive")
    try:
        chat = create_chat_provider(args)
        asyncio.run(
            run(
                chat,
                args.prompt,
                args.loop_token_limit,
                args.tool_output_limit_kib,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
