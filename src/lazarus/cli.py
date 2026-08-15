import argparse
import asyncio
import json
import os
import sys
from typing import cast

import kosong
from kosong.chat_provider import ChatProvider, ThinkingEffort
from kosong.message import Message, ToolCall
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolResult, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel


SYSTEM_PROMPT = """You are Lazarus, a coding agent working in {cwd}.

You have two tools:

`python` runs an IPython cell in one long-lived interpreter. Names, imports,
functions, objects, and IPython state survive every tool call and every new
loop. Use it for all computer work: inspect and edit files, run shell commands,
run tests, and keep useful state. Print only what you need to see.

`start_new_loop` runs one last IPython cell and then replaces the earlier chat
history with that call and its result. You decide when a fresh context would
help.

The `start_new_loop` cell is a free-form handoff to your next loop. There is no
required structure. Use normal Python: comments, variables, functions, cached
file slices, or anything else that will help. Preserve the main ask, what you
did and learned, relevant changes and test results, what remains, the next
action, and work that should not be repeated. Keep large useful values in the
interpreter instead of printing them.

After a new loop, you will see a notice followed by your retained handoff call
and its result. The interpreter is the same. Trust and use the state you left.
Continue from the recorded next action; do not repeat repository discovery or
reread preserved files without a reason.

Work carefully and autonomously. Inspect before editing, preserve unrelated
user changes, keep changes focused, check the diff, and run relevant tests.
Finish with a concise account of the result and any verification limits.
"""

NEW_LOOP_NOTICE = """A new Lazarus loop has started.

Earlier chat history was intentionally replaced. The assistant tool call and
tool result immediately below are your handoff from the previous loop. The
IPython interpreter and its state are unchanged. Continue the same task from
that handoff without repeating completed discovery or work.
"""

PROVIDERS = ("anthropic", "google", "kimi", "openai", "zai")
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "google": "gemini-3.7-flash",
    "kimi": "kimi-k3",
    "openai": "gpt-5.6-sol",
    "zai": "glm-5.2",
}
THINKING_EFFORTS = ("off", "low", "medium", "high", "xhigh", "max")
PYTHON_TOOL = "python"
NEW_LOOP_TOOL = "start_new_loop"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A coding agent with persistent IPython state.")
    parser.add_argument("--provider", choices=PROVIDERS, default="kimi")
    parser.add_argument("--model", help="Model ID; defaults to the provider's current model.")
    parser.add_argument("--thinking-effort", choices=THINKING_EFFORTS)
    parser.add_argument("--prompt", help="Run one request and exit.")
    return parser


def create_chat_provider(args: argparse.Namespace) -> ChatProvider:
    provider = args.provider
    model = str(args.model) if args.model else DEFAULT_MODELS[provider]

    match provider:
        case "kimi":
            from kosong.chat_provider.kimi import Kimi

            chat: ChatProvider = Kimi(model=model, stream=False)
        case "openai":
            from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

            chat = OpenAIResponses(model=model, stream=False)
        case "zai":
            from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

            api_key = os.getenv("ZAI_API_KEY")
            if not api_key:
                raise ValueError("ZAI_API_KEY is required for the Z.AI provider")
            chat = OpenAILegacy(
                model=model,
                api_key=api_key,
                base_url=os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/"),
                stream=False,
                reasoning_key="reasoning_content",
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


class PythonRuntime:
    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._reader_transport: asyncio.ReadTransport | None = None
        self._lock = asyncio.Lock()
        self.cwd = os.getcwd()

    async def run(self, code: str) -> ToolReturnValue:
        async with self._lock:
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
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            pass_fds=(write_fd,),
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
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.5)
                except TimeoutError:
                    process.kill()
                    await process.wait()

    async def close(self) -> None:
        async with self._lock:
            await self._forget_worker()


class CellTool(CallableTool2[CellParams]):
    params = CellParams

    def __init__(self, runtime: PythonRuntime, name: str, description: str) -> None:
        super().__init__(name=name, description=description)
        self.runtime = runtime

    async def __call__(self, params: CellParams) -> ToolReturnValue:
        _print_cell(self.name, params.code)
        result = await self.runtime.run(params.code)
        _print_result(result)
        return result


def _cell_output(response: dict[str, object]) -> str:
    parts = []
    if stdout := response.get("stdout"):
        parts.append(str(stdout).rstrip())
    if stderr := response.get("stderr"):
        parts.append(f"[stderr]\n{str(stderr).rstrip()}")
    return "\n".join(parts)


def _tool_message(result: ToolResult) -> Message:
    return Message(
        role="tool",
        tool_call_id=result.tool_call_id,
        content=_result_text(result.return_value),
    )


def _new_loop_history(
    tool_calls: list[ToolCall], results: list[ToolResult]
) -> list[Message] | None:
    for call, result in reversed(list(zip(tool_calls, results, strict=True))):
        if call.function.name == NEW_LOOP_TOOL and not result.return_value.is_error:
            return [
                Message(role="user", content=NEW_LOOP_NOTICE),
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


async def run_request(
    chat: ChatProvider,
    toolset: SimpleToolset,
    runtime: PythonRuntime,
    history: list[Message],
    user_input: str,
) -> list[Message]:
    history.append(Message(role="user", content=user_input))

    while True:
        step = await kosong.step(
            chat_provider=chat,
            toolset=toolset,
            history=history,
            system_prompt=SYSTEM_PROMPT.format(cwd=runtime.cwd),
        )
        history.append(step.message)
        results = await step.tool_results()
        result_messages = [_tool_message(result) for result in results]
        history.extend(result_messages)

        if new_history := _new_loop_history(step.tool_calls, results):
            history = new_history
            print("\n[new loop]\nPrevious chat history was replaced.")
            continue

        if not results:
            print(f"\n[assistant]\n{step.message.extract_text()}")
            return history


async def run(chat: ChatProvider, prompt: str | None) -> None:
    runtime = PythonRuntime()
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

    print(f"Lazarus · {chat.name} · {chat.model_name}")
    try:
        if prompt is not None:
            await run_request(chat, toolset, runtime, history, prompt)
            return

        while True:
            try:
                user_input = input("\n> ")
            except EOFError:
                break
            if user_input.strip() == "/quit":
                break
            history = await run_request(chat, toolset, runtime, history, user_input)
    finally:
        await runtime.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        chat = create_chat_provider(args)
        asyncio.run(run(chat, args.prompt))
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
