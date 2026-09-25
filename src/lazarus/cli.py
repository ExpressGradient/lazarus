import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import cast

import kosong
from kosong.chat_provider import ChatProvider, ThinkingEffort, TokenUsage
from kosong.message import Message, ToolCall
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolResult, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel, ConfigDict, Field

from lazarus.jobs import Jobs
from lazarus.session import Session
from lazarus.runtime import (
    DEFAULT_CELL_TIMEOUT,
    DEFAULT_TOOL_OUTPUT_LIMIT_KIB,
    PythonRuntime,
)


SYSTEM_PROMPT = """You are Lazarus, a coding agent starting in {cwd}.

You have three tools:

`python` runs an IPython cell in one long-lived interpreter. Names, imports,
functions, objects, and IPython state survive every tool call and every new
loop. It returns a job handle immediately by default; use `yield_after` to wait
briefly for a result. `timeout` is a separate execution deadline (300 seconds
by default). One cell runs at a time; a busy interpreter rejects new cells.

`job` observes execution outside the interpreter. Pass `id` to read new output,
`wait` to wait up to 60 seconds, or `cancel=true` to request interruption.
Omit `id` to list retained jobs. Reads never rerun code. Output has a byte
cursor; pass `cursor` to reread from a specific offset. Completion is reported
once between turns. If there is nothing useful to do, wait instead of polling.
Cancellation and timeouts may leave partial effects; inspect before retrying.

Python is your workspace and your tool-building language. Compose operations,
wrap awkward APIs, build small helpers, batch independent work, cache expensive
results, and inspect data programmatically. Use libraries, shell commands,
threads, and subprocesses creatively. Build abstractions when they save work.
Keep large objects in memory; print only evidence needed for the next decision.
For overlap, launch subprocesses from a short cell with explicit log files and
retain their handles. Background threads share globals and output with later
cells; prefer subprocesses for independent work. An asyncio task alone is not
a durable background job: the interpreter's event loop may stop between cells.
Wait for required work and check its result before claiming success. Track and
clean up processes you launch. Session exit stops the interpreter and its process
group, including servers; do not promise they will survive exit.

`start_new_loop` runs one last IPython cell and then replaces the earlier chat
history with that call and its result. You decide when a fresh context would
help. It waits for its own cell to finish and requires the interpreter to be
idle. Jobs and logs survive context resets. Use `job` to recover their handles.

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
DEFAULT_LOOP_TOKEN_LIMIT = 150_000


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
        help="Steer the model to start a new loop at this context size (default: 150k).",
    )
    parser.add_argument(
        "--tool-output-limit-kib",
        type=int,
        default=DEFAULT_TOOL_OUTPUT_LIMIT_KIB,
        metavar="KIB",
        help="Maximum tool output kept in context (default: 48 KiB).",
    )
    parser.add_argument("--prompt", help="Run one request and exit.")
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument(
        "--session-dir", help="New session directory for the journal and job logs."
    )
    sessions.add_argument(
        "--resume",
        metavar="DIR",
        help="Resume a session with a fresh interpreter; never replay cells.",
    )
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
    model_config = ConfigDict(extra="forbid")
    code: str
    timeout: float = Field(default=DEFAULT_CELL_TIMEOUT, gt=0, allow_inf_nan=False)
    yield_after: float = Field(default=0, ge=0, le=60, allow_inf_nan=False)


class JobParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    wait: float = Field(default=0, ge=0, le=60, allow_inf_nan=False)
    cancel: bool = False
    cursor: int | None = Field(default=None, ge=0)


class JobTool(CallableTool2[JobParams]):
    params = JobParams

    def __init__(self, jobs: Jobs) -> None:
        super().__init__(
            name="job",
            description="Read, wait for, or cancel a Python job without blocking the interpreter. Omit id to list jobs.",
        )
        self.jobs = jobs

    async def __call__(self, params: JobParams) -> ToolReturnValue:
        if params.id is None:
            if params.cancel or params.wait or params.cursor is not None:
                return ToolError(
                    message="id is required for wait, cancel, or cursor.",
                    brief="Missing job ID",
                )
            return ToolOk(output=self.jobs.listing())
        result = await self.jobs.inspect(
            params.id, wait=params.wait, cancel=params.cancel, cursor=params.cursor
        )
        _print_result(result)
        return result


class CellTool(CallableTool2[CellParams]):
    params = CellParams

    def __init__(self, jobs: Jobs, name: str, description: str) -> None:
        super().__init__(name=name, description=description)
        self.jobs = jobs

    async def __call__(self, params: CellParams) -> ToolReturnValue:
        _print_cell(self.name, params.code)
        if self.name == PYTHON_TOOL:
            result = await self.jobs.submit(
                params.code, params.timeout, params.yield_after
            )
        else:
            result = await self.jobs.submit(
                params.code, params.timeout, 0, wait_completion=True
            )
        _print_result(result)
        return result


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
    *,
    jobs: Jobs | None = None,
    session: Session | None = None,
    system_prompt: str | None = None,
    continuation: bool = False,
) -> list[Message]:
    def append(message: Message) -> None:
        if session:
            session.message(message)
        history.append(message)

    def append_completions() -> bool:
        notices = jobs.notifications() if jobs else []
        if notices:
            append(
                Message(role="user", content="[Job completion]\n" + "\n".join(notices))
            )
            print("\n[job completion]\n" + "\n".join(notices))
        return bool(notices)

    if not continuation:
        if session:
            session.record("request", task=user_input)
        append(Message(role="user", content=user_input))
    # cwd changes are reported in job results, never in the cached prefix.
    system_prompt = system_prompt or _system_prompt(
        getattr(runtime, "initial_cwd", runtime.cwd)
    )

    while True:
        append_completions()
        step = await kosong.step(
            chat_provider=chat,
            toolset=toolset,
            history=history,
            system_prompt=system_prompt,
        )
        token_totals.add(step.usage)
        _print_token_usage(token_totals)
        append(step.message)
        if text := step.message.extract_text():
            print(f"\n[assistant]\n{text}")
        results = await step.tool_results()
        for result in results:
            append(_tool_message(result))

        if new_history := _new_loop_history(user_input, step.tool_calls, results):
            token_totals.loops_started += 1
            token_totals.loop_context_tokens = 0
            token_totals.loop_steer_sent = False
            history = new_history
            if session:
                session.record(
                    "reset",
                    history=[message.model_dump(mode="json") for message in history],
                )
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
            append(Message(role="user", content=steer_message))
            token_totals.loop_steer_sent = True
            print(f"\n[steer]\n{steer_message}")

        if not results:
            if jobs and (active := jobs.active) and active.task is not None:
                print(f"\n[waiting for job {active.id}]")
                # No model polling loop when it has no independent work left.
                await asyncio.shield(active.task)
            if append_completions():
                continue
            return history


async def run(
    chat: ChatProvider,
    prompt: str | None,
    loop_token_limit: int,
    tool_output_limit_kib: int,
    session_dir: str | None = None,
    resume: str | None = None,
) -> None:
    session = Session(resume or session_dir, resume=resume is not None)
    runtime = PythonRuntime(tool_output_limit_kib)
    jobs = Jobs(runtime, session.directory / "jobs", session.record)
    toolset = SimpleToolset(
        [
            CellTool(
                jobs,
                PYTHON_TOOL,
                "Run a persistent IPython cell. Returns a job immediately; yield_after waits briefly. State survives calls and new loops.",
            ),
            CellTool(
                jobs,
                NEW_LOOP_TOOL,
                "Run a free-form handoff cell, replace chat history, and continue.",
            ),
            JobTool(jobs),
        ]
    )
    history: list[Message] = []
    token_totals = TokenTotals()
    system_prompt = _system_prompt(runtime.initial_cwd)

    print(f"Lazarus · {chat.name} · {chat.model_name}")
    print(f"Session: {session.directory}")
    try:
        if resume:
            history, task, system_prompt, cwd = session.restore()
            runtime.cwd = cwd
            runtime.initial_cwd = cwd
            if prompt is None:
                history = await run_request(
                    chat,
                    toolset,
                    runtime,
                    history,
                    task,
                    token_totals,
                    loop_token_limit,
                    jobs=jobs,
                    session=session,
                    system_prompt=system_prompt,
                    continuation=True,
                )
        else:
            session.record("session", system_prompt=system_prompt, cwd=runtime.cwd)
        if prompt is not None:
            await run_request(
                chat,
                toolset,
                runtime,
                history,
                prompt,
                token_totals,
                loop_token_limit,
                jobs=jobs,
                session=session,
                system_prompt=system_prompt,
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
                jobs=jobs,
                session=session,
                system_prompt=system_prompt,
            )
    finally:
        try:
            await jobs.close()
        finally:
            try:
                await runtime.close()
            finally:
                session.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.tool_output_limit_kib <= 0:
        parser.error("--tool-output-limit-kib must be positive")
    if args.loop_token_limit <= 0:
        parser.error("--loop-token-limit must be positive")
    try:
        chat = create_chat_provider(args)
        asyncio.run(
            run(
                chat,
                args.prompt,
                args.loop_token_limit,
                args.tool_output_limit_kib,
                args.session_dir,
                args.resume,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
