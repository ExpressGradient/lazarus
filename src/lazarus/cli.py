import argparse
import asyncio
import json
import os
import sys
from typing import cast

import kosong
from kosong.chat_provider import ChatProvider, ThinkingEffort, TokenUsage
from kosong.message import Message
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolResult, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

console = Console()
BLOCK_PADDING = (1, 2, 0, 2)
DEFAULT_CARRYOVER_THRESHOLD_TOKENS = 200_000


def read_carryover_threshold_tokens() -> int:
    raw_threshold = os.getenv("LAZARUS_CARRYOVER_THRESHOLD")

    if raw_threshold is None:
        return DEFAULT_CARRYOVER_THRESHOLD_TOKENS

    try:
        threshold = int(raw_threshold)
    except ValueError:
        print(
            "error: LAZARUS_CARRYOVER_THRESHOLD must be a positive integer",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if threshold <= 0:
        print(
            "error: LAZARUS_CARRYOVER_THRESHOLD must be a positive integer",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return threshold


CARRYOVER_THRESHOLD_TOKENS = read_carryover_threshold_tokens()

SYSTEM_PROMPT = """You are Lazarus, a Powerful Coding Agent.
Current working directory: {cwd}

You can take actions with the run_python tool.
It runs Python code in a persistent interpreter.
Variables, imports, helper functions, and other Python state survive between calls.

Use run_python to inspect and modify files.
Use it to run shell commands through subprocess.
Use it to execute tests, lint code, and analyze outputs.
Treat it as your workspace action tool.

Work like a careful senior engineer:
- Inspect relevant files and existing patterns before editing.
- Prefer the project's current style, tools, and abstractions.
- Keep changes focused on the user's request.
- Never overwrite or revert unrelated user changes.
- Check your diff before finishing.
- Run targeted tests, linters, or other verification when feasible.
- If verification is blocked, say exactly what blocked it.
- For long tasks, keep concise notes in Python state so you can continue
  accurately after carryover.

When context gets long, Lazarus may inject an internal carryover request for
one run_python cell, then reset chat history to the original user request plus
that cell and its result. If you see that compact history, continue from the
carried-over Python state as the next iteration of the same task.

Be proactive when the user asks for a change.
Implement it when you can.
Explain what you did and what you learned.
Keep responses concise and use Markdown when helpful."""

CARRYOVER_INSTRUCTION = """Context threshold reached.

You must now make exactly one run_python tool call and no prose.
Write one Python cell that preserves everything needed for the next iteration to continue the original user request after chat history is reset.

The cell can contain comments, strings, lists, dicts, imports, helper functions, variables, and any compact notes you want to leave for yourself.
Prefer storing a concise handoff in well-named variables or functions.
Do not perform unrelated workspace changes.
"""

PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "gemini": "google-genai",
    "google": "google-genai",
    "google-genai": "google-genai",
    "kimi": "kimi",
    "moonshot": "kimi",
    "openai": "openai-responses",
    "openai-legacy": "openai-legacy",
    "openai-responses": "openai-responses",
}

THINKING_EFFORTS = ("off", "low", "medium", "high", "xhigh", "max")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lazarus",
        description="Run the Lazarus coding agent CLI.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_ALIASES),
        default="kimi",
        help="Chat provider to use. Defaults to kimi.",
    )
    parser.add_argument(
        "--model",
        default="kimi-k2.6",
        help="Model name to pass to the selected provider. Defaults to kimi-k2.6.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable provider streaming.",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=THINKING_EFFORTS,
        help="Enable provider thinking/reasoning effort when supported.",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=8192,
        help="Default max_tokens for the Anthropic provider. Defaults to 8192.",
    )
    parser.add_argument(
        "--prompt",
        help="Run a single non-interactive job with the given prompt and exit.",
    )
    return parser


def create_chat_provider(args: argparse.Namespace) -> ChatProvider:
    provider_name = PROVIDER_ALIASES[args.provider]
    common_kwargs = {
        "model": args.model,
        "stream": args.stream,
    }

    match provider_name:
        case "kimi":
            from kosong.chat_provider.kimi import Kimi

            chat_provider: ChatProvider = Kimi(**common_kwargs)
        case "openai-responses":
            from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

            chat_provider = OpenAIResponses(**common_kwargs)
        case "openai-legacy":
            from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

            chat_provider = OpenAILegacy(**common_kwargs)
        case "anthropic":
            from kosong.contrib.chat_provider.anthropic import Anthropic

            chat_provider = Anthropic(
                **common_kwargs,
                default_max_tokens=args.anthropic_max_tokens,
            )
        case "google-genai":
            from kosong.contrib.chat_provider.google_genai import GoogleGenAI

            chat_provider = GoogleGenAI(**common_kwargs)
        case _:
            raise ValueError(f"Unsupported provider: {args.provider}")

    if args.thinking_effort:
        return chat_provider.with_thinking(cast(ThinkingEffort, args.thinking_effort))
    return chat_provider


class RunPythonParams(BaseModel):
    code: str


class RunPython(CallableTool2[RunPythonParams]):
    name = "run_python"
    description = (
        "Run Python code in a persistent interpreter. State persists across calls."
    )
    params = RunPythonParams

    def __init__(self) -> None:
        super().__init__()
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def __call__(self, params: RunPythonParams) -> ToolReturnValue:
        async with self._lock:
            print_block(
                "Run Python",
                Syntax(params.code, "python", line_numbers=True, word_wrap=True),
                "magenta",
            )

            if self._proc is None or self._proc.returncode is not None:
                self._proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-u",
                    "-m",
                    "lazarus.python_worker",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                )

            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            self._proc.stdin.write((json.dumps({"code": params.code}) + "\n").encode())
            await self._proc.stdin.drain()

            response = json.loads(await self._proc.stdout.readline())

            output = response["stdout"] + response["stderr"]
            duration = response["duration"]
            brief = f"Python code executed in {duration:.3f}s"

            if response["ok"]:
                result = ToolOk(output=output, brief=brief)
            else:
                result = ToolError(
                    message=response["error"],
                    output=output,
                    brief=brief,
                )

            print_tool_result(ToolResult(tool_call_id="", return_value=result))
            return result


def tool_result_to_message(result: ToolResult) -> Message:
    return Message(
        role="tool",
        tool_call_id=result.tool_call_id,
        content=tool_result_text(result),
    )


def print_block(
    title: str, renderable, border_style: str, subtitle: str | None = None
) -> None:
    console.print(
        Padding(
            Panel(
                renderable,
                title=title,
                border_style=border_style,
                subtitle=Text(subtitle, style="dim") if subtitle else None,
            ),
            BLOCK_PADDING,
        )
    )


def print_tool_result(result: ToolResult) -> None:
    value = result.return_value
    title = "Tool Error" if value.is_error else "Tool Output"
    border_style = "red" if value.is_error else "green"
    parts = []

    if value.output:
        parts.append(str(value.output))
    if value.is_error and value.message:
        parts.append(value.message)

    print_block(
        title,
        "\n".join(parts).rstrip() or "(no output)",
        border_style,
        subtitle=value.brief or None,
    )


def tool_result_text(result: ToolResult) -> str:
    value = result.return_value
    parts = []

    if value.output:
        parts.append(str(value.output))
    if value.message:
        parts.append(value.message)
    if value.brief:
        parts.append(value.brief)

    return "\n".join(parts).rstrip() or "(no output)"


def ask_user() -> str | None:
    console.print()
    user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
    if user_input.strip() == "/quit":
        return None
    return user_input


def _usage_text(usage: TokenUsage | None, total_input: int, total_output: int) -> str:
    if usage is None:
        return f"session: {total_input:,} in / {total_output:,} out"
    return f"{usage.input:,} in / {usage.output:,} out | session: {total_input:,} in / {total_output:,} out"


async def _run_request(
    chat_provider: ChatProvider,
    toolset: SimpleToolset,
    history: list[Message],
    user_input: str,
) -> list[Message]:
    original_user_message = Message(role="user", content=user_input)
    history.append(original_user_message)
    carryover_requested = False
    total_input = 0
    total_output = 0

    while True:
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            step_result = await kosong.step(
                chat_provider=chat_provider,
                toolset=toolset,
                history=history,
                system_prompt=SYSTEM_PROMPT.format(cwd=os.getcwd()),
            )
        history.append(step_result.message)

        if step_result.usage:
            total_input += step_result.usage.input
            total_output += step_result.usage.output

        tool_results = await step_result.tool_results()
        history.extend(tool_result_to_message(result) for result in tool_results)

        if carryover_requested:
            history = [
                original_user_message,
                step_result.message,
                *(tool_result_to_message(result) for result in tool_results),
            ]
            carryover_requested = False
            total_input = 0
            total_output = 0
            console.print(
                "[dim]Carryover cell saved; chat history reset for the next iteration.[/dim]"
            )
            continue

        if total_input + total_output >= CARRYOVER_THRESHOLD_TOKENS:
            history.append(Message(role="user", content=CARRYOVER_INSTRUCTION))
            carryover_requested = True
            console.print(
                f"[dim]Context threshold reached "
                f"({total_input + total_output:,}/"
                f"{CARRYOVER_THRESHOLD_TOKENS:,} tokens); requesting carryover cell.[/dim]"
            )
            continue

        if len(tool_results) == 0:
            subtitle = _usage_text(step_result.usage, total_input, total_output)
            print_block(
                "Assistant",
                Markdown(step_result.message.extract_text()),
                "blue",
                subtitle=subtitle,
            )
            return history

        console.print(
            f"[dim]{_usage_text(step_result.usage, total_input, total_output)}[/dim]"
        )


async def _main(chat_provider: ChatProvider, prompt: str | None = None) -> None:
    history: list[Message] = []

    toolset = SimpleToolset()
    toolset += RunPython()

    console.print(
        f"[dim]Using {chat_provider.name} provider with model "
        f"{chat_provider.model_name}.[/dim]"
    )

    if prompt is not None:
        await _run_request(chat_provider, toolset, history, prompt)
        return

    while True:
        user_input = ask_user()
        if user_input is None:
            console.print("[dim]Bye[/dim]")
            break

        history = await _run_request(chat_provider, toolset, history, user_input)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        chat_provider = create_chat_provider(args)
    except Exception as exc:
        parser.error(str(exc))

    try:
        asyncio.run(_main(chat_provider, prompt=args.prompt))
    except KeyboardInterrupt:
        console.print("\n[dim]Bye[/dim]")
