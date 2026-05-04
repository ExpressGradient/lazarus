# Lazarus

Lazarus is a terminal coding agent with persistent Python tools and carryover.
It gives the model a long-lived Python workspace, so the agent can inspect
files, edit code, run commands, execute tests, and keep useful state alive while
it works.

The project is intentionally small: the CLI loop lives in
`src/lazarus/cli.py`, and the persistent Python worker lives in
`src/lazarus/python_worker.py`.

## Features

- Chat with a coding agent from your terminal.
- Run against Kimi/Moonshot, OpenAI, Anthropic, or Google providers through
  `kosong`.
- Give the agent a persistent `run_python` tool for workspace actions.
- Preserve Python variables, imports, helper functions, and notes across tool
  calls.
- Automatically inject an internal carryover request when token usage gets
  large, then reset chat history while keeping the Python interpreter alive.
- Stream model output by default when the selected provider supports it.

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- API credentials for whichever provider you use

## Install

Install Lazarus as a `uv` tool:

```sh
uv tool install git+https://github.com/ExpressGradient/lazarus
lazarus
```

Or run it without installing:

```sh
uvx --from git+https://github.com/ExpressGradient/lazarus lazarus
```

Quit the CLI with:

```text
/quit
```

Run a single non-interactive job and exit:

```sh
lazarus --prompt "summarize this repository"
```

## Providers And Models

Lazarus defaults to the Kimi provider with `kimi-k2.6`:

```sh
lazarus
```

Choose a provider and model explicitly:

```sh
lazarus --provider kimi --model kimi-k2.6
lazarus --provider openai --model gpt-5.2
lazarus --provider anthropic --model claude-sonnet-4.6
lazarus --provider google --model gemini-3-pro
```

Supported provider aliases:

```text
anthropic
gemini
google
google-genai
kimi
moonshot
openai
openai-legacy
openai-responses
```

Set credentials using the environment variables expected by the selected
`kosong` provider. For example, Kimi/Moonshot setups commonly use:

```sh
export KIMI_API_KEY=...
export KIMI_BASE_URL=https://api.moonshot.ai/v1
```

## CLI Options

```text
--provider               Provider alias to use. Defaults to kimi.
--model                  Model name passed to the provider. Defaults to kimi-k2.6.
--no-stream              Disable provider streaming.
--thinking-effort        Reasoning effort: off, low, medium, high, xhigh, or max.
--anthropic-max-tokens   Default max_tokens for Anthropic. Defaults to 8192.
--prompt                 Run one non-interactive request and exit.
```

## How The Agent Works

Every user request is added to chat history and sent to the selected model with
the Lazarus system prompt. The model can answer directly or call `run_python`.

`run_python` executes code in a long-lived Python worker process. That means
state survives between tool calls inside the same Lazarus session:

```python
notes = {"current_task": "fix failing tests"}
```

Later tool calls can still read `notes`. This is also how carryover preserves
state after chat history is reset.

The agent is expected to use Python for workspace actions, including:

- reading and writing files
- running shell commands through `subprocess`
- executing tests and linters
- collecting compact notes for later iterations

## Carryover

Lazarus tracks token usage during each user request. When the accumulated input
and output usage for that request reaches `100,000` tokens, Lazarus injects an
internal carryover instruction into the conversation.

That instruction tells the agent to make exactly one `run_python` call that
stores whatever state the next iteration needs. After that tool call finishes,
Lazarus resets chat history to:

```python
[
    original_user_message,
    carryover_tool_call_message,
    carryover_tool_result_message,
]
```

Older user messages, assistant messages, and tool results are removed from chat
history. The Python interpreter process remains alive, so variables and helper
functions created by the carryover cell are still available.

In plain English: carryover keeps the current task moving by replacing a large
chat history with a compact Python handoff.

## Current Limitations

- Carryover is reactive. Lazarus injects the carryover request after token
  usage crosses the threshold during a request; it does not proactively compact
  before sending a large history to the provider.
- The agent has a Python tool, not a full shell tool. Shell commands should be
  run from Python with `subprocess`.
- Workspace changes are made by whatever Python code the model runs, so review
  diffs before committing important work.

## Development

Run from a local checkout while developing:

```sh
uv run lazarus
```

Useful files:

```text
src/lazarus/cli.py            CLI loop, provider setup, carryover behavior
src/lazarus/python_worker.py  Persistent Python execution worker
main.py                       Direct module entrypoint
pyproject.toml                Package metadata and console script
```

Check the package metadata:

```sh
uv run python -m lazarus.cli --help
```
