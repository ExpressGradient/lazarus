# Lazarus

Lazarus is a small coding agent built around one idea: an IPython interpreter
can be both the agent's computer and its memory.

The model has two tools:

- `python` runs a cell in a long-lived IPython process.
- `start_new_loop` runs a final handoff cell, discards earlier chat history,
  and continues with the same IPython process.

Both tools default to a 300-second timeout. The model can set `timeout` on any
call when a cell needs more or less time.

The model decides when to start a new loop. Lazarus also steers it toward a
handoff when the current context reaches 250,000 tokens. The handoff cell is
ordinary, free-form Python. It can preserve notes, functions, objects, relevant
file slices, commands, and anything else the next loop needs. There is no
checkpoint schema or helper API.

## Install

Lazarus requires Python 3.12+ and `uv`.

```sh
uv tool install git+https://github.com/ExpressGradient/lazarus
lazarus
```

Run directly from a checkout:

```sh
uv run lazarus
```

Run one request and exit:

```sh
lazarus --prompt "fix the failing tests"
```

Start a fresh context loop earlier, for example at 150,000 tokens:

```sh
lazarus --loop-token-limit 150000
```

Quit an interactive session with `/quit`.

## Providers

Lazarus uses `kosong` and supports Kimi, OpenAI Responses, Anthropic, Google,
and generic OpenAI-compatible Chat Completions APIs. The named providers have
default models; `openai-legacy` requires an explicit model ID.

```sh
lazarus --provider kimi       # kimi-k3
lazarus --provider openai     # gpt-5.6-sol
lazarus --provider anthropic  # claude-opus-5
lazarus --provider google     # gemini-3.7-flash
lazarus --provider openai-legacy --model your-model
```

Set the credentials expected by the chosen provider before running Lazarus.
For `openai-legacy`, set `OPENAI_API_KEY`. Set `OPENAI_BASE_URL` for a compatible
server; if omitted, it uses OpenAI's default endpoint. APIs that return thinking
in an extra message field can set `OPENAI_REASONING_KEY`, such as
`reasoning_content`.

## Execution model

IPython runs in a child process. Requests and results use a private JSON channel,
so Python and subprocess output cannot corrupt the protocol. Standard input is
detached from that channel, output is capped before returning it to the model,
and names remain alive until Lazarus exits or the worker process dies. Concurrent
tool requests are serialized, and each displayed call stays paired with its
result.

When `start_new_loop` succeeds, Lazarus retains only:

1. The original user task.
2. The assistant's handoff tool call.
3. The handoff tool result.

The system prompt and IPython process stay unchanged. The retained tool call
makes the reset explicit, while restoring the original task prevents the goal
from depending on the model's handoff. Lazarus adds no separate reset message.

## Context and token usage

After every model response, Lazarus prints a `LAZARUS_TOKEN_USAGE` JSON record
with cumulative input, cache-read, cache-creation, output, total, and successful
loop-reset counts. This makes long agent runs measurable without changing the
model conversation. Reset counts remain telemetry and are not added to the
system prompt.

Automatic steering uses the size of the latest context, not cumulative billing
usage. Cached and uncached input are counted once, along with the latest output.
At 250,000 tokens by default, Lazarus adds one user message asking the model to
compact its useful state into a handoff and call `start_new_loop`. Change the
threshold with `--loop-token-limit`. A successful reset clears that loop's
steering state while lifetime usage totals continue accumulating.

## Development

```sh
uv run python -m lazarus.cli --help
```

The implementation is intentionally small:

```text
src/lazarus/cli.py            providers, tools, and agent loop
src/lazarus/python_worker.py  persistent IPython worker
```
