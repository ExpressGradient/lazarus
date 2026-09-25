# Lazarus

Lazarus is a small coding agent built around one idea: an IPython interpreter
can be both the agent's computer and its memory.

The model has three tools:

- `python` starts a cell in a long-lived IPython process and returns a job handle
  immediately. Set `yield_after` (0–60 seconds) to wait briefly for its result.
- `job` reads progress, waits, or requests cancellation outside the interpreter.
- `start_new_loop` runs a final handoff cell, discards earlier chat history,
  and continues with the same IPython process. It requires an idle interpreter
  and waits for its own cell to finish.

Both cell tools default to a 300-second execution deadline, configurable with
`timeout`. Yielding or ending a status wait does **not** cancel execution.
Cancellation and deadlines first interrupt the worker's process group. Surviving
children in that group are killed. Interpreter state survives if recovery and
child cleanup succeed; otherwise the group is stopped and a fresh worker starts
on the next call. Partial file writes and other side effects are never rolled back.

The model decides when to start a new loop. Lazarus also steers it toward a
handoff when the current context reaches 150,000 tokens. The handoff cell is
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

Set a different context-loop threshold, for example 250,000 tokens:

```sh
lazarus --loop-token-limit 250000
```

Change the maximum tool output kept in context, for example to 64 KiB:

```sh
lazarus --tool-output-limit-kib 64
```

Quit an interactive session with `/quit`.

## Providers

Lazarus uses `kosong` and supports Kimi, OpenAI Responses, Codex subscription
usage, Anthropic, Google, and generic OpenAI-compatible Chat Completions APIs.
The named providers have default models; `openai-legacy` requires an explicit
model ID.

```sh
lazarus --provider kimi       # kimi-k3
lazarus --provider codex      # gpt-5.6-sol, uses `codex login`
lazarus --provider openai     # gpt-5.6-sol
lazarus --provider anthropic  # claude-opus-5
lazarus --provider google     # gemini-3.7-flash
lazarus --provider openai-legacy --model your-model
```

Set the credentials expected by the chosen provider before running Lazarus.
For `codex`, run `codex login`; usage counts against that ChatGPT subscription.
For `openai-legacy`, set `OPENAI_API_KEY`. Set `OPENAI_BASE_URL` for a compatible
server; if omitted, it uses OpenAI's default endpoint. APIs that return thinking
in an extra message field can set `OPENAI_REASONING_KEY`, such as
`reasoning_content`.

## Execution model

IPython runs in a child process. Requests and results use a private JSON channel,
so Python and subprocess output cannot corrupt the protocol. Standard input is
detached from that channel. Names, functions, imports, and objects survive calls
and context resets, until the worker or session exits.

One cell runs at a time. A second cell or handoff receives a busy error without
executing. Job observation stays responsive because it runs in the host process.
For parallel work, the model can launch subprocesses from a short cell, redirect
output to explicit files, and retain their handles. A completed cell does not
mean those subprocesses finished. Ordinary asyncio tasks may stop advancing
between cells; they are not durable background jobs.

The model is encouraged to compose operations, write useful wrappers, batch work,
cache results, and inspect data with Python. Large objects stay in memory; only
useful evidence needs to enter the conversation.

For example, these are tool arguments, not functions injected into IPython:

```text
python(code="import time; print('started', flush=True); time.sleep(20); print('done')")
# Returns {"job_id": "...", "status": "running", ...}

job(id="...", wait=10)       # New output, status, and a byte cursor
job(id="...", cursor=0)      # Reread from the start without rerunning code
job(id="...", cancel=true)   # Request interruption; wait/read to confirm completion
job()                       # List retained jobs and their log paths
```

Each job reports its state, elapsed time, working directory, interpreter generation,
and log path. Standard output and standard error share a live log in arrival order.
Subprocesses must flush their output for immediate progress visibility. Automatic
reads advance a cursor; explicit cursor reads leave it unchanged. Output is capped
at 48 KiB by default, retaining the first third and final two thirds when truncated.
Complete logs stay on disk. The most recent 20 observed jobs are retained in memory;
unobserved results are never evicted.

Completion notices are appended once between model turns. A result already read
through `job` needs no extra notice. When the model stops calling tools while a
cell is still running, Lazarus waits for completion and gives the result back to
the model before ending the request. Session exit stops the worker and its process
group. Processes deliberately detached into their own groups must be managed by
the model.

When `start_new_loop` succeeds, Lazarus retains only:

1. The original user task for the current request.
2. The assistant's handoff tool call.
3. The handoff tool result.

The system prompt, interpreter, jobs, and logs stay unchanged. The retained tool
call makes the reset explicit. Ordinary turns only append messages; completion
notices and working-directory changes do not rewrite the system prompt or earlier
messages. This preserves a stable prefix for provider-side caching between resets;
actual cache use depends on the provider.

## Sessions and recovery

Each session prints its directory under `~/.local/state/lazarus/sessions/`.
Its append-only `journal.jsonl` records messages, calls, job outcomes, and context
resets. Job logs live in its `jobs/` directory. These files persist after exit;
delete old session directories when no longer needed.

```sh
lazarus --session-dir ./my-session --prompt "fix the failing tests"
lazarus --resume ./my-session
```

Resume restores the conversation and task with a **fresh interpreter**. It never
replays cells or restores live Python objects. It explicitly marks missing tool
results as unknown and tells the model to inspect files, logs, and any surviving
processes before retrying. Only a partial final journal write is trimmed during
recovery. A session lock prevents two agents from resuming the same journal.

## Context and token usage

Interactive terminal sessions use `You:` and `Lazarus:` labels and show one
compact session-token summary after each completed reply. Empty input is ignored.

With `--prompt` or redirected input/output, each model response still produces a
`LAZARUS_TOKEN_USAGE` JSON record with cumulative input, cache-read, cache-creation,
output, total, and successful loop-reset counts. This makes long agent runs
measurable without changing the
model conversation. Reset counts remain telemetry and are not added to the
system prompt.

Automatic steering uses the size of the latest context, not cumulative billing
usage. Cached and uncached input are counted once, along with the latest output.
At 150,000 tokens by default, Lazarus adds one user message asking the model to
compact its useful state into a handoff and call `start_new_loop`. Change the
threshold with `--loop-token-limit`. A successful reset clears that loop's
steering state while lifetime usage totals continue accumulating.

## Development

```sh
uv run python -m lazarus.cli --help
ruff check src
ruff format --check src
pyright --pythonpath .venv/bin/python src
```

The implementation is intentionally small:

```text
src/lazarus/cli.py            providers, tools, and agent loop
src/lazarus/jobs.py           yielding jobs, progress, and cancellation
src/lazarus/runtime.py        worker supervision and recovery
src/lazarus/python_worker.py  persistent IPython worker and live output
src/lazarus/session.py        append-only journal and conversation recovery
```

## License

MIT
