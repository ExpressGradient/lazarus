# Lazarus

Lazarus is a small coding agent whose computer and working memory are the same
long-lived IPython interpreter. It can inspect a repository, edit files, run
commands, keep useful Python objects between steps, and compact long conversations
without throwing away interpreter state.

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```sh
uv tool install git+https://github.com/ExpressGradient/lazarus
cd your-project
lazarus
```

Run from a checkout with `uv run lazarus`. In an interactive session, use `/quit`
to exit and Ctrl-C to stop the current turn without exiting Lazarus.

### Common workflows

```sh
# Complete one task and exit
lazarus --prompt "find the failing tests, fix the cause, and verify the fix"

# Use Codex through an existing ChatGPT login
codex login
lazarus --provider codex --thinking-effort high

# Pin a provider and model
lazarus --provider anthropic --model claude-opus-5

# Keep the journal and logs at a known location, then resume later
lazarus --session-dir ./.lazarus-session
lazarus --resume ./.lazarus-session

# Show complete tool calls and output in the terminal
lazarus --verbose

# Give a large task more context and retain more tool output
lazarus --loop-token-limit 250000 --tool-output-limit-kib 64
```

`--prompt` is useful in scripts: normal replies go to stdout and each model call
emits a machine-readable `LAZARUS_TOKEN_USAGE {...}` line. Interactive mode shows
a compact context and cumulative token summary instead.

Good prompts give the agent an outcome and verification target, not a sequence of
shell commands. For example:

```text
Trace why the API test is flaky, make the smallest safe fix, and run the relevant tests.
Review this branch against main for correctness issues. Do not edit files.
Upgrade the dependency, update affected code, and summarize any behavior changes.
```

## Providers

The default provider is Kimi. Lazarus uses `kosong` and supports:

```sh
lazarus --provider kimi       # default: kimi-k3
lazarus --provider codex      # default: gpt-5.6-sol; requires `codex login`
lazarus --provider openai     # default: gpt-5.6-sol
lazarus --provider anthropic  # default: claude-opus-5
lazarus --provider google     # default: gemini-3.7-flash
lazarus --provider openai-legacy --model your-model
```

Set the credentials required by the selected provider. `openai-legacy` requires
`OPENAI_API_KEY`; `OPENAI_BASE_URL` can point it at an OpenAI-compatible server.
For servers that return reasoning in a separate field, set
`OPENAI_REASONING_KEY` (for example, `reasoning_content`). Use `--model` to
override any provider default and `--thinking-effort` to select `off`, `low`,
`medium`, `high`, `xhigh`, or `max` where supported.

## What the agent can do

The model receives three tools:

- `python` runs code in a persistent IPython worker. Imports, variables, functions,
  and objects survive between calls and context resets.
- `job` reads progress, waits, rereads output from a byte offset, or cancels the
  active cell without blocking the host process.
- `start_new_loop` runs a final handoff cell and replaces old chat history while
  preserving the worker, jobs, logs, and system prompt.

A quick cell can wait up to 60 seconds with `yield_after`; otherwise it immediately
returns a job handle. This is yielding, not cancellation. Cells have a 300-second
default deadline, and only one cell can execute at a time.

Conceptually, model tool calls look like this:

```text
python(code="from pathlib import Path; print(Path('pyproject.toml').read_text())", yield_after=1)
python(code="import subprocess; build = subprocess.Popen([...], stdout=open('build.log', 'w'))")
job(id="...", wait=10)
job(id="...", cursor=0)
job(id="...", cancel=true)
```

For parallel work, the agent starts subprocesses from a short cell, writes their
output to files, and retains their handles. A completed cell does not imply that
those subprocesses have finished. Ordinary asyncio tasks may stop advancing
between cells and should not be used as durable background jobs.

Tool output sent back to the model is capped at 48 KiB by default; the complete
combined stdout/stderr stream remains in the job log. Terminal previews are even
shorter unless `--verbose` is enabled. Use `--tool-output-limit-kib` when a task
needs more output in context.

## Sessions, interruption, and recovery

By default, session data is stored under:

```text
~/.local/state/lazarus/sessions/<timestamp>-<id>/
├── journal.jsonl
└── jobs/*.log
```

The journal is append-only. A complete assistant response is persisted before any
tool call executes, so a dropped model stream cannot dispatch a partial call.
Interrupted or uncertain calls are marked and never replayed automatically.

`--resume DIR` restores the conversation and last working directory, but starts a
**fresh interpreter**. Files and logs survive; Python variables, objects, and old
job handles do not. The original system prompt is retained, including its starting
working directory, local date (`YYYY-MM-DD`), and skill index. A lock prevents two
processes from resuming the same journal.

On Ctrl-C or timeout, Lazarus first interrupts the worker process group. If the
worker recovers, Python state remains available; otherwise Lazarus reports that
state was lost and creates a fresh worker on the next cell. Filesystem writes and
other partial side effects are never rolled back. Session exit stops the worker
and its process group, but deliberately detached processes must be managed
separately.

## Long-context behavior

At 150,000 context tokens, Lazarus asks the model to save useful state in a
handoff and call `start_new_loop`. A successful reset retains only the current
user task, the handoff call, and its result; the IPython worker and on-disk evidence
stay intact. Change the threshold with `--loop-token-limit`.

The system prompt and ordinary history are not rewritten between turns. Keeping
that prefix stable makes provider-side prompt caching possible. The context number
shown in the terminal is the latest model call's input plus output; session token
totals are cumulative for the current process and restart on resume.

## Skills

Lazarus discovers optional `SKILL.md` files globally and from the current project
up to its Git root. Only a compact name, description, and path index enters the
system prompt; the agent reads full instructions from disk when needed.

```sh
# Project-local: .agents/skills/
bunx skills add <repo-or-path> --agent universal

# Global: ~/.agents/skills/
bunx skills add <repo-or-path> --agent universal --global
```

Each skill needs YAML frontmatter with `name` and `description`. The nearest
project definition wins; `disable-model-invocation: true` hides a skill from the
index. The index is fixed when a session starts, so begin a new session after
changing skill metadata.

## Implementation

The request path is deliberately small:

1. `cli.py` builds a stable system prompt and sends history and tool schemas to a
   provider through `kosong`.
2. The full model response is appended to the session journal.
3. `jobs.py` dispatches cells to one supervised IPython child in `runtime.py`.
4. A private JSON channel carries bounded control metadata; cell stdout/stderr goes
   directly to a log, so arbitrary output cannot corrupt the protocol.
5. Tool results are appended to history and generation continues until the model
   stops calling tools or explicitly starts a new context loop.

```text
src/lazarus/cli.py            CLI, providers, tools, and agent loop
src/lazarus/jobs.py           job lifecycle, output limits, and cancellation
src/lazarus/runtime.py        IPython worker supervision and recovery
src/lazarus/python_worker.py  cell execution and live output capture
src/lazarus/session.py        append-only journal and resume logic
src/lazarus/skills.py         skill discovery and prompt catalog
```

## Development

```sh
uv run python -m lazarus.cli --help
uv run python -m unittest discover -s tests
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright --pythonpath .venv/bin/python src
```

## License

MIT
