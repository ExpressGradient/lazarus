# Lazarus

Lazarus is a small coding agent built around one idea: an IPython interpreter
can be both the agent's computer and its memory.

The model has two tools:

- `python` runs a cell in a long-lived IPython process.
- `start_new_loop` runs a final handoff cell, discards earlier chat history,
  and continues with the same IPython process.

The model decides when to start a new loop. Its handoff cell is ordinary,
free-form Python. It can preserve notes, functions, objects, relevant file
slices, commands, and anything else the next loop needs. There is no checkpoint
schema or helper API.

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

Quit an interactive session with `/quit`.

## Providers

Lazarus uses `kosong` and supports Kimi, OpenAI, Anthropic, Google, and Z.AI.
Each provider has a default model, while `--model` accepts any model ID
supported by that provider.

```sh
lazarus --provider kimi       # kimi-k3
lazarus --provider openai     # gpt-5.6-sol
lazarus --provider anthropic  # claude-opus-5
lazarus --provider google     # gemini-3.7-flash
lazarus --provider zai        # glm-5.2
```

Set the credentials expected by the chosen provider before running Lazarus.
For Z.AI, set `ZAI_API_KEY`. `ZAI_BASE_URL` can override its default general API
endpoint, `https://api.z.ai/api/paas/v4/`.

## Execution model

IPython runs in a child process. Requests and results use a private JSON channel,
so Python and subprocess output cannot corrupt the protocol. Standard input is
detached from that channel, output is capped before returning it to the model,
and names remain alive until Lazarus exits or the worker process dies.

When `start_new_loop` succeeds, Lazarus retains only:

1. A notice explaining that a new loop began.
2. The assistant's handoff tool call.
3. The handoff tool result.

The system prompt and IPython process stay unchanged, so the next loop can use
the cell source and the state it created without rediscovering the project.

## Development

```sh
uv run python -m lazarus.cli --help
```

The implementation is intentionally small:

```text
src/lazarus/cli.py            providers, tools, and agent loop
src/lazarus/python_worker.py  persistent IPython worker
```
