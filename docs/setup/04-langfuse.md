# 4. Trace the Agent with Local Langfuse

**Goal:** run [Langfuse](https://langfuse.com) on your own computer and see
every step the agent takes: the prompt sent to the LLM, its answer, each tool
call with its arguments and result, timing, tokens and cost.

## Background

- **Langfuse** is an open-source tool for watching LLM apps. It records each
  agent run as a **trace** that you open in a web UI.
- We run our own Langfuse server in **Docker**, so traces stay on your
  computer and it works without internet. It uses about 2.5 GB of RAM while
  running and about 6 GB of disk for the Docker images.
- RAI connects to Langfuse when `use_langfuse = true` in `config.toml` and
  `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. It stays `false`
  in Git, so tracing is opt-in.

!!! info "Why Langfuse server v3 and not v4?"
    `rai-core` requires the Langfuse Python SDK below version 4. The v4
    server shows traces from these older SDKs with a delay, so we pin the
    server to v3, which matches.

## Quick start

You need Docker with the Compose plugin
([install guide](https://docs.docker.com/engine/install/)). Check it works:

```bash
docker compose version
```

**1. Start the server.** The first start downloads the Docker images and
takes several minutes. Later starts take about 30 seconds.

```bash
pixi run langfuse-up
```

This also runs `pixi run langfuse-setup` first, which:

- creates `docker/langfuse/.env` with random passwords for the server, and
- adds `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to your `.env`, so the
  agent can send traces.

**2. Turn tracing on** in `config.toml`:

```toml title="config.toml"
[tracing.langfuse]
use_langfuse = true
host = "http://localhost:3000"
```

**3. Run the agent** as usual (see [Getting Started](../getting-started.md#6-run-the-agent-with-the-fake-robot))
and send it a task.

**4. Open the UI** at <http://localhost:3000> and log in:

- **Email:** `dev@walkie.local`
- **Password:** the `LANGFUSE_INIT_USER_PASSWORD` line in
  `docker/langfuse/.env`

```bash
grep LANGFUSE_INIT_USER_PASSWORD docker/langfuse/.env
```

Open the `walkie-agent-ros` project and click **Tracing**. Each task you sent
is one trace, named `LangGraph`. Click it to see each LLM call and tool call
in order.

## Commands

| Command                   | What it does                                                         |
| ------------------------- | -------------------------------------------------------------------- |
| `pixi run langfuse-up`    | Start the server and wait until the UI is ready                      |
| `pixi run langfuse-down`  | Stop the server. Your traces are kept.                               |
| `pixi run langfuse-logs`  | Show the server logs. Press ++ctrl+c++ to stop.                      |
| `pixi run langfuse-reset` | Delete the server, **all traces** and its passwords, to start fresh. |

After `langfuse-reset`, remove the `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` lines from `.env` before the next `langfuse-up`, so
new keys can be written. The setup script warns you if you forget.

The server keeps running after a reboot until you run `langfuse-down`.

## What is in the repository

| File                         | What it is                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| `docker/langfuse/compose.yml` | The Langfuse server: web UI, worker, Postgres, ClickHouse, Redis and MinIO, in six containers. |
| `scripts/langfuse_setup.py`  | Creates the passwords and API keys. Never overwrites existing ones.                           |
| `docker/langfuse/.env`       | Generated passwords. Ignored by Git.                                                          |
| `tests/conftest.py`          | Turns tracing off during tests, whatever `config.toml` says.                                  |

`compose.yml` is based on the
[official Langfuse file](https://github.com/langfuse/langfuse/blob/v3.225.11/docker-compose.yml),
with these changes:

- All ports listen on `127.0.0.1` only, so other computers on the network
  (for example, competition Wi-Fi) cannot reach the server.
- No default passwords. They come from `docker/langfuse/.env`.
- The `LANGFUSE_INIT_*` settings create the user, project and API keys on
  first start, so there is nothing to click in the UI.
- Usage statistics to Langfuse are turned off.

!!! warning "Do not edit the passwords in `docker/langfuse/.env`"
    The databases store their passwords the first time they start. If you
    change the file afterwards, the server cannot log in to its own
    databases. Use `pixi run langfuse-reset` to start over instead.

## Troubleshooting

**`LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set`** when starting the
agent: tracing is on but `.env` has no keys. Run `pixi run langfuse-setup`,
or set `use_langfuse = false`.

**`port is already allocated`** on `langfuse-up`: another program uses port
3000 or 9090. Stop it, or change the left-hand port in `compose.yml`
(for example `127.0.0.1:3001:3000`), and set `host` in `config.toml` and
`NEXTAUTH_URL` in `compose.yml` to match.

**Agent runs but no traces appear:** check the server is up with
`docker compose --file docker/langfuse/compose.yml ps`, and that `host` in
`config.toml` is `http://localhost:3000`. Traces usually appear within a few
seconds.

**Every LLM call appears twice in a trace:** this is a known RAI 2.12
behavior. RAI adds the Langfuse handler once for the whole run and again for
each LLM call. The LLM is only called once, but the token and cost totals in
Langfuse are doubled.
