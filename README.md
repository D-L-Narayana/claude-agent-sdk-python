# Claude Agent SDK for Python

Python SDK for Claude Agent. See the [Claude Agent SDK documentation](https://platform.claude.com/docs/en/agent-sdk/python) for more information.

## Installation

```bash
pip install claude-agent-sdk
```

**Prerequisites:**

- Python 3.10+

**Note:** The Claude Code CLI is automatically bundled with the package - no separate installation required! The SDK will use the bundled CLI by default. If you prefer to use a system-wide installation or a specific version, you can:

- Install Claude Code separately: `curl -fsSL https://claude.ai/install.sh | bash`
- Specify a custom path: `ClaudeAgentOptions(cli_path="/path/to/claude")`

## Quick Start

```python
import anyio
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Basic Usage: query()

`query()` is an async function for querying Claude Code. It returns an `AsyncIterator` of response messages. See [src/claude_agent_sdk/query.py](src/claude_agent_sdk/query.py).

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

# Simple query
async for message in query(prompt="Hello Claude"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)

# With options
options = ClaudeAgentOptions(
    system_prompt="You are a helpful assistant",
    max_turns=1
)

async for message in query(prompt="Tell me a joke", options=options):
    print(message)
```

### Using Tools

By default, Claude has access to the full [Claude Code toolset](https://code.claude.com/docs/en/settings#tools-available-to-claude) (Read, Write, Edit, Bash, and others). `allowed_tools` is a permission allowlist: listed tools are auto-approved, and unlisted tools fall through to `permission_mode` and `can_use_tool` for a decision. It does not remove tools from Claude's toolset. To block specific tools, use `disallowed_tools`. See the [permissions guide](https://platform.claude.com/docs/en/agent-sdk/permissions) for the full evaluation order.

```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],  # auto-approve these tools
    permission_mode='acceptEdits'  # auto-accept file edits
)

async for message in query(
    prompt="Create a hello.py file",
    options=options
):
    # Process tool use and results
    pass
```

### Working Directory

```python
from pathlib import Path

options = ClaudeAgentOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### System Prompt

By default, Claude Code builds the system prompt on a session's first request, records it, and reuses it on every later request, including after you resume the session. A changed custom prompt, or changed `append` text on the `claude_code` preset, then has no effect until the session is compacted or you start a new session. To rebuild the prompt on every request instead, for example while you iterate on its wording, set `snapshot` to `False` in the `{"type": "preset", ...}` or `{"type": "custom", ...}` dict:

```python
options = ClaudeAgentOptions(
    system_prompt={"type": "custom", "prompt": "You are a release bot.", "snapshot": False}
)
```

Requires Claude Code CLI 2.1.257 or later. Before 2.1.265, a session with an `append` or custom prompt recorded it only when `snapshot` was True. See [Modifying system prompts](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts#change-the-prompt-of-an-existing-session) for details.

### Structured output

Pass a JSON Schema as `output_format` and the run ends with a result that matches it. The parsed value is on `ResultMessage.structured_output`; it is `None` when the run failed or the CLI returned no structured result.

```python
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

schema = {
    "type": "object",
    "properties": {
        "file_count": {"type": "integer"},
        "has_tests": {"type": "boolean"},
    },
    "required": ["file_count", "has_tests"],
}

options = ClaudeAgentOptions(
    output_format={"type": "json_schema", "schema": schema},
    allowed_tools=["Glob", "Grep"],
)

async for message in query(
    prompt="Count the Python files under src/ and check whether there are tests.",
    options=options,
):
    if isinstance(message, ResultMessage) and not message.is_error:
        report = message.structured_output  # dict matching the schema
        print(report["file_count"], report["has_tests"])
```

The schema dict follows the Messages API `json_schema` output format. Claude may use tools while producing the answer, so `allowed_tools` and `permission_mode` apply as usual.

## ClaudeSDKClient

`ClaudeSDKClient` supports bidirectional, interactive conversations with Claude
Code. See [src/claude_agent_sdk/client.py](src/claude_agent_sdk/client.py).

Unlike `query()`, `ClaudeSDKClient` additionally enables **custom tools** and **hooks**, both of which can be defined as Python functions.

While connected you can also steer the session: `interrupt()`, `set_model()`, `set_permission_mode()`, `rewind_files()`, `stop_task()`, `get_mcp_status()`, `toggle_mcp_server()`, `reconnect_mcp_server()`, `get_context_usage()` and `get_server_info()`. The `session_id` property reports the session id of the current conversation once the CLI has announced it, and `cli_version` the version of the Claude Code binary the client is talking to.

When a [session store](#session-stores) is configured, `mirror_stats` returns a `MirrorStats` snapshot (importable from `claude_agent_sdk`) of the transcript mirror's lifetime counters — `frames_enqueued`, `entries_enqueued`, `flushes`, `entries_flushed`, `batches_failed`, `entries_dropped` — and `None` otherwise (and before `connect()` / after `disconnect()`). A non-zero `batches_failed` or `entries_dropped` means the mirror gave up on at least one batch (each was also surfaced as a `MirrorErrorMessage`), so the store may be missing entries — after a timed-out attempt the write may still have landed — and an idempotent `sync_session_to_store()` catch-up is warranted.

### Custom Tools (as In-Process SDK MCP Servers)

A **custom tool** is a Python function that you can offer to Claude, for Claude to invoke as needed.

Custom tools are implemented in-process MCP servers that run directly within your Python application, eliminating the need for separate processes that regular MCP servers require.

For an end-to-end example, see [MCP Calculator](examples/mcp_calculator.py).

#### Creating a Simple Tool

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient

# Define a tool using the @tool decorator
@tool("greet", "Greet a user", {"name": str})
async def greet_user(args):
    return {
        "content": [
            {"type": "text", "text": f"Hello, {args['name']}!"}
        ]
    }

# Create an SDK MCP server
server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",
    tools=[greet_user]
)

# Use it with Claude. allowed_tools pre-approves the tool so it runs
# without a permission prompt; it does not control tool availability.
options = ClaudeAgentOptions(
    mcp_servers={"tools": server},
    allowed_tools=["mcp__tools__greet"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")

    # Extract and print response
    async for msg in client.receive_response():
        print(msg)
```

#### Tool input types

`@tool` turns a dict of Python types (or a `TypedDict` class) into the JSON Schema Claude sees and arguments are validated against before your function runs; invalid input is reported back to Claude as an error result instead of raising. A full JSON Schema dict passes through untouched.

Supported hints: `str`, `int`, `float`, `bool`, `None`; `list[X]`, `tuple[X, ...]`, `set[X]` (`uniqueItems`), fixed `tuple[A, B]` (`prefixItems`); `dict[str, X]` (`additionalProperties`); `X | None` / `Optional[X]` (`anyOf` with `null`, so a `None` argument validates); `Literal[...]` and `enum.Enum` subclasses (`enum`); nested `TypedDict`s with `NotRequired` / `Required` / `ReadOnly`; `datetime`, `date`, `time`, `UUID` (string with `format`), `Decimal` (number); `Annotated[X, "description"]` at any nesting to attach a description; `Any` accepts anything. Anything else is sent as a string.

#### Bringing a FastMCP / MCPServer server

`create_sdk_mcp_server()` is a convenience. Servers written with mcp's high-level class work as SDK servers too — `FastMCP` on mcp 1.x, `MCPServer` on mcp 2.x — alongside any lowlevel `mcp.server.Server`: pass the instance as `instance` of an `{"type": "sdk", ...}` entry (or an `McpSdkServerConfig`) and the SDK unwraps the low-level server it drives, so tools declared with `@server.tool()` reach Claude unchanged.

```python
from mcp.server.mcpserver import MCPServer  # mcp 2.x; on 1.x: from mcp.server.fastmcp import FastMCP

from claude_agent_sdk import ClaudeAgentOptions

server = MCPServer("notes")

@server.tool()
def add_note(text: str) -> str:
    return f"saved: {text}"

options = ClaudeAgentOptions(
    mcp_servers={"notes": {"type": "sdk", "name": "notes", "instance": server}},
    allowed_tools=["mcp__notes__add_note"],
)
```

Anything else passed as `instance` is rejected before the CLI starts: `query()` / `ClaudeSDKClient.connect()` raise `ValueError` (`ClaudeAgentOptions.mcp_servers['name']: ...`) whose message names what was received and what is accepted.

#### MCP server logs

Log messages an SDK MCP server sends to its client (`ctx.info(...)` / `ctx.warning(...)` in a FastMCP/MCPServer tool, `session.send_log_message(...)` on a lowlevel server) are forwarded to the Python logger `claude_agent_sdk.mcp.<server name>` — `debug` → DEBUG, `info`/`notice` → INFO, `warning` → WARNING, `error` → ERROR, `critical`/`alert`/`emergency` → CRITICAL — with the server's own logger name in the message and the raw fields on the record (`mcp_server`, `mcp_level`, `mcp_logger`, `mcp_data`). Progress notifications are logged there at DEBUG.

```python
import logging

logging.getLogger("claude_agent_sdk.mcp").setLevel(logging.INFO)
```

#### Benefits Over External MCP Servers

- **No subprocess management** - Runs in the same process as your application
- **Better performance** - No IPC overhead for tool calls
- **Simpler deployment** - Single Python process instead of multiple
- **Easier debugging** - All code runs in the same process
- **Type safety** - Direct Python function calls with type hints

#### Migration from External Servers

```python
# BEFORE: External MCP server (separate process)
options = ClaudeAgentOptions(
    mcp_servers={
        "calculator": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "calculator_server"]
        }
    }
)

# AFTER: SDK MCP server (in-process)
from my_tools import add, subtract  # Your tool functions

calculator = create_sdk_mcp_server(
    name="calculator",
    tools=[add, subtract]
)

options = ClaudeAgentOptions(
    mcp_servers={"calculator": calculator}
)
```

#### Mixed Server Support

You can use both SDK and external MCP servers together:

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "internal": sdk_server,      # In-process SDK server
        "external": {                # External subprocess server
            "type": "stdio",
            "command": "external-server"
        }
    }
)
```

### Hooks

A **hook** is a Python function that the Claude Code _application_ (_not_ Claude) invokes at specific points of the Claude agent loop. Hooks can provide deterministic processing and automated feedback for Claude. Read more in [Intercept and control agent behavior with hooks](https://platform.claude.com/docs/en/agent-sdk/hooks).

For more examples, see examples/hooks.py.

#### Example

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

async def check_bash_command(input_data, tool_use_id, context):
    tool_name = input_data["tool_name"]
    tool_input = input_data["tool_input"]
    if tool_name != "Bash":
        return {}
    command = tool_input.get("command", "")
    block_patterns = ["foo.sh"]
    for pattern in block_patterns:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Command contains invalid pattern: {pattern}",
                }
            }
    return {}

options = ClaudeAgentOptions(
    allowed_tools=["Bash"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command]),
        ],
    }
)

async with ClaudeSDKClient(options=options) as client:
    # Test 1: Command with forbidden pattern (will be blocked)
    await client.query("Run the bash command: ./foo.sh --help")
    async for msg in client.receive_response():
        print(msg)

    print("\n" + "=" * 50 + "\n")

    # Test 2: Safe command that should work
    await client.query("Run the bash command: echo 'Hello from hooks example!'")
    async for msg in client.receive_response():
        print(msg)
```

The `hookEventName` in `hookSpecificOutput` must name the event the hook is registered for (`"PreToolUse"` above, `"UserPromptSubmit"` for a prompt hook, and so on). Matchers registered on the same event are dispatched concurrently by the CLI, so design each hook to be independent.

## Cancellation


Hooks and the `can_use_tool` callback receive a cooperative cancellation signal: `context.signal` on the `ToolPermissionContext` passed to `can_use_tool`, and `context["signal"]` in a hook's `HookContext`. It is an `AbortSignal` (exported from `claude_agent_sdk`) and is aborted when Claude Code abandons the request it was asking about — it sends a cancel for the pending hook or permission prompt (reason `"cancelled by Claude Code"`) — and when the query closes while the callback is still running (reason `"query closed"`).

```python
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny

async def can_use_tool(tool_name, input_data, context):
    signal = context.signal  # AbortSignal

    def on_abort(sig):
        print(f"permission prompt for {tool_name} abandoned: {sig.reason}")

    signal.on_abort(on_abort)  # fires immediately if already aborted

    decision = await ask_reviewer(tool_name, input_data)  # your own slow check
    if signal.aborted:
        return PermissionResultDeny(message="request was cancelled")
    return PermissionResultAllow() if decision else PermissionResultDeny()

options = ClaudeAgentOptions(can_use_tool=can_use_tool)
```

`signal.aborted` and `signal.reason` can be polled; `await signal.wait()` returns once the signal is aborted and is itself cancellable, which makes it convenient to race against your own work. The SDK also cancels the task running the callback when the CLI cancels the request, so long-running callbacks should still expect a cancellation exception at any `await`. An exception raised by a hook or callback is reported back to the CLI as `"<ExceptionType>: <message>"` and does not end the session.

## Session management

Every run is recorded as a JSONL transcript under the Claude Code projects directory (`~/.claude/projects/<project>/<session_id>.jsonl`, or `$CLAUDE_CONFIG_DIR/projects/...`). The SDK ships synchronous helpers for reading and editing those transcripts without starting the CLI; each takes an optional `directory=` (the project directory, with the same semantics as `ClaudeAgentOptions.cwd`) and otherwise searches every project.

```python
from claude_agent_sdk import (
    delete_session,
    fork_session,
    get_session_info,
    get_session_messages,
    get_subagent_messages,
    list_sessions,
    list_subagents,
    rename_session,
    tag_session,
)

project = "/path/to/project"

# Newest first; use limit/offset to paginate. include_worktrees=False skips git worktrees.
for info in list_sessions(directory=project, limit=20):
    print(info.session_id, info.summary, info.last_modified, info.tag)

session_id = list_sessions(directory=project, limit=1)[0].session_id
info = get_session_info(session_id, directory=project)  # one session's metadata

# user/assistant messages in conversation order; .message is the raw API message dict
for message in get_session_messages(session_id, directory=project):
    print(message.type, message.uuid, message.message["content"])

# Subagent transcripts; each message carries the parent_tool_use_id that spawned the agent
for agent_id in list_subagents(session_id, directory=project):
    for message in get_subagent_messages(session_id, agent_id, directory=project):
        print(agent_id, message.parent_tool_use_id, message.type)

rename_session(session_id, "Release notes draft", directory=project)
tag_session(session_id, "release", directory=project)   # tag_session(id, None) clears it
fork = fork_session(session_id, directory=project, up_to_message_id=message.uuid)
print(fork.session_id)                                   # a new session with fresh UUIDs
delete_session(fork.session_id, directory=project)      # removes the transcript and its subagents
```

To continue a recorded session, pass its id as `ClaudeAgentOptions(resume=session_id)` (add `fork_session=True` to branch into a new session id, and `resume_session_at=<message uuid>` to truncate at an earlier point), or `continue_conversation=True` for the most recent session in the working directory.

`to_sdk_message(message)` converts a `SessionMessage` into the same `UserMessage` / `AssistantMessage` objects a live stream yields (with typed content blocks), or `None` when the stored payload cannot be parsed, so one rendering path can serve both history and live output.

```python
from claude_agent_sdk import AssistantMessage, TextBlock, get_session_messages, to_sdk_message

for stored in get_session_messages(session_id, directory=project):
    message = to_sdk_message(stored)
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
```

### Moving sessions between disk and a store

When you use a [session store](#session-stores), three helpers move transcripts between the local projects directory and the store:

- `await import_session_to_store(session_id, store, directory=..., include_subagents=True)` replays an on-disk session (and its subagent transcripts, whose `.meta.json` sidecars become `agent_metadata` entries) into `store.append()` in batches of 500 entries / 1 MiB (`batch_size=` overrides the count). The destination `project_key` is the same one the live mirror uses, so the imported session is resumable with `ClaudeAgentOptions(session_store=store, resume=session_id)` from the original working directory. Adapters should treat `entry["uuid"]` as an idempotency key; a malformed JSONL line is skipped with a warning rather than aborting the import.
- `await sync_session_to_store(session_id, store, directory=...)` is the idempotent catch-up: it loads what the store already holds, appends only the entries that are missing (by `uuid`, or by deep equality for entries without one), in file order after what the store already holds — store readers rebuild the conversation from `parentUuid`, so storage order does not matter — and returns a `SessionSyncReport` with `appended` / `skipped` counts for the main transcript and a `subagents` map of `(appended, skipped)` per subagent subpath (the `.meta.json` sidecar counts as one entry there and is re-appended only when it changed). Run it after a `MirrorErrorMessage` told you a live-mirror batch was dropped; a second run reports `appended == 0`.
- `await export_session_from_store(store, session_id, directory=..., overwrite=False)` is the inverse: it writes the stored session (plus subagent transcripts and their `.meta.json` sidecars) into the real projects directory as `<projects_dir>/<project_key>/<session_id>.jsonl` and returns that `Path`, so the plain CLI can `claude --resume <session_id>` from that directory and `get_session_messages()` / `list_sessions()` can read it. It raises `FileExistsError` unless `overwrite=True`, `ValueError` for a non-UUID id and `FileNotFoundError` when the store has no entries for the session. The check holds against concurrent writers too: with `overwrite=False` the files are committed with exclusive creates, so a transcript another process writes while the export is still reading the store is never replaced — the export fails with `FileExistsError` and removes only what it created. Pass `overwrite=True` to replace the main transcript and the subagent files the store holds (files the store does not know about are left in place).

```python
from claude_agent_sdk import export_session_from_store, sync_session_to_store

report = await sync_session_to_store(session_id, store, directory=project)
print(report.appended, report.skipped, report.subagents)

path = await export_session_from_store(store, session_id, directory=project)
print(path)  # .../projects/<project_key>/<session_id>.jsonl
```

## Session stores

A `SessionStore` mirrors transcripts to storage you control — object storage, a database, another service — while the CLI keeps writing its local files. Pass an instance as `ClaudeAgentOptions(session_store=store)`:

```python
from claude_agent_sdk import ClaudeAgentOptions, InMemorySessionStore, MirrorErrorMessage, query

store = InMemorySessionStore()  # for tests and development; data is lost at exit

async for message in query(prompt="Hello", options=ClaudeAgentOptions(session_store=store)):
    if isinstance(message, MirrorErrorMessage):
        # A batch could not be mirrored after retries; the local transcript is intact.
        print("mirror gap:", message.key, message.error)
```

- Every transcript line is passed to `store.append()` **after** the CLI's local write succeeds. `session_store_flush="batched"` (the default) coalesces entries and flushes once per turn or when 500 entries / 1 MiB are pending; `"eager"` flushes in the background after every frame for near-real-time delivery. Either way a slow adapter never stalls message streaming.
- Failed batches are retried and then dropped with a `MirrorErrorMessage` (a `SystemMessage` with `subtype == "mirror_error"` and `key` / `error` fields) in the message stream. The conversation continues; use `sync_session_to_store` (above) to repair the gap.
- `resume=session_id` with a `session_store` set materializes the session from the store when the local transcript is absent, so a conversation can be resumed on another machine. `load_timeout_ms` (default 60 000) bounds each `store.load()` / `list_subkeys()` call during that step.

### SQLiteSessionStore


For a durable store without any third-party client, use the SQLite store shipped in `claude_agent_sdk.stores`. It is built on the standard-library `sqlite3` module, holds one connection, runs every operation off the event loop, and uses WAL mode for file databases:

```python
from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.stores import SQLiteSessionStore

async with SQLiteSessionStore("sessions.db") as store:
    async for message in query(prompt="Hello", options=ClaudeAgentOptions(session_store=store)):
        print(message)
```

`SQLiteSessionStore(path, *, table_prefix="claude_session")` accepts `":memory:"` as the path; `table_prefix` names the two tables it creates (`<prefix>_entries`, `<prefix>_summaries`) and must be a plain identifier. Parent directories are not created implicitly (a missing directory raises `sqlite3.OperationalError`). Call `await store.aclose()` when you are not using it as an async context manager; every operation after `aclose()` raises `RuntimeError`. It implements every optional `SessionStore` method, including `list_session_summaries`, so `list_sessions_from_store()` is served from the store's summary sidecars — no per-session `load()` calls.

### Reading and editing stored sessions

Each disk helper has an async, store-backed counterpart that takes the store first and derives the `project_key` from `directory=` (default: the current working directory): `list_sessions_from_store`, `get_session_info_from_store`, `get_session_messages_from_store`, `list_subagents_from_store`, `get_subagent_messages_from_store`, `rename_session_via_store`, `tag_session_via_store`, `delete_session_via_store` and `fork_session_via_store`.

```python
from claude_agent_sdk import fork_session_via_store, list_sessions_from_store

for info in await list_sessions_from_store(store, directory=project, limit=10):
    print(info.session_id, info.summary)

fork = await fork_session_via_store(store, session_id, directory=project)
```

### Writing your own store

Implement the `SessionStore` protocol from `claude_agent_sdk`: `append(key, entries)` and `load(key)` are required; `list_sessions`, `list_session_summaries`, `delete` and `list_subkeys` are optional and the SDK probes for them at runtime (a duck-typed class works; subclassing `SessionStore` is optional). Keys are `{"project_key", "session_id"}` plus a `subpath` for subagent transcripts. Validate an adapter with the shipped conformance harness, which asserts the behavioral contracts every adapter must meet and skips the ones for optional methods you do not implement. If your store defines `aclose()`, the harness calls it on every store it creates — including when a contract fails:

```python
import pytest
from claude_agent_sdk.testing import run_session_store_conformance

@pytest.mark.anyio
async def test_my_store_conformance():
    await run_session_store_conformance(lambda: MyStore(...))
```

Reference adapters for S3 (`boto3`), Redis (`redis.asyncio`) and Postgres (`asyncpg`) — with notes on retention, key schemes and the tests that exercise them — live in [examples/session_stores/README.md](examples/session_stores/README.md). They are copy-into-your-project code, not part of the package.

## Observability

- **CLI stderr.** `ClaudeAgentOptions(stderr=callback)` delivers every line the Claude Code process writes to stderr to your callback (see [examples/stderr_callback_example.py](examples/stderr_callback_example.py)). Without a callback the SDK re-emits the CLI's stderr to your process's `sys.stderr`, and keeps a bounded tail of the most recent lines: when the CLI exits non-zero the `ProcessError` carries it in `.stderr`, and a `CLIConnectionError` raised because the CLI died while connecting includes it in its message, so an authentication or startup failure shows its real cause instead of only an exit code. Set `CLAUDE_AGENT_SDK_INHERIT_STDERR=1` to make the subprocess inherit the parent's stderr file descriptor directly instead (no tail is captured then).
- **Distributed tracing.** `pip install "claude-agent-sdk[otel]"` installs `opentelemetry-api`. When it is importable and a span is active at connect time, the SDK injects the W3C `traceparent` / `tracestate` of that span into the CLI's environment (`TRACEPARENT` / `TRACESTATE`) so the CLI's own spans parent under your trace. Values you set explicitly in `ClaudeAgentOptions.env` always win, and the injection is best-effort: it never fails a connect.
- **Hook lifecycle events.** `include_hook_events=True` adds `HookEventMessage`s to the stream — a `SystemMessage` subclass whose `subtype` is `"hook_started"` or `"hook_response"`, with `hook_event_name` (e.g. `"PreToolUse"`) and the raw payload in `data`.
- **Partial messages.** `include_partial_messages=True` adds a `StreamEvent` per API stream event (`event` is the raw Anthropic stream event dict; `uuid`, `session_id` and `parent_tool_use_id` identify where it belongs), so you can render text as it is generated. See [examples/include_partial_messages.py](examples/include_partial_messages.py).
- **SDK MCP server logs.** Log messages an in-process MCP server sends to its client are forwarded to the Python logger `claude_agent_sdk.mcp.<server name>` at the matching level, with the raw fields (`mcp_server`, `mcp_level`, `mcp_logger`, `mcp_data`) on the record; progress notifications are logged there at DEBUG. See [MCP server logs](#mcp-server-logs).
- **CLI version.** The SDK probes `claude -v` when it connects (set `CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK=1` to skip it) and warns about versions older than the minimum it supports. The probe result is cached per binary and exposed as `ClaudeSDKClient.cli_version`.

## Types

See [src/claude_agent_sdk/types.py](src/claude_agent_sdk/types.py) for complete type definitions:

- `ClaudeAgentOptions` - Configuration options
- `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
- `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks
- `StreamEvent`, `RateLimitEvent`, `ConversationResetMessage` - Stream, rate-limit and `/clear` events
- `TaskStartedMessage`, `TaskProgressMessage`, `TaskUpdatedMessage`, `TaskNotificationMessage` - Background task lifecycle (subclasses of `SystemMessage`)
- `HookEventMessage`, `MirrorErrorMessage` - Hook lifecycle events and session-store mirror failures (subclasses of `SystemMessage`)
- `ThinkingBlock`, `ServerToolUseBlock`, `ServerToolResultBlock` - Thinking and server-side tool blocks
- `SDKSessionInfo`, `SessionMessage`, `SessionStore`, `SessionKey` - Session listing and session-store types

Lifecycle frames the CLI emits are typed as well, each a `SystemMessage` subclass so existing `isinstance(msg, SystemMessage)` checks still match and `data` still holds the raw frame: `InitMessage` (`subtype == "init"`: `session_id`, `model`, `cwd`, `tools`, `mcp_servers`, `permission_mode`, `slash_commands`, `agents`, `skills`, `plugins`, `claude_code_version`, ...), `CompactBoundaryMessage` (`trigger`, `pre_tokens`) and `StatusMessage` (`status`, `permission_mode`). Top-level `tool_progress`, `tool_use_summary` and `auth_status` frames become `ToolProgressMessage` (`tool_use_id`, `tool_name`, `elapsed_time_seconds`), `ToolUseSummaryMessage` and `AuthStatusMessage`. Content blocks the SDK does not model are no longer dropped: they arrive as `UnknownBlock(type, data)`, redacted thinking as `RedactedThinkingBlock`, and every server-side `*_tool_result` block as `ServerToolResultBlock`. Code that matches exhaustively on the `Message` or `ContentBlock` unions should add arms for these.

## Error Handling

```python
from claude_agent_sdk import (
    ClaudeSDKError,      # Base error
    CLINotFoundError,    # Claude Code not installed
    CLIConnectionError,  # Connection issues
    ProcessError,        # Process failed
    ResultError,         # Run ended with an error result (subclass of ProcessError)
    CLIJSONDecodeError,  # JSON parsing issues
)

try:
    async for message in query(prompt="Hello"):
        pass
except CLINotFoundError:
    print("Please install Claude Code")
except ResultError as e:
    # The CLI reported a terminal error result (also yielded as the final
    # ResultMessage) and exited. Structured fields are on the exception.
    print(f"Run failed: {e.subtype=} {e.terminal_reason=} {e.api_error_status=}")
    print(e.result or e.errors)
except ProcessError as e:
    print(f"Process failed with exit code: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"Failed to parse response: {e}")
```

All SDK exceptions derive from `ClaudeSDKError`:

| Exception | Raised when | Attributes |
| --- | --- | --- |
| `CLIConnectionError` | The CLI could not be started or died before the session was established, or a write hit a terminated process. | message (includes the recent stderr tail when the CLI died during connect) |
| `CLINotFoundError` (subclass of `CLIConnectionError`) | No `claude` binary was found (bundled, on `PATH`, or at `cli_path`). | message names the path that was tried |
| `ProcessError` | The CLI exited non-zero. | `exit_code`, `stderr` (recent stderr tail) |
| `ResultError` (subclass of `ProcessError`) | The CLI ended the run with an error `result` frame (max turns, budget, API error, ...). | `subtype`, `errors`, `result`, `api_error_status`, `terminal_reason`, `session_id`, `data`, `exit_code` |
| `CLIJSONDecodeError` | A stdout line was not valid JSON. | `line`, `original_error` |
| `MessageParseError` | A frame had a recognized `type` but a shape the parser could not read (for example a required field missing). | `data` (the raw frame) |
| `ControlRequestError` | The CLI answered an SDK control request (`interrupt`, `set_model`, `get_mcp_status`, ...) with an error, or the request could not be delivered. | `subtype` (the request's subtype), `request_id` |
| `ControlRequestTimeoutError` (subclass of `ControlRequestError`) | No response to a control request arrived in time; the message starts with `Control request timeout:`. | `timeout` (seconds), plus the inherited fields |

Invalid option combinations — `resume` together with `continue_conversation`, `session_id` with `resume`/`continue_conversation` without `fork_session`, `resume_session_at` without `resume`, `resume_drops_turn` without `resume_session_at`, a negative `max_turns` or `max_budget_usd`, an `mcp_servers` entry of `type: "sdk"` without an `instance` or whose `instance` is not an MCP server, a plugin whose `type` is not `"local"` — raise `ValueError` when the client connects or the query starts, before any subprocess is spawned. Setting both `thinking` and `max_thinking_tokens` emits a `DeprecationWarning` (`thinking` wins).

See [src/claude_agent_sdk/\_errors.py](src/claude_agent_sdk/_errors.py) for all error types.

## Available Tools

See the [Claude Code documentation](https://code.claude.com/docs/en/settings#tools-available-to-claude) for a complete list of available tools.

## Examples

See [examples/quick_start.py](examples/quick_start.py) for a complete working example.

See [examples/streaming_mode.py](examples/streaming_mode.py) for comprehensive examples involving `ClaudeSDKClient`. You can even run interactive examples in IPython from [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py) — that file is a set of copy-paste cells with top-level `await`, not a runnable script.

Other scripts under [examples/](examples/): hooks (`hooks.py`), tool permission callbacks (`tool_permission_callback.py`), in-process MCP tools (`mcp_calculator.py`), custom agents (`agents.py`, `filesystem_agents.py`), `setting_sources.py`, `system_prompt.py`, `tools_option.py`, `max_budget_usd.py`, plugins (`plugin_example.py`), partial messages (`include_partial_messages.py`), the stderr callback (`stderr_callback_example.py`), and a trio variant (`streaming_mode_trio.py`). The examples call the Claude API and need a working CLI and credentials; they are linted with ruff and statically checked by [tests/test_examples_static.py](tests/test_examples_static.py) (compile, Python 3.10 compatibility, imports resolve against the package) without being run.

## Migrating from Claude Code SDK

If you're upgrading from the Claude Code SDK (versions < 0.1.0), please see the [CHANGELOG.md](CHANGELOG.md#010) for details on breaking changes and new features, including:

- `ClaudeCodeOptions` → `ClaudeAgentOptions` rename
- Merged system prompt configuration
- Settings isolation and explicit control
- New programmatic subagents and session forking features

## Development

If you're contributing to this project, run the initial setup script to install git hooks:

```bash
./scripts/initial-setup.sh
```

This installs a pre-push hook that runs lint checks before pushing, matching the CI workflow. To skip the hook temporarily, use `git push --no-verify`.

Install the development dependencies with `pip install -e ".[dev]"` (the `dev` extra includes `pyyaml`, which `tests/test_workflow_gates.py` uses to parse the Test workflow and check the real-API job gates), then run the same checks CI runs:

```bash
python -m ruff check src/ tests/ scripts/ examples/
python -m ruff format --check src/ tests/ scripts/ examples/
python -m mypy src/ scripts/
python -m pytest tests/
```

The pre-push hook and the Lint workflow run `ruff check` and `ruff format --check` over `src/ tests/ scripts/ examples/` (the copy-paste IPython snippet `examples/streaming_mode_ipython.py` is excluded via `pyproject.toml` because its cells use top-level `await`); CI additionally runs `mypy src/ scripts/`. The Test workflow runs the unit suite on Linux, macOS and Windows with the newest `mcp` and on the oldest supported `mcp` (1.23.0), and its `test-min-python` job runs it on Python 3.10, the oldest version the package declares, so code and examples must stay 3.10-compatible (no `asyncio.timeout`, `datetime.UTC` or `ExceptionGroup`). The real-API jobs (`test-e2e`, `test-e2e-docker`, `test-examples`) are skipped, not failed, when the workload-identity variables are not configured for the repository, so on a fork a passing Test workflow covers only the offline suite; see [e2e-tests/README.md](e2e-tests/README.md).

The unit suite runs every async test under both asyncio and trio and needs neither the CLI nor credentials; the end-to-end tests under [e2e-tests/](e2e-tests/README.md) do.

### Building Wheels Locally

To build wheels with the bundled Claude Code CLI:

```bash
# Install build dependencies
pip install build twine

# Build wheel with bundled CLI
python scripts/build_wheel.py

# Build with specific version
python scripts/build_wheel.py --version 0.1.4

# Build with specific CLI version
python scripts/build_wheel.py --cli-version 2.0.0

# Clean bundled CLI after building
python scripts/build_wheel.py --clean

# Skip CLI download (use existing)
python scripts/build_wheel.py --skip-download
```

The build script:

1. Downloads Claude Code CLI for your platform
2. Bundles it in the wheel
3. Builds both wheel and source distribution
4. Checks the package with twine

See `python scripts/build_wheel.py --help` for all options.

### Release Workflow

Releases are published by GitHub Actions; see [RELEASING.md](RELEASING.md) for the full flow. There are two triggers:

- **Automatic** — a commit `chore: bump bundled CLI version to X.Y.Z` pushed to `main` (which updates `src/claude_agent_sdk/_cli_version.py`) runs the Test workflow; on success `.github/workflows/auto-release.yml` increments the SDK patch version and releases it.
- **Manual** — run `.github/workflows/publish.yml` from the Actions tab with its single input, `version` (`MAJOR.MINOR.PATCH`, optionally with a PEP 440 `aN`/`bN`/`rcN`, `.postN` or `.devN` segment; no leading `v`). It first runs the unit suite on Python 3.10–3.13 and lint (`ruff check` / `ruff format --check` over `src/ tests/ scripts/ examples/`, `mypy src/ scripts/`).

Both hand `version` and the previous release tag to the reusable `.github/workflows/build-and-publish.yml`, which builds wheels with the bundled CLI on five platforms, runs `scripts/update_version.py` to set the version in `pyproject.toml` and `src/claude_agent_sdk/_version.py` (rejecting anything outside the format above and leaving both files untouched on error), builds the sdist, checks the PyPI size quota, uploads the sdist and wheels with `PYPI_API_TOKEN`, commits the version bump (`chore: release vX.Y.Z`) plus a best-effort Claude-generated `CHANGELOG.md` entry, pushes them directly to `main` over `DEPLOY_KEY`, and creates the `vX.Y.Z` tag and GitHub Release. Claude API access for the changelog step uses workload identity federation (the job's OIDC token, `id-token: write`), configured through repository variables, not a static API key. The bundled CLI version is tracked separately from the package version, so a CLI bump releases a new package without code changes.

## License and terms

Use of this SDK is governed by Anthropic's [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms), including when you use it to power products and services that you make available to your own customers and end users, except to the extent a specific component or dependency is covered by a different license as indicated in that component's LICENSE file.
