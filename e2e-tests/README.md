# End-to-end tests

The tests in this directory drive the real Claude Code CLI against the Claude
API. They are the only part of the test suite that needs credentials and costs
money; the unit suite under `tests/` runs offline and is what `python -m pytest`
runs by default (`testpaths = ["tests"]` in `pyproject.toml`), so this directory
must always be named explicitly.

## Requirements

- **Claude Code CLI.** Either an installed `claude` on `PATH`
  (`curl -fsSL https://claude.ai/install.sh | bash`) or the CLI bundled into an
  installed wheel. Check with `claude -v`.
- **The SDK with dev dependencies:** `pip install -e ".[dev]"`.
- **Credentials** — see the next section.

## Authentication

### Locally: `ANTHROPIC_API_KEY`

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m pytest e2e-tests/ -v -m e2e
```

The tests do not call the API themselves; the CLI subprocess the SDK spawns
does, so it reads the key. `e2e-tests/conftest.py` also provides a
session-scoped `api_key` fixture that **fails** (it does not skip) when
`ANTHROPIC_API_KEY` is unset — request it from a test that must not run without
an API key.

### In CI: workload identity federation

CI does not use a long-lived API key. The `test-e2e`, `test-e2e-docker` and
`test-examples` jobs in [`.github/workflows/test.yml`](../.github/workflows/test.yml)
call the composite action
[`.github/actions/setup-claude-auth`](../.github/actions/setup-claude-auth/action.yml),
which:

1. mints a GitHub OIDC identity token for the audience
   `https://api.anthropic.com` (the jobs request `id-token: write`);
2. writes it to a file and exports `ANTHROPIC_IDENTITY_TOKEN_FILE` pointing at
   it;
3. starts a detached refresher that rewrites the file every few minutes, so CLI
   processes started late in the job still exchange a valid token.

The CLI exchanges that token for a Claude API access token using the repository
variables `ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID` and
`ANTHROPIC_SERVICE_ACCOUNT_ID`, which the jobs pass through as environment
variables. The Docker job bind-mounts the token file into the container so the
refresher on the host keeps the in-container copy fresh too.

The `test-e2e`, `test-e2e-docker` and `test-examples` jobs run only for pushes
and for pull requests whose head branch lives in this repository, only after the
unit-test job has passed, and only when the repository variables
`ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID` and
`ANTHROPIC_SERVICE_ACCOUNT_ID` are configured. A fork (or any repository)
without those variables skips them: a green Test run there means the offline
unit, lint and compatibility jobs passed — not that the real-API end-to-end
tests or examples ran. Pull requests from external forks never run them,
because a fork cannot mint the identity token.

CI also runs `python scripts/trust_workspace.py` first: a fresh checkout is an
untrusted workspace, and Claude Code would otherwise drop the project-scoped
`permissions.allow` grants committed in `.claude/settings.json` and warn about
it on stderr. Run the script yourself if you see that warning locally.

## The pinned default model

`e2e-tests/conftest.py` sets `ANTHROPIC_DEFAULT_MODEL=claude-opus-5` (with
`os.environ.setdefault`, so a value already in your environment is kept). Tests
that do not pass `model=` — and `set_model(None)`, which returns to the CLI's
default — would otherwise use whatever default the installed CLI ships with,
which changes between CLI releases and has broken the suite before (CLI 2.1.280
moved the default to a model the API rejected for the organization CI runs
under). `ANTHROPIC_MODEL` would not cover `set_model(None)`, which is why the
pin uses `ANTHROPIC_DEFAULT_MODEL`. A test's own `model=` still wins. The
`test-examples` CI job pins the same value for the example scripts.

## Running the tests

```bash
# Everything (the -m e2e marker is registered in e2e-tests/conftest.py and
# every test here carries it; selecting it keeps accidental non-e2e
# collection out)
python -m pytest e2e-tests/ -v -m e2e

# One module
python -m pytest e2e-tests/test_structured_output.py -v

# One test
python -m pytest "e2e-tests/test_sdk_mcp_tools.py::test_sdk_mcp_tool_execution" -v
```

The e2e `conftest.py` pins the anyio backend to asyncio; the unit suite runs
every async test under both asyncio and trio, but repeating real API calls on
a second backend would double cost without exercising additional SDK code.

### In Docker

[`scripts/test-docker.sh`](../scripts/test-docker.sh) builds
[`Dockerfile.test`](../Dockerfile.test) (a `python:3.12-slim` image with the
CLI installed and the workspace trusted) and runs the tests inside it, which
catches container-specific issues such as #406:

```bash
./scripts/test-docker.sh unit                          # unit tests only, no credentials
ANTHROPIC_API_KEY=sk-ant-... ./scripts/test-docker.sh e2e
ANTHROPIC_API_KEY=sk-ant-... ./scripts/test-docker.sh all
```

The script passes `ANTHROPIC_API_KEY` into the container; the CI Docker job
uses the identity-token bind mount described above instead.

## Cost

Every test makes real API calls. Most use a short prompt and a single turn;
tests that run tools or spawn subagents (for example `test_run_end.py`,
`test_forward_subagent_text.py`, `test_subagent_session_reads.py` and
`test_agents_and_settings.py`) cost more because each subagent is its own
model call. The suite runs on every push to `main` and on every in-repo pull
request (on three operating systems plus Docker), so keep new tests to the
smallest prompt and `max_turns` that still prove the behavior.

## Test modules

| Module | Covers |
| --- | --- |
| `test_agents_and_settings.py` | Inline `AgentDefinition`s (including large definitions sent through `initialize`), filesystem agents loaded via `setting_sources`, and `setting_sources` itself (CLI defaults, `user` only, `project` included) |
| `test_conversation_reset.py` | `/clear` in streaming input mode emits a `ConversationResetMessage` |
| `test_dynamic_control.py` | `set_permission_mode`, `set_model`, `interrupt` on `ClaudeSDKClient` |
| `test_error_results.py` | A terminal API error surfaces as `ResultError` with the result payload |
| `test_forward_subagent_text.py` | `forward_subagent_text` delivers attributed subagent text; off by default |
| `test_hook_events.py` | Hook event types: `PreToolUse` `additionalContext`, `PostToolUse` with `tool_use_id`, `Notification`, several hooks together |
| `test_hooks.py` | Hook outputs: `permissionDecision` + `reason`, `continue`/`stopReason`, `additionalContext` |
| `test_include_partial_messages.py` | `include_partial_messages`: `StreamEvent` delivery, thinking deltas, disabled by default |
| `test_message_origin.py` | The `origin` stamped on a streamed user message round-trips to the turn's `ResultMessage` |
| `test_run_end.py` | When a one-shot `query()` with hooks closes stdin; follow-up turns after a background subagent (#1190) |
| `test_sdk_mcp_tools.py` | In-process SDK MCP tools: execution, permission enforcement, multiple tools, no pre-approval |
| `test_session_store_resume_settings.py` | A `SessionStore`-backed resume seeds the user's `settings.json` into the temporary `CLAUDE_CONFIG_DIR` |
| `test_stderr_callback.py` | The `stderr` callback is wired up and stays silent on a clean run from an empty working directory |
| `test_structured_output.py` | `output_format` with a JSON schema: flat, nested, enum, and with tool use |
| `test_subagent_session_reads.py` | Subagent transcript reads recover `parent_tool_use_id` |
| `test_tool_permissions.py` | `can_use_tool` is invoked; works with string and async-iterable prompts in `query()` |
| `test_truncating_resume.py` | Truncating resume with `resume_session_at` / `resume_drops_turn` |
| `test_verbatim_prompts.py` | `verbatim_prompts`: `@/path` expansion by default, verbatim delivery when enabled |

## Troubleshooting

- **`ANTHROPIC_API_KEY environment variable is required for e2e tests`** — the
  `api_key` fixture failed; export the key (or run under the CI federation
  setup).
- **`CLINotFoundError`** — install the CLI or point the test at one with
  `ClaudeAgentOptions(cli_path=...)`; check `claude -v`.
- **HTTP 400 mentioning the model or permission-mode attestation** — the CLI
  default model changed; see "The pinned default model" above.
- **Warnings about an untrusted workspace / dropped permission grants** — run
  `python scripts/trust_workspace.py` from the checkout.
- **Timeouts** — check network access to `api.anthropic.com` and the account's
  quota; the CLI prints the cause on stderr, which you can capture with
  `ClaudeAgentOptions(stderr=print)`.

## Adding new e2e tests

1. Mark tests with `@pytest.mark.e2e` and `@pytest.mark.anyio`.
2. Keep prompts tiny and set `max_turns` where it does not change what you
   test. Either leave `model=` unset so the pinned default applies, or pick a
   cheap model explicitly (several tests use `model="haiku"`).
3. Give file-writing tests a `tmp_path` working directory.
4. Assert on real messages (`ToolUseBlock`, `ResultMessage`, ...) rather than on
   printed output.
5. Document any special setup in this README and add the module to the table
   above.
