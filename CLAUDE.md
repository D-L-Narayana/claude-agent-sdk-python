# Workflow

```bash
# Lint and style
# Check for issues and fix automatically (examples/ is linted too; the
# IPython snippet is excluded via pyproject.toml)
python -m ruff check src/ tests/ scripts/ examples/ --fix
python -m ruff format src/ tests/ scripts/ examples/

# Typecheck
python -m mypy src/ scripts/

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_client.py
```

# Codebase Structure

- `src/claude_agent_sdk/` - Main package
  - `client.py` - ClaudeSDKClient for interactive sessions
  - `query.py` - One-shot query function
  - `types.py` - Type definitions
  - `stores/` - Durable `SessionStore` implementations shipped with the SDK (SQLite)
  - `testing/` - Conformance harness for third-party `SessionStore` adapters
  - `_internal/` - Internal implementation details
    - `transport/subprocess_cli.py` - CLI subprocess management
    - `query.py` - Control protocol (hooks, permissions, SDK MCP, run-end logic)
    - `message_parser.py` - Message parsing logic
    - `sessions.py`, `session_*.py` - Session listing, resume, import/sync/export
