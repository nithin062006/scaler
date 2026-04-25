# Contributing to GraphForge

## Running tests

This sandbox doesn't have outbound PyPI access, so set up tests locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The test suite covers:

- `tests/test_graph_schema.py` — graph dataclass invariants, lookup helpers,
  cycle detection, depth, structural hashing.
- `tests/test_signature.py` — parameter-name parser used by the dispatcher
  to validate edge `arg_mapping` coverage.
- `tests/test_actions.py` — every implemented action's success path, every
  failure mode (collision, unknown ref, would-create-cycle, template arg
  mismatch, arg-mapping coverage), and rollback atomicity.

## Style

- `ruff check graphforge tests` — lint
- `mypy graphforge` — type-check (strict, see `pyproject.toml`)

## Branching

Land features on small focused branches. The risk register
(PROPOSAL.md §9) ranks subsystems by failure mode — start with the
materializer ↔ round-trip parser pair, since they share the most
correctness surface.
