"""GraphForge — graph-first code generation environment for long-horizon RL.

The agent constructs Python programs by mutating a typed function-call graph;
source files are a deterministic projection of the canonical graph.

Top-level subsystems:
  graph         canonical graph schema (Modules, Nodes, Edges)
  actions       eleven-action surface, atomic dispatcher with rollback
  types         signature parser + edge type-flow validator
  templates     ~25-template body library, parameterized
  materializer  graph -> Python source
  parser        Python source -> graph (round-trip)
  validator     parse / import / mypy --strict gate
  behavioral    hypothesis-based property test runner
  constraints   per-kind constraint checker dispatch
  reward        reward engine (per-turn + terminal)
  tasks         task bank + variant generator
  server        FastAPI OpenEnv server
  training      GRPO multi-turn rollout

See README.md for design rationale and PROPOSAL.md for the full spec.
"""

__version__ = "0.0.1"
