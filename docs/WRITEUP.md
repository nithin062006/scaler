# GraphForge — writeup

A condensed walk-through of the problem, env design, training approach, and
results. Pair this with the [README](../README.md) and the
[training notebook](../training/notebook.ipynb).

## 1. Why a graph, not a text editor?

LMs that write code today emit source as token sequences. That representation
is dense in characters but **sparse in structure**. To answer "which functions
call this one?" or "would this edit create a circular import?" the agent
must re-parse text every turn. Mistakes early in construction propagate
silently because the next token doesn't know about the type six functions
ago.

GraphForge inverts the artifact. The function-call graph is canonical;
source files are deterministic projections. The agent never edits source —
it mutates a typed structure. `query_subgraph(neighbors:main.register)`
costs ~150 tokens; the same question against text would require all the
surrounding source. Type information is first-class, so a wrong arg-mapping
is rejected at action-application time, before it enters the graph.

## 2. The task

Tier-0 task (the one trained against in this submission):

> Build a tiny single-module package called 'validators'. It should expose
> a function `is_email(s: str) -> bool` that returns True for well-formed
> email addresses and False otherwise. Use the `validate_with_regex` body
> template with the EMAIL pattern. The module must materialize cleanly to
> runnable Python.

Visible spec: 4 constraints. Hidden spec: 3 more (module size, no stray
nodes, acyclic imports). Episode cap: 20 turns. Token budget: 4 000.

The reward function pays +1 per satisfied structural constraint, +5 if all
satisfied, -8 if materialization fails, plus an efficiency bonus gated on
full success. Per-turn shaping penalizes failed mutations, malformed
output, and duplicates. The agent has to plan, not just churn.

## 3. The environment

OpenEnv-compliant: extends `openenv.core.Environment[Action, Observation, State]`
with strongly-typed pydantic in/out. Internally delegates to the engine in
`graphforge/`:

```
agent → /step → GraphForgeEnvironment.step
                  ├─ TypeAdapter(Action).validate_python  → typed action
                  ├─ dispatch(graph, action)              → atomic + rollback
                  ├─ score_turn(outcome, dup, tokens)     → per-turn reward
                  ├─ if Submit / cap:
                  │     materialize → full_check          → materialization gate
                  │     evaluate_all(graph, constraints)  → struct/behav split
                  │     score_terminal(...)               → +1 per sat / +5 bonus / etc
                  └─ return (obs, reward, done)
```

The materializer is a pure function over the graph: same graph in, same
source out, deterministic ordering, deduped imports. The validator runs
`compile()` per file (parse-only for now; mypy gate is next). Constraint
checking is per-kind dispatch over the graph.

## 4. Training: rejection-sampling SFT

GRPO needs a GPU bigger than a free T4 once you add a KL ref model. We
shipped SFT instead, which is faster, fits in 16 GB, and produces clean
plots.

```
generate → reject-sample → format → SFT
   │            │             │       │
   ▼            ▼             ▼       ▼
 oracle    threshold    chat-template TRL SFTTrainer
 + model   reward >= 5  prompt+completion + LoRA r=16
```

**Why oracle trajectories?** A 0.5B model rarely produces a successful
trajectory unprompted. Without seed examples the rejection-sampling filter
keeps zero examples and SFT has nothing to learn from. The oracle gives
the model a teacher trace it can imitate; from there it generalizes within
the action vocabulary.

**Why LoRA?** Speed and VRAM headroom. With LoRA on
`q/k/v/o_proj` we update ~0.5 % of the params and free up the rest of the
T4 for batch size and grad accumulation.

## 5. What the plots show

- **Baseline vs trained.** Untrained Qwen2.5-0.5B mostly emits
  free-form prose without `<action>` tags, scoring −2 per turn until the
  episode cap fires (~−40 terminal reward). After SFT, the model emits
  the canonical sequence reliably and the reward distribution shifts to
  a tight cluster around the oracle's terminal reward (~+15 to +20 with
  the efficiency bonus included).
- **Loss curve.** SFT loss decreases monotonically; nothing exotic — it's
  imitation on a small but coherent dataset.
- **Reward histogram.** The two distributions barely overlap. The
  improvement isn't a small mean shift — it's a regime change.

## 6. Limitations and what's next

- **One task.** Tier-0 is solvable, but tier-1 / tier-2 (multiple modules,
  type-flow constraints, behavioral tests) are scaffolded and not yet
  trained against. The training pipeline is task-agnostic, so adding tasks
  is data work, not engineering.
- **Materialization gate is parse-only.** The next gate (`mypy --strict`
  in a sandboxed subprocess with hard timeouts) is stubbed in
  `graphforge.validator`.
- **SFT, not GRPO.** Once we have access to an A100, the proposal's GRPO
  multi-turn rollout (`graphforge.training.rollout`) is wired and ready
  to drive `trl.GRPOTrainer`.

## 7. What this proves

- The env produces a real, non-trivial reward signal that a 0.5B model can
  improve against in under 20 minutes on a free T4.
- The graph-first representation is workable: `materialize ∘ dispatch` is
  fast enough to score every step in real time.
- The hackathon-required artifacts are reproducible end-to-end from the
  notebook, not from cached numbers.
