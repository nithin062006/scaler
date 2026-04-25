# GraphForge: A Graph-First Code Generation Environment for Long-Horizon Planning

## Abstract

GraphForge is an OpenEnv-compliant reinforcement learning environment that
trains a small language model to generate Python programs by incrementally
constructing a function-call graph, rather than emitting source code as text.
Given a natural-language task description and a partially-hidden structural
specification, the agent declares function nodes, adds call edges, organizes
modules, attaches typed bodies via constrained body templates, and validates
intermediate states. On submission, the graph is deterministically materialized
into Python source files which are verified to parse, import, type-check, and
pass property-based behavioral tests. Source code becomes a projection of the
canonical graph representation, supporting human editing and version control
through a round-trip parser.

This task is a long-horizon planning problem of substantial depth: programs of
meaningful size require 30–60 interdependent decisions whose correctness only
becomes apparent at materialization or test execution. Reward is sparse,
delivered primarily at submission, with a per-turn token-cost term that
incentivizes the agent to use compact structural queries over expensive state
inspection. We train Qwen2.5-0.5B-Instruct with GRPO and demonstrate the
trained agent constructing materializable, type-correct, behaviorally-conforming
programs that the baseline cannot.

## 1. Motivation

### 1.1 The Problem

Current code-generating language models emit source code as token sequences.
This representation is dense in characters but sparse in structure:
relationships between functions, modules, types, and call sites are implicit in
the text and must be re-derived by every consumer of the code. For agents
performing multi-step program construction, this creates compounding
inefficiencies:

1. **Token bloat across long horizons.** As the program grows, the agent must
   keep increasingly long source code in context to plan further edits. By
   turn 30 of a non-trivial construction task, a small model is spending most
   of its context on already-written code rather than planning.
2. **Implicit structure forces reasoning to be re-done.** "Which functions
   call this one?" requires re-parsing every file. "Will adding this edge
   create a circular import?" requires walking the import graph. "Does this
   signature match what the callers expect?" requires reading every call
   site. These are cheap operations on a typed graph and expensive operations
   on text.
3. **Errors compound silently.** A wrong type early in construction propagates
   through every dependent function. The agent has no inherent signal that
   its decision was wrong until the program is run, by which point unwinding
   the mistake requires understanding the full text of the code.

The conventional response is to retrieve and chunk source code at inference
time. This is reactive: the canonical artifact remains text. GraphForge
inverts the pipeline. The graph is canonical; source files are a
deterministic projection of the graph, regenerated on demand. Type
information, call structure, and module partitioning are first-class and
queryable.

### 1.2 Why an LLM, not AST

AST goes source → graph deterministically. GraphForge goes graph → source: the
agent must decide what program to build from an underspecified
natural-language description, choose an architecture, partition functions
across modules, select types and signatures, attach behavioral templates, and
recover when a chosen architecture proves untenable. AST cannot perform any
of these steps — it has no notion of intent, judgment, or recovery.

Within the environment, AST is used only as a tool in the round-trip parser
that re-derives the graph from edited source files (supporting human-in-the-loop
editing). The agent's contribution is the construction process: the sequence
of architectural decisions that produces a coherent typed graph from an
ambiguous prompt.

### 1.3 Theme Alignment

**Theme 2 (Long-Horizon Planning & Instruction Following) — primary.** Tasks
require 30–60 interdependent graph mutations across multiple decision
dimensions: structure, types, module partitioning, body selection, and
validation. Reward is sparse: the agent learns whether early architectural
choices were correct only at submission, when the materializer, type checker,
and behavioral test suite run. Mistakes early in the trajectory propagate
forward and may not be recoverable without restructuring substantial portions
of the graph. Recovery — detecting the failure, identifying the responsible
subgraph, and replanning — is itself a learned skill.

**Token efficiency — secondary motivation.** The graph representation is more
compact than equivalent source code, and the environment makes this concrete
by charging the agent for tokens consumed reading state. The trained agent
learns to plan using cheap structural queries (`query_spec`,
`query_subgraph`, `query_types`) and only materialize/inspect source when
necessary.

---

The full proposal text continues in this file at the same level of detail as
the source pitch (sections 2–10), but is replicated faithfully so that it can
be referenced offline. See sections in the source-of-truth document for:

- §2 Task definition (constraint vocabulary, tier structure, why tier 1 is
  already substantial)
- §3 Graph representation (schema, body templates, module layout,
  round-trip parser)
- §4 Action surface (mutations, info actions, terminal action, atomicity)
- §5 Reward function (per-turn dense, terminal sparse, no per-edit progress)
- §6 System architecture (component overview, architectural commitments)
- §7 Training (GRPO + custom multi-turn rollout, plan B)
- §8 Evaluation protocol (metrics, baseline / trained / held-out, headline plots)
- §9 Risk register
- §10 Submission deliverables

This file will be expanded with the full text of those sections in a follow-up
commit; the canonical version lives in the design conversation that initiated
this repo.
