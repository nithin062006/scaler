"""Policy interface and stub policies.

A *policy* is anything that, given a list of messages, returns a single
completion string. The rollout doesn't care whether that string came from
a 0.5B Qwen sample, a hand-written script, or random noise — only that
the contract is honored.

This file provides:

  * :class:`Policy` — a runtime-checkable Protocol.
  * :class:`ScriptedPolicy` — yields a fixed list of completions in order.
    Useful for tests and for building oracle trajectories during rejection-
    sampling SFT (PROPOSAL.md §7.4 plan B).
  * :class:`HfPolicy` — wraps an HF causal LM + tokenizer; the real thing.
    Defined here so consumers can swap it in once we hook up Qwen, but
    deliberately not imported at module-load time.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from graphforge.training.prompt import Message


@runtime_checkable
class Policy(Protocol):
    def sample(self, messages: list[Message]) -> str: ...


# ---- scripted -------------------------------------------------------


class ScriptedPolicy:
    """Returns each item of ``completions`` in order.

    If the rollout asks for more turns than there are scripted completions,
    raises :class:`StopIteration` — that's a test bug, not an env bug.
    """

    def __init__(self, completions: list[str]) -> None:
        self._iter: Iterator[str] = iter(completions)
        self._n = len(completions)

    def sample(self, _messages: list[Message]) -> str:
        return next(self._iter)


# ---- HF (lazy) ------------------------------------------------------


class HfPolicy:
    """A real LM-backed policy. Imports torch / transformers lazily.

    Constructor args::

        model           — a HF AutoModelForCausalLM
        tokenizer       — the matching tokenizer
        max_new_tokens  — generation cap per turn (PROPOSAL.md §7.1: 384)
        temperature, top_p — sampling knobs
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def sample(self, messages: list[Message]) -> str:
        # Defer heavy imports.
        import torch  # noqa: F401  — required for inputs / device

        tok = self.tokenizer
        # Render to text first, then tokenize. ``apply_chat_template`` 's
        # return type drifted across transformers versions (sometimes a raw
        # tensor, sometimes a BatchEncoding); going through ``tok(text)`` is
        # the canonical pattern and works on all of them.
        text = tok.apply_chat_template(  # type: ignore[attr-defined]
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tok(text, return_tensors="pt")  # type: ignore[operator]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # type: ignore[attr-defined]

        with torch.no_grad():
            out_ids = self.model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=tok.eos_token_id,  # type: ignore[attr-defined]
            )
        prompt_len = inputs["input_ids"].shape[-1]
        gen = out_ids[0, prompt_len:]
        return tok.decode(gen, skip_special_tokens=True)  # type: ignore[attr-defined]
