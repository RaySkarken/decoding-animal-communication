"""
Core dataclasses for adaptive tokenization.

These are plain Python dataclasses — no TensorFlow / PyTorch dependency. They
carry the state of an evolving vocabulary and per-segment labels.

Design notes:
- A `Token` always has an integer `id` and a `centroid` in embedding space,
  regardless of whether it was created by seed clustering, split, merge, or
  sequence-level BPE.
- Composite tokens (produced by BPE merging of frequent bigrams) keep a
  `children` tuple pointing to their constituent atomic tokens. The
  distinction matters for interpretation but not for downstream tokenization.
- `TokenizerState` is the single object that holds everything needed to
  resume or evaluate a tokenizer run: current vocabulary, per-segment labels,
  per-file sequences, and an operation log for post-hoc analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


OperationKind = Literal["seed", "split", "merge", "add", "prune", "bpe_merge", "iterate"]


@dataclass
class Token:
    """A single vocabulary entry.

    Atomic tokens (from clustering) have ``children = ()`` and ``is_composite = False``.
    Composite tokens (from BPE / PMI merging of bigrams) record the chain of
    children; a composite's centroid is inherited from its constituent
    acoustic membership, typically the mean of the underlying atomic centroids
    weighted by frequency.
    """

    id: int
    centroid: np.ndarray              # (d,) — in the embedding space used to seed
    member_ids: np.ndarray            # indices into the original segment array
    is_composite: bool = False
    children: tuple[int, ...] = ()    # atomic-token ids that compose this one
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(len(self.member_ids))

    def __repr__(self) -> str:
        kind = "composite" if self.is_composite else "atomic"
        return f"Token(id={self.id}, {kind}, size={self.size}, children={self.children})"


@dataclass
class Operation:
    """One logged action taken by the tokenizer — used for thesis figures."""

    kind: OperationKind
    iteration: int
    affected_ids: tuple[int, ...]      # tokens created or destroyed
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass
class TokenizerState:
    """Mutable state of a tokenizer over the course of a run.

    The invariant is: ``labels[i]`` is the current token id assigned to segment
    ``i``, and every id referenced in ``labels`` is present in ``tokens``.
    """

    tokens: dict[int, Token]           # id -> Token
    labels: np.ndarray                 # (n_segments,) — current atomic label per segment
    sequences: list[list[int]]         # per-file token sequences (possibly composite)
    iteration: int = 0
    history: list[Operation] = field(default_factory=list)

    # ── derived / convenience ────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def atomic_vocab_size(self) -> int:
        return sum(1 for t in self.tokens.values() if not t.is_composite)

    @property
    def composite_vocab_size(self) -> int:
        return sum(1 for t in self.tokens.values() if t.is_composite)

    def next_token_id(self) -> int:
        """Allocate a fresh integer id, larger than any existing one."""
        return max(self.tokens) + 1 if self.tokens else 0

    def log(self, op: Operation) -> None:
        self.history.append(op)

    def active_token_ids(self) -> set[int]:
        """Ids currently used by at least one segment or one sequence position."""
        ids: set[int] = set(int(x) for x in np.unique(self.labels) if x >= 0)
        for seq in self.sequences:
            ids.update(seq)
        return ids

    def validate(self) -> None:
        """Raise if the invariants are violated. For debugging."""
        active = self.active_token_ids()
        missing = active - set(self.tokens)
        if missing:
            raise ValueError(f"labels/sequences reference missing token ids: {sorted(missing)}")
        for tid, tok in self.tokens.items():
            if tid != tok.id:
                raise ValueError(f"dict key {tid} != Token.id {tok.id}")

    def summary(self) -> dict[str, Any]:
        """Short diagnostic dict — handy for logging."""
        sizes = sorted((t.size for t in self.tokens.values()), reverse=True)
        return {
            "iteration": self.iteration,
            "vocab_size": self.vocab_size,
            "atomic": self.atomic_vocab_size,
            "composite": self.composite_vocab_size,
            "n_segments": int(len(self.labels)),
            "n_sequences": len(self.sequences),
            "top5_token_sizes": sizes[:5],
            "noise_fraction": float((self.labels == -1).mean()),
            "n_operations": len(self.history),
        }
