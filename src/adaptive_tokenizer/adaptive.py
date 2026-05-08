"""
AdaptiveTokenizer — the full iterative stack (A3).

Wraps :class:`AcousticTokenizer` (A1) and a :class:`SequenceMerger`
(A2: BPEMerger or B1: PMIMerger) in an iterative refinement loop inspired by
BEATs [Chen 2023]:

    Repeat until convergence:
        1. Acoustic pass: split/merge/add/prune on current embeddings.
        2. Sequence pass: merge frequent bigrams / high-PMI bigrams into
           composite tokens.
        3. Check convergence: did vocab_size change? Did silhouette improve?

Convergence is deliberately conservative — we stop as soon as the vocabulary
is stable across two consecutive iterations, or when ``max_outer_iters`` is
reached.

The outer loop alternation helps because:

- After a sequence merge, the set of atomic tokens used has effectively
  shifted — some atomic ids may now be rare (most of their occurrences
  eaten by composites). An extra acoustic pass can then prune them.
- After an acoustic split, new atomic ids may form novel bigrams worth
  merging.

Design notes:

- The acoustic pipeline operates on segment embeddings ``X``; the sequence
  pipeline operates on token sequences in ``state.sequences``. Both are
  kept consistent through :class:`TokenizerState`.
- Composite tokens produced by the merger live in ``state.tokens`` with
  ``is_composite=True``. They are NOT touched by acoustic ops (so we don't
  accidentally split a valid linguistic unit back into parts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from .acoustic import AcousticTokenizer, AcousticTokenizerConfig
from .sequence import BPEMerger, PMIMerger, SequenceMerger, SequenceMergerConfig
from .types import Operation, TokenizerState


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveTokenizerConfig:
    acoustic: AcousticTokenizerConfig = field(default_factory=AcousticTokenizerConfig)
    sequence: SequenceMergerConfig = field(default_factory=SequenceMergerConfig)
    sequence_kind: str = "bpe"              # "bpe" | "pmi" | "none"
    max_outer_iters: int = 3
    min_vocab_delta: int = 0                # stop when |vocab_t − vocab_{t-1}| ≤ this
    verbose: bool = True


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

class AdaptiveTokenizer:
    """Full adaptive pipeline: acoustic + sequence in alternating passes.

    Usage:

        tok = AdaptiveTokenizer()
        state = tok.fit(X, sequences_per_file=seg_ids_per_file)
        # downstream:
        state.vocab_size
        state.sequences              # lists of atomic + composite token ids
        state.history                # full operation log

    ``fit`` returns the final :class:`TokenizerState`.
    """

    def __init__(self, config: Optional[AdaptiveTokenizerConfig] = None):
        self.cfg = config or AdaptiveTokenizerConfig()
        self.acoustic = AcousticTokenizer(self.cfg.acoustic)
        self.merger: Optional[SequenceMerger] = {
            "bpe": BPEMerger(self.cfg.sequence),
            "pmi": PMIMerger(self.cfg.sequence),
            "none": None,
        }.get(self.cfg.sequence_kind, None)
        if self.merger is None and self.cfg.sequence_kind != "none":
            raise ValueError(f"unknown sequence_kind: {self.cfg.sequence_kind}")

    def fit(
        self,
        X: np.ndarray,
        *,
        seed_labels: Optional[np.ndarray] = None,
        sequences_per_file: Optional[list[list[int]]] = None,
    ) -> TokenizerState:
        """Run the full iterative loop. Parameters mirror :meth:`AcousticTokenizer.fit`."""
        # First acoustic fit (with full inner split/merge/add/prune loop)
        state = self.acoustic.fit(X, seed_labels=seed_labels,
                                    sequences_per_file=sequences_per_file)
        if self.cfg.verbose:
            print(f"[outer 0] after seed: {state.summary()}")

        prev_vocab = state.vocab_size
        pbar = tqdm(range(self.cfg.max_outer_iters),
                     desc='A3 outer loop',
                     disable=not self.cfg.verbose,
                     leave=True)
        for outer in pbar:
            state.iteration = outer + 1
            state.log(Operation(
                kind="iterate", iteration=state.iteration,
                affected_ids=(),
                metrics_before={"vocab_size": float(state.vocab_size)},
                note=f"outer iteration {outer + 1}",
            ))

            # Sequence pass (if any)
            if self.merger is not None:
                before = state.vocab_size
                self.merger.fit(state)
                after = state.vocab_size
                if self.cfg.verbose:
                    pbar.write(f"[outer {outer + 1}] sequence pass: vocab {before} -> {after}")

            # Acoustic step — single pass, not full inner loop (to avoid
            # re-splitting everything from scratch)
            touched = self.acoustic.fit_step(X, state)
            if self.cfg.verbose:
                pbar.write(f"[outer {outer + 1}] acoustic pass: vocab={state.vocab_size}, any_op={touched}")

            state.validate()
            pbar.set_postfix(vocab=state.vocab_size,
                              atomic=state.atomic_vocab_size,
                              composite=state.composite_vocab_size)

            if abs(state.vocab_size - prev_vocab) <= self.cfg.min_vocab_delta:
                if self.cfg.verbose:
                    pbar.write(f"[outer {outer + 1}] converged (Δvocab ≤ {self.cfg.min_vocab_delta})")
                break
            prev_vocab = state.vocab_size
        pbar.close()

        return state


# ────────────────────────────────────────────────────────────────────────────
# Convenience factories for the runner notebook
# ────────────────────────────────────────────────────────────────────────────

def default_acoustic_only(ac_cfg: Optional[AcousticTokenizerConfig] = None) -> AdaptiveTokenizer:
    """Acoustic ops only (A1) — no sequence merging. Baseline variant."""
    return AdaptiveTokenizer(AdaptiveTokenizerConfig(
        acoustic=ac_cfg or AcousticTokenizerConfig(),
        sequence_kind="none",
    ))


def default_bpe(ac_cfg: Optional[AcousticTokenizerConfig] = None,
                 seq_cfg: Optional[SequenceMergerConfig] = None) -> AdaptiveTokenizer:
    """A1 + BPE — the primary thesis pipeline."""
    return AdaptiveTokenizer(AdaptiveTokenizerConfig(
        acoustic=ac_cfg or AcousticTokenizerConfig(),
        sequence=seq_cfg or SequenceMergerConfig(),
        sequence_kind="bpe",
    ))


def default_pmi(ac_cfg: Optional[AcousticTokenizerConfig] = None,
                 seq_cfg: Optional[SequenceMergerConfig] = None) -> AdaptiveTokenizer:
    """A1 + PMI — alternative merger variant for A/B comparison."""
    return AdaptiveTokenizer(AdaptiveTokenizerConfig(
        acoustic=ac_cfg or AcousticTokenizerConfig(),
        sequence=seq_cfg or SequenceMergerConfig(),
        sequence_kind="pmi",
    ))
