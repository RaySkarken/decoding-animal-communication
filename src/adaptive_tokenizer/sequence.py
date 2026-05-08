"""
Sequence-level mergers on top of acoustic tokens.

Two variants share the same interface :class:`SequenceMerger`:

- :class:`BPEMerger` — classical Byte-Pair Encoding: at each step merge the
  most frequent bigram into a new composite token, until a stopping
  criterion (max merges, frequency threshold, vocab size cap).
- :class:`PMIMerger` — Pointwise Mutual Information variant: at each step
  merge the bigram with the highest PMI above a threshold. This is more
  robust to imbalanced frequency distributions than BPE (rare-but-
  meaningful pairs can still win).

Both operate on ``TokenizerState.sequences`` (lists of atomic token ids) and
register the new composite tokens in ``TokenizerState.tokens``, preserving
the chain through the ``children`` attribute.

The merger does NOT touch ``state.labels`` — per-segment labels stay atomic
for downstream acoustic analyses (silhouette, per-cluster metrics). The
composite tokens live only in the sequence representation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from .types import Operation, Token, TokenizerState


# ────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ────────────────────────────────────────────────────────────────────────────

def _count_bigrams(sequences: list[list[int]]) -> Counter[tuple[int, int]]:
    bg: Counter[tuple[int, int]] = Counter()
    for seq in sequences:
        for p in pairwise(seq):
            bg[p] += 1
    return bg


def _count_unigrams(sequences: list[list[int]]) -> Counter[int]:
    ug: Counter[int] = Counter()
    for seq in sequences:
        ug.update(seq)
    return ug


def _rewrite_sequences_with_merge(sequences: list[list[int]],
                                    pair: tuple[int, int],
                                    new_id: int) -> list[list[int]]:
    """Greedy left-to-right rewrite replacing ``pair`` with ``new_id``."""
    a, b = pair
    out: list[list[int]] = []
    for seq in sequences:
        new_seq: list[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out.append(new_seq)
    return out


def _composite_centroid_from_pair(state: TokenizerState,
                                    id_a: int, id_b: int) -> np.ndarray:
    """Weighted mean of the two tokens being merged — independent of
    atomic-children lookup so it survives later atomic-token removal."""
    tok_a = state.tokens[id_a]
    tok_b = state.tokens[id_b]
    wa = max(tok_a.size, 1)
    wb = max(tok_b.size, 1)
    return (tok_a.centroid * wa + tok_b.centroid * wb) / (wa + wb)


def _composite_members_from_pair(state: TokenizerState,
                                    id_a: int, id_b: int) -> np.ndarray:
    """Union of the pair's member_ids."""
    tok_a = state.tokens[id_a]
    tok_b = state.tokens[id_b]
    return np.concatenate([tok_a.member_ids, tok_b.member_ids])


# ────────────────────────────────────────────────────────────────────────────
# Configuration (shared)
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SequenceMergerConfig:
    max_merges: int = 30                 # hard cap on composite tokens created
    min_bigram_count: int = 10           # do not merge bigrams below this
    min_sequences_containing: int = 3    # bigram must appear in ≥ this many sequences
    pmi_threshold: float = 0.5           # PMI variant only
    stop_on_size: Optional[int] = None   # stop when vocab reaches this size
    verbose: bool = False
    show_progress: bool = True           # tqdm bar over merge steps


# ────────────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────────────

class SequenceMerger:
    """Abstract base for sequence-level mergers."""

    def __init__(self, config: Optional[SequenceMergerConfig] = None):
        self.cfg = config or SequenceMergerConfig()

    def fit(self, state: TokenizerState) -> TokenizerState:
        """In-place mutation of ``state``; returns it for fluency."""
        kind = type(self).__name__.replace('Merger', '').lower()
        pbar = tqdm(range(self.cfg.max_merges),
                     desc=f'{kind} merge',
                     disable=not self.cfg.show_progress,
                     leave=False)
        for step_idx in pbar:
            before = state.composite_vocab_size
            if not self._step(state):
                break
            pbar.set_postfix(vocab=state.vocab_size,
                              composites=state.composite_vocab_size)
        pbar.close()
        return state

    def _step(self, state: TokenizerState) -> bool:
        """One merge step. Returns False if no more candidates."""
        raise NotImplementedError

    # shared helpers for subclasses

    def _per_seq_occurrence_count(self,
                                     sequences: list[list[int]],
                                     pair: tuple[int, int]) -> int:
        """How many sequences contain at least one occurrence of ``pair``."""
        a, b = pair
        count = 0
        for seq in sequences:
            for p in pairwise(seq):
                if p == (a, b):
                    count += 1
                    break
        return count

    def _commit_merge(self,
                       state: TokenizerState,
                       pair: tuple[int, int],
                       stat: float,
                       kind_note: str) -> int:
        """Allocate a new composite token, rewrite sequences, log op.
        Returns the new token id.

        Robustness:
        - Centroid / members computed **from the pair directly**, not from a
          deep traversal of atomic descendants — so subsequent removal of
          atomic ids (by acoustic ops) cannot corrupt this composite.
        - ``children`` is flattened for bookkeeping only; a graceful
          ``_flatten`` treats missing ids as leaves so we never crash on
          orphaned atomic references.
        """
        new_id = state.next_token_id()

        def _flatten(tid: int) -> tuple[int, ...]:
            # Graceful: if the id is no longer registered in state.tokens,
            # treat it as a leaf. This keeps the historical record intact
            # even after acoustic ops removed or merged the atomic away.
            if tid not in state.tokens:
                return (tid,)
            tok = state.tokens[tid]
            if not tok.is_composite:
                return (tid,)
            out: list[int] = []
            for c in tok.children:
                out.extend(_flatten(c))
            return tuple(out)

        children = _flatten(pair[0]) + _flatten(pair[1])
        composite = Token(
            id=new_id,
            centroid=_composite_centroid_from_pair(state, pair[0], pair[1]),
            member_ids=_composite_members_from_pair(state, pair[0], pair[1]),
            is_composite=True,
            children=children,
            meta={"made_from_pair": pair, "stat": float(stat), "kind": kind_note},
        )
        state.tokens[new_id] = composite
        state.sequences = _rewrite_sequences_with_merge(state.sequences, pair, new_id)
        state.log(Operation(
            kind="bpe_merge",
            iteration=state.iteration,
            affected_ids=(pair[0], pair[1], new_id),
            metrics_after={"vocab_size": float(state.vocab_size),
                            "composite_vocab_size": float(state.composite_vocab_size)},
            note=f"{kind_note}: merged {pair[0]} + {pair[1]} -> {new_id}  (stat={stat:.3g})",
        ))
        if self.cfg.verbose:
            print(f"[merge #{state.composite_vocab_size}] {pair[0]} + {pair[1]} -> {new_id}  ({kind_note}={stat:.3g})")
        return new_id


# ────────────────────────────────────────────────────────────────────────────
# BPE — merge most frequent bigram
# ────────────────────────────────────────────────────────────────────────────

class BPEMerger(SequenceMerger):
    """Classical BPE: greedily merge the most frequent bigram.

    Stopping criteria:
      - ``max_merges`` reached
      - the most frequent bigram has count < ``min_bigram_count``
      - fewer than ``min_sequences_containing`` distinct sequences contain it
      - ``stop_on_size`` reached
    """

    def _step(self, state: TokenizerState) -> bool:
        if self.cfg.stop_on_size is not None and state.vocab_size >= self.cfg.stop_on_size:
            return False
        bigrams = _count_bigrams(state.sequences)
        if not bigrams:
            return False
        pair, count = bigrams.most_common(1)[0]
        if count < self.cfg.min_bigram_count:
            return False
        seq_support = self._per_seq_occurrence_count(state.sequences, pair)
        if seq_support < self.cfg.min_sequences_containing:
            return False
        self._commit_merge(state, pair, stat=float(count), kind_note="bpe-freq")
        return True


# ────────────────────────────────────────────────────────────────────────────
# PMI — merge bigram with highest PMI (positive)
# ────────────────────────────────────────────────────────────────────────────

class PMIMerger(SequenceMerger):
    """Mutual-information-based merge: pick (a, b) maximising
    PMI(a, b) = log p(a, b) / (p(a) · p(b)),
    subject to:
      - PMI(a, b) ≥ ``pmi_threshold``
      - count(a, b) ≥ ``min_bigram_count``
      - sequence support ≥ ``min_sequences_containing``

    PMI tends to promote bigrams whose co-occurrence exceeds chance, so it
    handles imbalanced token frequencies more gracefully than raw BPE.
    """

    def _step(self, state: TokenizerState) -> bool:
        if self.cfg.stop_on_size is not None and state.vocab_size >= self.cfg.stop_on_size:
            return False
        bigrams = _count_bigrams(state.sequences)
        unigrams = _count_unigrams(state.sequences)
        if not bigrams or not unigrams:
            return False
        total_bg = sum(bigrams.values())
        total_ug = sum(unigrams.values())

        best_pair = None
        best_pmi = -math.inf
        best_count = 0
        for pair, cnt in bigrams.items():
            if cnt < self.cfg.min_bigram_count:
                continue
            a, b = pair
            p_ab = cnt / total_bg
            p_a = unigrams[a] / total_ug
            p_b = unigrams[b] / total_ug
            if p_a == 0 or p_b == 0:
                continue
            pmi = math.log2(p_ab / (p_a * p_b))
            if pmi > best_pmi:
                best_pair = pair
                best_pmi = pmi
                best_count = cnt
        if best_pair is None or best_pmi < self.cfg.pmi_threshold:
            return False
        seq_support = self._per_seq_occurrence_count(state.sequences, best_pair)
        if seq_support < self.cfg.min_sequences_containing:
            return False
        self._commit_merge(state, best_pair, stat=float(best_pmi), kind_note="pmi")
        return True
