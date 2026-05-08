"""
Unit tests for src/adaptive_tokenizer/.

Run with:
    python -m pytest tests/test_adaptive_tokenizer.py -v

These tests exercise the invariants (state validity after each operation),
not end-to-end performance. Synthetic data is used throughout.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptive_tokenizer import (
    AcousticTokenizer,
    AcousticTokenizerConfig,
    AdaptiveTokenizer,
    AdaptiveTokenizerConfig,
    BPEMerger,
    PMIMerger,
    SequenceMergerConfig,
    Token,
    TokenizerState,
    default_acoustic_only,
    default_bpe,
    full_evaluation,
    hp1_evaluate,
    silhouette_of,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

def _synthetic_clusters(n_per_cluster: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [5, 8], [15, 8]])
    return np.concatenate([c + rng.normal(0, 0.4, size=(n_per_cluster, 2))
                            for c in centers])


@pytest.fixture
def synth_X():
    return _synthetic_clusters()


@pytest.fixture
def synth_sequences(synth_X):
    n = synth_X.shape[0]
    # 24 sequences of length 10
    return [list(range(i * 10, (i + 1) * 10)) for i in range(n // 10)]


# ────────────────────────────────────────────────────────────────────────────
# Invariants — validate() after every operation
# ────────────────────────────────────────────────────────────────────────────

def test_state_validates_after_seed(synth_X):
    tok = AcousticTokenizer(AcousticTokenizerConfig(max_iterations=0))
    state = tok._seed(synth_X, seed_labels=None)
    state.validate()


def test_state_validates_after_fit(synth_X, synth_sequences):
    tok = default_acoustic_only(AcousticTokenizerConfig(max_iterations=2,
                                                           random_state=0))
    state = tok.acoustic.fit(synth_X, sequences_per_file=synth_sequences)
    state.validate()


def test_state_validates_after_bpe(synth_X, synth_sequences):
    tok = default_bpe(
        ac_cfg=AcousticTokenizerConfig(max_iterations=1, random_state=0),
        seq_cfg=SequenceMergerConfig(max_merges=3, min_bigram_count=2,
                                        min_sequences_containing=1),
    )
    state = tok.fit(synth_X, sequences_per_file=synth_sequences)
    state.validate()
    assert state.composite_vocab_size >= 1


# ────────────────────────────────────────────────────────────────────────────
# Seed correctness
# ────────────────────────────────────────────────────────────────────────────

def test_seed_from_existing_labels(synth_X):
    labels = np.array([0] * 60 + [1] * 60 + [2] * 60 + [3] * 60)
    tok = AcousticTokenizer(AcousticTokenizerConfig(max_iterations=0))
    state = tok._seed(synth_X, seed_labels=labels)
    assert state.vocab_size == 4
    assert set(state.tokens.keys()) == {0, 1, 2, 3}
    np.testing.assert_array_equal(state.labels, labels)


def test_seed_hdbscan_finds_clusters(synth_X):
    tok = default_acoustic_only(AcousticTokenizerConfig(
        seed_min_cluster_frac=0.05, seed_min_samples=10,
        max_iterations=0, random_state=0,
    ))
    state = tok.acoustic.fit(synth_X)
    # should recover 4 clusters on well-separated synthetic data
    assert state.vocab_size >= 3
    assert silhouette_of(state, synth_X) > 0.5


# ────────────────────────────────────────────────────────────────────────────
# Individual operations
# ────────────────────────────────────────────────────────────────────────────

def test_merge_consolidates_close_clusters():
    """Two very close synthetic clusters should be merged."""
    rng = np.random.default_rng(0)
    X = np.concatenate([
        rng.normal([0, 0], 0.3, size=(50, 2)),
        rng.normal([0.5, 0], 0.3, size=(50, 2)),   # very close → mergeable
        rng.normal([10, 0], 0.3, size=(50, 2)),    # far away → not mergeable
    ])
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
    cfg = AcousticTokenizerConfig(
        enable_split=False, enable_add=False, enable_prune=False,
        merge_distance_quantile=0.4,   # permissive
        merge_silhouette_tolerance=0.1,
        max_iterations=3, random_state=0,
    )
    tok = AcousticTokenizer(cfg)
    state = tok.fit(X, seed_labels=labels)
    state.validate()
    assert state.vocab_size == 2, f"expected 2 after merge, got {state.vocab_size}"


def test_prune_removes_small_cluster():
    rng = np.random.default_rng(0)
    X = np.concatenate([
        rng.normal([0, 0], 0.3, size=(100, 2)),
        rng.normal([10, 0], 0.3, size=(100, 2)),
        rng.normal([5, 5], 0.3, size=(5, 2)),    # tiny cluster → prune
    ])
    labels = np.array([0] * 100 + [1] * 100 + [2] * 5)
    cfg = AcousticTokenizerConfig(
        enable_split=False, enable_merge=False, enable_add=False,
        prune_min_size=10,
        max_iterations=1, random_state=0,
    )
    state = AcousticTokenizer(cfg).fit(X, seed_labels=labels)
    state.validate()
    assert 2 not in state.tokens
    assert state.vocab_size == 2


# ────────────────────────────────────────────────────────────────────────────
# BPE merger
# ────────────────────────────────────────────────────────────────────────────

def test_bpe_creates_composite_and_rewrites():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(60, 2))
    labels = np.array([0, 1] * 30)
    # 10 sequences, pattern 0-1 repeating is the obvious bigram
    seqs = [list(range(i * 6, (i + 1) * 6)) for i in range(10)]
    state = AcousticTokenizer(AcousticTokenizerConfig(
        enable_split=False, enable_merge=False, enable_add=False, enable_prune=False,
        max_iterations=0,
    )).fit(X, seed_labels=labels, sequences_per_file=seqs)
    bpe = BPEMerger(SequenceMergerConfig(max_merges=1, min_bigram_count=1,
                                           min_sequences_containing=1))
    bpe.fit(state)
    state.validate()
    assert state.composite_vocab_size >= 1
    # every composite must have non-empty children, all atomic
    composites = [t for t in state.tokens.values() if t.is_composite]
    for c in composites:
        assert len(c.children) >= 2
        for child in c.children:
            assert child in state.tokens
            assert not state.tokens[child].is_composite


def test_pmi_prefers_correlated_pairs():
    """Construct two bigrams: one rare-but-correlated, one frequent-but-chance."""
    # 4 atomic tokens; pairs (0,1) always together; (2,3) sometimes co-occur.
    # Raw frequency of (0,1) < raw freq of (2,2) (lots of self-loops for token 2)
    # but PMI of (0,1) is much higher.
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(40, 2))
    labels = np.array([0, 1, 2, 3, 2, 2, 2, 2] * 5)
    seqs = [list(range(i * 8, (i + 1) * 8)) for i in range(5)]
    state = AcousticTokenizer(AcousticTokenizerConfig(
        enable_split=False, enable_merge=False, enable_add=False, enable_prune=False,
        max_iterations=0,
    )).fit(X, seed_labels=labels, sequences_per_file=seqs)
    pmi = PMIMerger(SequenceMergerConfig(max_merges=1, min_bigram_count=3,
                                           min_sequences_containing=1,
                                           pmi_threshold=0.0))
    pmi.fit(state)
    state.validate()
    composites = [t for t in state.tokens.values() if t.is_composite]
    assert len(composites) >= 1
    # the picked pair should be (0, 1) — high PMI
    picked = composites[0].children
    assert picked == (0, 1), f"PMI picked {picked}, expected (0, 1)"


# ────────────────────────────────────────────────────────────────────────────
# Evaluation smoke
# ────────────────────────────────────────────────────────────────────────────

def test_full_evaluation_runs(synth_X, synth_sequences):
    tok = default_acoustic_only(AcousticTokenizerConfig(
        seed_min_cluster_frac=0.05, seed_min_samples=10,
        max_iterations=1, random_state=0,
    ))
    state = tok.acoustic.fit(synth_X, sequences_per_file=synth_sequences)
    # 4 true clusters -> fake per-sequence context aligned
    ctx_seq = []
    for seq in synth_sequences:
        first_seg = seq[0]
        ctx_seq.append(first_seg // 60)
    ctx_seg = np.array([i // 60 for i in range(synth_X.shape[0])])
    res = full_evaluation(
        state, embedding=synth_X,
        contexts_per_segment=ctx_seg,
        contexts_per_sequence=ctx_seq,
        run_hp1=True, hp1_feature_bundles=('bos',),
        random_state=0,
    )
    assert "silhouette" in res.metrics
    assert res.metrics["silhouette"] > 0.3
    assert "hp1_f1_original_bos" in res.metrics


def test_iterative_adaptive_does_not_crash_when_acoustic_ops_touch_composites():
    """Regression for the KeyError seen when an outer iteration does:
       (a) BPE merges atomic 0 + 1 -> composite C
       (b) next acoustic pass tries to prune or merge atomic 0 / 1
    Expected: locked_atomic_ids protects the children; no KeyError."""
    rng = np.random.default_rng(0)
    # 5 well-separated clusters, 100 seqs of length 10 with strong bigram
    X = np.concatenate([
        rng.normal([0, 0], 0.4, size=(100, 2)),
        rng.normal([10, 0], 0.4, size=(100, 2)),
        rng.normal([5, 8], 0.4, size=(100, 2)),
        rng.normal([15, 8], 0.4, size=(100, 2)),
        rng.normal([0, 15], 0.4, size=(100, 2)),
    ])
    n = X.shape[0]
    seqs = [list(range(i * 10, (i + 1) * 10)) for i in range(n // 10)]
    cfg = AdaptiveTokenizerConfig(
        acoustic=AcousticTokenizerConfig(
            seed_min_cluster_frac=0.05,
            seed_min_samples=10,
            split_silhouette_threshold=0.40,
            add_outlier_quantile=0.95,
            add_min_size=15,
            prune_min_size=15,
            max_iterations=2,
            random_state=0,
            show_progress=False,
        ),
        sequence=SequenceMergerConfig(
            max_merges=5, min_bigram_count=2, min_sequences_containing=1,
            show_progress=False,
        ),
        sequence_kind='bpe',
        max_outer_iters=3,
        verbose=False,
    )
    tok = AdaptiveTokenizer(cfg)
    state = tok.fit(X, sequences_per_file=seqs)     # should NOT raise
    state.validate()
    # At least one composite should have been created; atomic children
    # should still be present in state.tokens (they're "locked").
    composites = [t for t in state.tokens.values() if t.is_composite]
    assert len(composites) >= 1
    for c in composites:
        for child in c.children:
            assert child in state.tokens, (
                f"composite {c.id} has child {child} missing from state.tokens")


def test_hp1_evaluate_delta_zero_on_random_sequences():
    """On truly random sequences with unrelated contexts, Δ should be ~0."""
    rng = np.random.default_rng(0)
    sequences = [list(rng.integers(0, 5, size=10)) for _ in range(200)]
    contexts = list(rng.integers(0, 3, size=200))
    state = TokenizerState(
        tokens={i: Token(id=i, centroid=np.zeros(2), member_ids=np.array([]))
                for i in range(5)},
        labels=np.zeros(1000, dtype=int),
        sequences=sequences,
    )
    res = hp1_evaluate(state, contexts, feature_bundle='bos', random_state=0, permute=True)
    assert abs(res['hp1_f1_delta']) < 0.05
