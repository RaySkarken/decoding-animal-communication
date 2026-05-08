"""
Adaptive tokenization for animal vocal communication.

Public API:

    from src.adaptive_tokenizer import (
        AcousticTokenizer, AcousticTokenizerConfig,
        BPEMerger, PMIMerger, SequenceMergerConfig,
        AdaptiveTokenizer, AdaptiveTokenizerConfig,
        TokenizerState, Token, Operation,
        full_evaluation, hp1_evaluate, per_emitter_proxy_agreement,
        default_acoustic_only, default_bpe, default_pmi,
    )

See ``docs/thesis_draft.md`` §4.4 for a description of the pipeline, and
``notebooks/adaptive_tokenizer_experiment.ipynb`` for the runner.
"""

from .acoustic import AcousticTokenizer, AcousticTokenizerConfig
from .adaptive import (
    AdaptiveTokenizer,
    AdaptiveTokenizerConfig,
    default_acoustic_only,
    default_bpe,
    default_pmi,
)
from .evaluation import (
    EvaluationResult,
    full_evaluation,
    hp1_evaluate,
    per_emitter_proxy_agreement,
    silhouette_of,
    noise_fraction_of,
    stability,
)
from .sequence import BPEMerger, PMIMerger, SequenceMergerConfig
from .types import Operation, Token, TokenizerState

__all__ = [
    # acoustic
    "AcousticTokenizer", "AcousticTokenizerConfig",
    # sequence
    "BPEMerger", "PMIMerger", "SequenceMergerConfig",
    # adaptive
    "AdaptiveTokenizer", "AdaptiveTokenizerConfig",
    "default_acoustic_only", "default_bpe", "default_pmi",
    # evaluation
    "EvaluationResult",
    "full_evaluation", "hp1_evaluate", "per_emitter_proxy_agreement",
    "silhouette_of", "noise_fraction_of", "stability",
    # types
    "Operation", "Token", "TokenizerState",
]
