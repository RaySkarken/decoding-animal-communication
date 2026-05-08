"""Per-context tokenizer family with Bayes classification by maximum likelihood.

Implements the method described in Chapter 2 of the thesis: given a training
corpus of segments with context labels, fit one generative tokenizer per
context, and at inference predict the unknown context by argmax of the
class-conditional log-likelihood.

Three tokenizer variants are supported as alternative implementations of
step 3 in §2.2 of the methodology, all unified by a common interface
(a `log_likelihood(X)` method):

    - DPGMMTokenizer:    Dirichlet-Process Gaussian Mixture (most direct)
    - HDBSCANTokenizer:  HDBSCAN clusters + post-hoc Gaussian approximation
    - KMeansTokenizer:   k-means prototypes + softmax over squared distances

The exported class `PerContextFamily` wraps a dict ``{c: T_c}`` and exposes
`fit` and `predict_context` for end-to-end training/inference.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers: Gaussian mixture log-likelihood
# ---------------------------------------------------------------------------

def _gauss_logpdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Log N(x | mu, Sigma) per row, with jitter for singular Sigma."""
    D = X.shape[1]
    S = Sigma + 1e-6 * np.eye(D)
    sign, logdet = np.linalg.slogdet(S)
    inv = np.linalg.inv(S)
    diff = X - mu
    mahal = np.einsum('ij,jk,ik->i', diff, inv, diff)
    return -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)


def _logmix_logpdf(components: List[tuple], X: np.ndarray) -> np.ndarray:
    """log sum_k pi_k N(x | mu_k, Sigma_k) per row of X."""
    if not components:
        return np.full(len(X), -1e10)
    logs = np.stack([np.log(max(p, 1e-12)) + _gauss_logpdf(X, mu, S)
                     for (p, mu, S) in components], axis=1)
    m = logs.max(axis=1)
    return m + np.log(np.exp(logs - m[:, None]).sum(axis=1))


# ---------------------------------------------------------------------------
# Tokenizer variants
# ---------------------------------------------------------------------------

class _BaseTokenizer:
    """Common interface for all per-context tokenizers.

    After `fit`, each instance must expose:
        - `n_prototypes: int`
        - `log_likelihood(X: np.ndarray) -> np.ndarray`
        - `prototype_weights: np.ndarray`  (shape (K,), sums to 1)
        - `prototype_centers: np.ndarray`  (shape (K, d))
    """

    n_prototypes: int
    prototype_weights: np.ndarray
    prototype_centers: np.ndarray

    def fit(self, X: np.ndarray) -> '_BaseTokenizer':
        raise NotImplementedError

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DPGMMTokenizer(_BaseTokenizer):
    """Dirichlet-Process Gaussian Mixture tokenizer.

    Number of active components is data-driven: ``n_components`` is an upper
    bound and the concentration prior ``weight_concentration_prior`` controls
    how many components end up with non-negligible mass. This is the most
    direct way to compute p_c(x) used in the Bayes classification rule.
    """

    def __init__(self, n_components: int = 15,
                 weight_concentration_prior: float = 0.1,
                 max_iter: int = 150, random_state: int = 0):
        self.n_components = n_components
        self.weight_concentration_prior = weight_concentration_prior
        self.max_iter = max_iter
        self.random_state = random_state
        self._bgm: Any = None

    def fit(self, X: np.ndarray) -> 'DPGMMTokenizer':
        from sklearn.mixture import BayesianGaussianMixture
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm = BayesianGaussianMixture(
                n_components=self.n_components,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.weight_concentration_prior,
                covariance_type='full',
                max_iter=self.max_iter,
                random_state=self.random_state,
            ).fit(X)
        self.prototype_weights = np.asarray(self._bgm.weights_)
        self.prototype_centers = np.asarray(self._bgm.means_)
        # active components: those with non-trivial weight
        self.n_prototypes = int((self.prototype_weights > 1e-3).sum())
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return self._bgm.score_samples(X)


class HDBSCANTokenizer(_BaseTokenizer):
    """HDBSCAN clusters with post-hoc Gaussian approximation.

    HDBSCAN alone does not produce a probability density, so after clustering
    we fit a Gaussian to each cluster (mean + empirical covariance) and use
    a weighted mixture of those Gaussians for likelihood.
    """

    def __init__(self, min_cluster_size: Optional[int] = None,
                 min_samples: int = 10,
                 cluster_selection_epsilon: float = 0.05):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self._components: List[tuple] = []

    def fit(self, X: np.ndarray) -> 'HDBSCANTokenizer':
        from sklearn.cluster import HDBSCAN
        mcs = self.min_cluster_size or max(20, int(0.02 * len(X)))
        hdb = HDBSCAN(min_cluster_size=mcs,
                       min_samples=self.min_samples,
                       cluster_selection_epsilon=self.cluster_selection_epsilon)
        hdb.fit(X)
        comps: List[tuple] = []
        for k in sorted(set(hdb.labels_)):
            if k < 0:
                continue
            m = hdb.labels_ == k
            if m.sum() < 5:
                continue
            Xi = X[m]
            mu = Xi.mean(axis=0)
            if Xi.shape[0] > 1:
                Sigma = np.cov(Xi.T)
            else:
                Sigma = np.eye(X.shape[1]) * 0.1
            comps.append((m.sum() / len(X), mu, Sigma))
        if not comps:
            mu = X.mean(axis=0)
            Sigma = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
            comps.append((1.0, mu, Sigma))
        self._components = comps
        self.n_prototypes = len(comps)
        self.prototype_weights = np.array([c[0] for c in comps])
        self.prototype_centers = np.stack([c[1] for c in comps])
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return _logmix_logpdf(self._components, X)


class KMeansTokenizer(_BaseTokenizer):
    """k-means prototypes with softmax over squared distances as density.

    Effectively a Gaussian mixture with a single shared isotropic covariance
    ``sigma^2 I`` estimated from mean within-cluster variance. Simplest of the
    three variants, no explicit density fitting.
    """

    def __init__(self, n_clusters: int = 15, random_state: int = 0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._components: List[tuple] = []

    def fit(self, X: np.ndarray) -> 'KMeansTokenizer':
        from sklearn.cluster import KMeans
        K = min(self.n_clusters, max(2, len(X) // 10))
        km = KMeans(n_clusters=K, random_state=self.random_state, n_init=10).fit(X)
        comps: List[tuple] = []
        for k in range(K):
            m = km.labels_ == k
            if m.sum() < 2:
                continue
            Xi = X[m]
            mu = Xi.mean(axis=0)
            Sigma = np.cov(Xi.T) if Xi.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
            comps.append((m.sum() / len(X), mu, Sigma))
        self._components = comps
        self.n_prototypes = len(comps)
        self.prototype_weights = np.array([c[0] for c in comps])
        self.prototype_centers = np.stack([c[1] for c in comps])
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return _logmix_logpdf(self._components, X)


# ---------------------------------------------------------------------------
# Family of per-context tokenizers + Bayes classifier
# ---------------------------------------------------------------------------

TOKENIZER_REGISTRY: Dict[str, type] = {
    'dpgmm': DPGMMTokenizer,
    'hdbscan': HDBSCANTokenizer,
    'kmeans': KMeansTokenizer,
}


@dataclass
class PerContextFamily:
    """A collection {T_c} of context-conditional tokenizers with Bayes rule.

    Parameters
    ----------
    variant : {'dpgmm', 'hdbscan', 'kmeans'}
        Which tokenizer class to use for each T_c.
    prior : {'empirical', 'uniform'}
        Class prior p(c) in the Bayes rule.
    min_segments_per_context : int
        Minimum number of training segments required to fit T_c for a context.
    kwargs : dict
        Passed to the tokenizer constructor.
    """

    variant: str = 'dpgmm'
    prior: str = 'empirical'
    min_segments_per_context: int = 30
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    tokenizers: Dict[int, _BaseTokenizer] = field(default_factory=dict, init=False)
    log_prior: Dict[int, float] = field(default_factory=dict, init=False)
    contexts: List[int] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, contexts: Sequence[int],
            seed: int = 0,
            prior_counts: Optional[Dict[int, int]] = None) -> 'PerContextFamily':
        """Fit T_c for each c in `contexts` using segments where y == c.

        Parameters
        ----------
        X, y : arrays of segments and their context labels
        contexts : iterable of context ids to fit
        seed : random state used by tokenizers that support one
        prior_counts : optional {c: count} that overrides empirical segment
            counts when computing log p(c). Use this when the unit of
            classification is different from the unit of X/y rows (e.g. you
            fit on segments but classify whole vocalizations; pass the
            vocalization-level counts).
        """
        if self.variant not in TOKENIZER_REGISTRY:
            raise ValueError(f'Unknown variant: {self.variant}')
        tok_cls = TOKENIZER_REGISTRY[self.variant]
        kwargs = dict(self.tokenizer_kwargs)
        if 'random_state' in tok_cls.__init__.__code__.co_varnames:
            kwargs.setdefault('random_state', seed)

        fit_counts = {c: int((y == c).sum()) for c in contexts}
        if prior_counts is None:
            prior_counts = fit_counts
        total_prior = max(sum(prior_counts.values()), 1)
        for c in contexts:
            if fit_counts[c] < self.min_segments_per_context:
                continue
            self.tokenizers[c] = tok_cls(**kwargs).fit(X[y == c])
            if self.prior == 'uniform':
                self.log_prior[c] = 0.0
            else:
                self.log_prior[c] = float(np.log(max(prior_counts.get(c, 1), 1) / total_prior))
        self.contexts = sorted(self.tokenizers.keys())
        return self

    def log_likelihood_per_context(self, X_seq: np.ndarray) -> Dict[int, float]:
        """Return {c: sum_i log p_c(x_i)} for a sequence of segments."""
        return {c: float(tok.log_likelihood(X_seq).sum())
                for c, tok in self.tokenizers.items()}

    def predict_context(self, X_seq: np.ndarray) -> int:
        """Argmax-Bayes over contexts for one sequence. Raises if no fitted tokenizer."""
        if not self.tokenizers:
            raise RuntimeError('No tokenizers fitted; call .fit first')
        best_c, best_score = None, -np.inf
        for c, tok in self.tokenizers.items():
            score = tok.log_likelihood(X_seq).sum() + self.log_prior[c]
            if score > best_score:
                best_score = score
                best_c = c
        return int(best_c)

    def predict_many(self, sequences: Iterable[np.ndarray]) -> np.ndarray:
        return np.array([self.predict_context(s) for s in sequences], dtype=int)

    @property
    def vocabulary_sizes(self) -> Dict[int, int]:
        return {c: tok.n_prototypes for c, tok in self.tokenizers.items()}
