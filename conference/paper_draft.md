# Does token order matter? A controlled study of tokenized animal vocalizations

*Working draft — NeurIPS Workshop on AI for Non-Human Animal Communication (or sibling venue). Non-archival.*

## Abstract

A growing line of work tokenizes animal vocalizations into discrete sequences and
applies sequence models, motivated by the premise that token **order** carries
communicative information. We test this premise directly and rigorously. Across two
species — Egyptian fruit bats (*Rousettus aegyptiacus*; graded vocal system; Prat
2017) and common marmosets (*Callithrix jacchus*; structured calls;
InfantMarmosetsVox) — and three sequence-model families (a small BERT with
masked-token pretraining, k-NN with Levenshtein distance, and an order-agnostic
bag-of-tokens classifier), we compare classification of behavioral context, call-type,
and caller identity on **real vs. within-vocalization shuffled** token order, with
cross-individual splits and confidence intervals. We find: (1) **token-order effects
are small in both species** (Δ macro-F1 ≈ 0.01–0.03) — the token *multiset* dominates
classification; (2) **sequence length is a major confound** that prior work ignores:
order is statistically indistinguishable from shuffled at short sequences in both
species, with any effect emerging only at longer lengths; (3) **pooled across length,
marmosets show a small but significant order benefit while bats show none**, but the
effect is not robust per length-band, so we report it as a modest, length-modulated
species contrast rather than a clean dissociation. Separately, we show a constructive
result: a **domain-matched self-supervised (SSL) tokenizer** is the best discrete
repertoire for bats — surpassing a per-context DP-GMM density baseline and a
mel-UMAP k-means tokenizer on context macro-F1, next-token perplexity, and agreement
with an independent acoustic reference, *despite lower silhouette* (a reminder that
silhouette is a poor quality proxy). Finally, we flag a leakage pitfall: per-context
token vocabularies trivially encode the label for any token-sequence classifier. Our
controls (shuffle ablation, length stratification, cross-individual splits, CIs) are
absent from recent token-sequence work and temper its central premise.

## 1. Introduction

Computational bioacoustics increasingly borrows the discrete-token, sequence-model
toolkit from NLP: vocalizations are segmented, quantized into tokens, and modeled with
edit-distance or transformer classifiers. The implicit assumption is that the *order*
of tokens within a vocalization carries information — i.e., that animal communication
has combinatorial syntax that sequence models can exploit. Recent work (Sarkar et al.,
2025) tokenizes marmoset and dog calls and applies k-NN with Levenshtein distance,
explicitly framing the goal as *leveraging sequential structure*. However, such work
typically (i) never isolates whether order itself helps (no shuffle control), (ii)
reports no confidence intervals, (iii) does not control for sequence length, and (iv)
does not study behavioral context or graded vocal systems such as bats.

We provide the missing controls. Our contributions:

1. A **shuffle-controlled order ablation** across two species, three sequence-model
   families, and three tasks (behavioral context, call-type, caller), with
   cross-individual splits and CIs.
2. Identification of **sequence length as a confound**: order is irrelevant at short
   sequences in both species; reported effects depend on length.
3. A **modest, length-modulated species contrast**: pooled, marmosets (structured
   calls) show a small significant order benefit; bats (graded) show none.
4. A **constructive SSL tokenizer** that is the best discrete repertoire for bats, and
   evidence that silhouette mis-ranks tokenizer quality.
5. A **leakage caveat** for per-context tokenization in sequence classifiers.

## 2. Related work

Unsupervised bioacoustic tokenization (Sainburg 2020; Goffinet 2021; Assom 2025) builds
a single global token vocabulary via UMAP+HDBSCAN or DP-GMM, discarding behavioral
context. Discrete-token sequence analysis of animal calls (Sarkar & Magimai-Doss 2025)
uses VQ of HuBERT frames + k-NN/Levenshtein, finding discrete tokens underperform
continuous baselines and suggesting "more sophisticated sequence modeling" as future
work. Self-supervised + deep-clustering pipelines for time series (SensorSCAN;
Golyadkin, Pozdnyakov, Zhukov, Makarov 2023) motivate our domain-matched SSL tokenizer.
Associative vs. combinatorial syntax (Townsend 2020) and permutation tests (Farine
2022) frame the order question; we operationalize it with neural shuffle controls.

## 3. Data and methods

**Corpora.** Egyptian fruit bats (Prat 2017): 153k segments, 8 behavioral contexts,
41 individuals, 250 kHz. Marmosets (InfantMarmosetsVox): 72,921 calls, 11 call-types,
10 callers, 44.1 kHz. Sub-units: bat *segments* (mel 21×32) or *frames* (per-segment
21 mel frames); marmoset mel-spectrogram *frames* per call.

**Tokenizers.** k-means on mel/UMAP features; per-context k-means/DP-GMM; agglomerative;
and a **domain-matched SSL** encoder (NT-Xent contrastive, positives = two sub-units of
the same vocalization, no labels) followed by k-means. Vocabulary V swept in [10,120].

**Order control.** For each vocalization, tokens are kept in temporal order (real) or
randomly permuted within the vocalization (shuffled), preserving the multiset. Models
are trained and evaluated in the same regime; Δ = macro-F1(real) − macro-F1(shuffled).

**Models.** TinyBERT (2 layers, d=64) with masked-token pretraining + [CLS] fine-tuning;
k-NN + normalized Levenshtein (replicating Sarkar et al.); bag-of-tokens logistic
regression (order-agnostic lower bound). Causal TinyBERT for next-token perplexity.

**Protocol.** Bats: cross-individual (30 train / 11 test emitters), 5 seeds. Marmosets:
stratified random split (caller-ID is closed-set), 5 seeds. Primary metric macro-F1;
95% CIs via t over seeds. bits-per-token for next-token.

## 4. Results

### 4.1 Token order is (nearly) irrelevant for bats

| tokenizer | model | task | Δ(real−shuf) 95% CI |
|---|---|---|---|
| mel-UMAP k-means | BERT | context | [−0.004, +0.004] n.s. |
| agglomerative | BERT | context | [−0.003, +0.002] n.s. |
| mel-UMAP k-means | kNN-Levenshtein | context | [−0.012, +0.023] n.s. |
| SSL k-means | BERT | context | [−0.002, +0.002] n.s. |
| mel-UMAP k-means | BERT | caller | [−0.006, +0.003] n.s. |
| frame-level (len ~78) | BERT | context | [−0.005, +0.021] n.s. |

Order is statistically indistinguishable from shuffled across tokenizers, models, and
tasks — including at long frame-level sequences. bag ≈ BERT, so the multiset carries
the signal. Intrinsically, real order is only marginally more next-token-predictable
than shuffled (~0.04 bits/token, CI>0): weak phonotactic but not semantic structure.

### 4.2 Sequence length is a confound; marmosets show a small order benefit

Marmoset call-type, length-stratified (5 seeds): Δ = +0.002 (len 3) n.s.; +0.010
(len 22) sig; +0.031 (len 48) n.s. (high variance). Bat frame-level long (len 78):
+0.008 n.s. **At the matched 41–96 band both species are n.s.** Pooled over length,
marmosets show a significant order benefit (call-type +0.029 [+0.018,+0.041]; caller
+0.030 [+0.018,+0.042]); bats show none. We therefore report a *small, length-modulated
species contrast*, not a clean dissociation: order effects are small everywhere, the
multiset dominates, and length must be controlled before attributing structure.

### 4.3 A domain-matched SSL tokenizer is the best discrete repertoire (bats)

| method | context macro-F1 |
|---|---|
| Assom baseline | 0.237 |
| mel-UMAP k-means | 0.276 |
| per-context DP-GMM (prior best generative) | 0.313 |
| **SSL k-means (V=120)** | **0.336** |
| SSL continuous (same encoder) | 0.362 |

SSL wins at every V (+0.07–0.12 over mel-UMAP); 10 SSL tokens ≈ 110 per-context tokens;
discretization cost only +0.025 (vs ~0.11 for mel-UMAP). SSL tokens also agree more
with an independent DTW acoustic reference (ARI 0.048 vs 0.041; NMI 0.454 vs 0.441)
*despite lower silhouette* (0.088 vs 0.191) — silhouette mis-ranks quality.
Cross-species replication (marmosets): _SSL vs mel-frame tokens — pending._

### 4.4 Leakage caveat

Per-context tokenization assigns disjoint token-ID ranges per context, so a
token-sequence classifier reads the label off the ID range: bag-LR reaches macro-F1
≈ 1.000. Per-context tokenizers must be evaluated by generative max-likelihood, not by
token-sequence classifiers.

## 5. Discussion & limitations

Our headline is cautionary and constructive: (a) token-order effects in tokenized
animal vocalizations are small and confounded by sequence length; "leverage sequential
structure" claims need shuffle and length controls; (b) the token multiset, not order,
drives classification; (c) a domain-matched SSL tokenizer is a strong, near-lossless
discrete representation when discrete symbols are needed. Limitations: small effect
sizes limit per-band power; two species only; bat vs marmoset featurization is
analogous but not identical; marmoset caller-ID is closed-set; bat order at long
frame-level may be underpowered. These do not affect the main, well-powered pooled
comparisons.

## 6. Conclusion

Token order contributes little to behavioral-context, call-type, or caller decoding in
two species of tokenized vocalizations; the multiset dominates, and apparent order
effects are small and length-dependent. A domain-matched SSL tokenizer is the best
discrete repertoire. We urge shuffle/length/CI controls and label-leakage checks in
animal-vocalization token-sequence research.
