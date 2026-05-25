# Conference paper — running findings & skeleton

**Working title:** *Does sequential structure help? A controlled study of tokenized
bat vocalizations for behavioral context and caller identity.*

**Target:** NeurIPS Workshop on AI for Non-Human Animal Communication (recurring;
non-archival) or sibling bioacoustics venue.

## Draft abstract
Recent work tokenizes animal vocalizations into discrete sequences and applies
sequence models, *assuming* token order carries communicative information. We test
this assumption rigorously on Egyptian fruit bats (Prat 2017) for two tasks —
behavioral context and caller identity — with within-vocalization shuffle controls,
cross-individual splits, and confidence intervals (controls absent from prior work).
Across four tokenizers (mel-UMAP, agglomerative, per-context, and a domain-matched
SSL tokenizer) and three model families (BERT, kNN+Levenshtein, bag-of-tokens),
token **order is task-irrelevant**: shuffling within a vocalization does not change
context or caller macro F1 (all 95% CIs on Δ contain 0). Intrinsically, real order
is only marginally more next-token-predictable than shuffled (~0.04 bits/token),
revealing weak **phonotactic** but not **semantic** structure — associative, not
combinatorial, syntax, *measured* rather than assumed. Constructively, a
domain-matched self-supervised tokenizer (NT-Xent on mel segments) is the best
discrete repertoire (context macro F1 0.34 vs 0.31 per-context DP-GMM, 0.28 mel-UMAP,
robust across V∈[10,120]) and makes discretization nearly lossless (cost +0.025 vs
its own continuous probe, vs ~0.11 for hand-pipeline tokens). We also flag a leakage
pitfall: per-context token-IDs trivially encode the label for sequence classifiers.

## Master table (context macro F1, cross-bat 30/11, 5 seeds)
| method | macro F1 | notes |
|---|---|---|
| Assom baseline (thesis) | 0.237 | global HDBSCAN + RF |
| mel-UMAP k-means + bag/BERT | 0.230 / 0.276 | hand pipeline |
| per-context DP-GMM (thesis best generative) | 0.313 | density, max-likelihood |
| **SSL tokens (V=120) bag-LR** | **0.336** | discrete, advisor lineage |
| SSL continuous (same encoder) | 0.362 | discretization cost only +0.025 |
| (thesis SSL linear probe, richer) | 0.385 | continuous ceiling |

**Positioning vs SOTA (Sarkar et al., arXiv 2511.10190, NeurIPS-WS 2025):**
they tokenize animal calls (HuBERT→VQ) and classify call-type/caller with
kNN+Levenshtein, claiming to "leverage sequential structure" — but with **no order
control, no CIs, no behavioral context, no bats**, and their discrete tokens lose
to continuous baselines. We supply exactly those missing controls on a new corpus
and task.

---

## Results (Egyptian fruit bat, Prat 2017; cross-bat 30/11; 5 seeds; macro F1)

### Finding 1 — token ORDER carries no usable signal (robust negative)
Real vs within-vocalization SHUFFLED order, Δ(real−shuf) with 95% CI:

| tokenizer | model | task | real | shuf | Δ 95%CI |
|---|---|---|---|---|---|
| mel-UMAP k-means30 | BERT(MLM+CLS) | context | 0.276 | 0.276 | [−0.004,+0.004] n.s. |
| agglomerative30 | BERT | context | 0.271 | 0.271 | [−0.003,+0.002] n.s. |
| mel-UMAP k-means30 | kNN+Levenshtein | context | 0.270 | 0.265 | [−0.012,+0.023] n.s. |
| agglomerative30 | kNN+Levenshtein | context | 0.260 | 0.260 | [−0.010,+0.009] n.s. |
| SSL k-means30 | BERT | context | 0.332 | 0.332 | [−0.002,+0.002] n.s. |
| mel-UMAP k-means30 | BERT | caller-ID | 0.162 | 0.163 | [−0.006,+0.003] n.s. |

→ Order n.s. across **tokenizers × model families × tasks** for classification.
Signal is in the token **multiset** (bag-of-tokens ≈ BERT).

### Finding 1b — but weak phonotactic structure EXISTS (the dissociation)
Intrinsic next-token predictability, real vs shuffled order (bits/token; lower=better):

| tokenizer | real bpt | shuf bpt | Δ(shuf−real) 95%CI |
|---|---|---|---|
| mel-UMAP k-means30 | 4.191 | 4.233 | [+0.034,+0.050] **predictable** |
| SSL k-means30 | 2.869 | 2.905 | [+0.024,+0.047] **predictable** |

→ Real order is significantly (CI excludes 0) more next-token-predictable than
shuffled, but the effect is tiny (~0.04 bpt, ~1%). **Dissociation:** sequential
structure is measurable (phonotactic) yet **task-irrelevant** (order n.s. for
context/caller). Associative, not combinatorial, syntax — with the controls prior
work lacked. (SSL bpt 2.87 ≪ mel 4.19 → SSL tokenizer also more predictable.)

### Finding 2 — domain-matched SSL tokenizer is the best DISCRETE repertoire (positive)
| method | macro F1 (context) | note |
|---|---|---|
| Assom baseline (thesis) | 0.237 | global HDBSCAN + RF |
| mel-UMAP k-means + BERT | 0.276 | hand pipeline |
| per-context DP-GMM (thesis best generative) | 0.313 | density, max-likelihood |
| **SSL tokens + bag-LR / BERT** | **0.326 / 0.332** | NT-Xent, advisor lineage |
| SSL continuous (linear probe) | 0.385 | ceiling (no discretization) |

→ SSL tokenization > prior discrete/generative methods; closes most of the
discrete→continuous gap. Recommended tokenizer when discrete symbols are required
(n-gram LM, MDL, cross-species alignment).

**Robust across vocabulary size** (bag-LR, context macro F1, cross-bat 5 seeds):

| V | mel-UMAP | SSL | gap |
|---|---|---|---|
| 10 | 0.188 | 0.311 | +0.123 |
| 15 | 0.217 | 0.321 | +0.104 |
| 30 | 0.230 | 0.326 | +0.096 |
| 60 | 0.240 | 0.324 | +0.084 |
| 120 | 0.263 | **0.335** | +0.072 |

→ SSL wins at every V (+0.07–0.12). Efficiency: SSL with **10 tokens** (0.311) ≈
thesis per-context DP-GMM with ~110 tokens (0.313).

**Discretization is nearly lossless with SSL tokens** (same SSL encoder, cross-bat 5 seeds):
- continuous (mean+std pool → LR): 0.362 [0.339, 0.384]
- discrete (k-means V=120 → bag-LR): 0.336 [0.300, 0.372]
- cost = **+0.025** [+0.006, +0.045] (significant but small)

vs mel-UMAP discretization which loses ~0.11. → A good (SSL) tokenizer makes
discrete symbolic analysis viable for behavioral context.

**Label-free corroboration (agreement with independent DTW acoustic proxy):**
| tokenizer | ARI(proxy) | AMI | NMI | silhouette |
|---|---|---|---|---|
| mel-UMAP V=120 | 0.041 | 0.229 | 0.441 | 0.191 |
| SSL V=120 | **0.048** | **0.247** | **0.454** | 0.088 |

→ SSL tokens agree MORE with independent acoustics at every V — **despite LOWER
silhouette**. Silhouette misleads; downstream F1 + acoustic agreement + next-token
perplexity all favor SSL. (Reinforces the thesis's "silhouette is a poor quality
proxy" point.)

---

## HEADLINE — cross-species dissociation (order matters for marmosets, not bats)
Same pipeline (sub-unit tokens -> BERT real vs within-call SHUFFLED order + bag-LR),
InfantMarmosetsVox. **Preliminary (twin_2 only; full run pending download):**

| species (system) | task | real | shuf | Δ(real−shuf) 95%CI | BERT vs bag |
|---|---|---|---|---|---|
| bat (graded) | context | 0.276 | 0.276 | [−0.004,+0.004] **n.s.** | ≈ (0.276/0.230) |
| bat (graded) | caller | 0.162 | 0.163 | [−0.006,+0.003] **n.s.** | ≈ |
| marmoset (structured) | call-type | 0.578 | 0.564 | **[+0.004,+0.022]** helps | ≫ (0.578/0.404) |
| marmoset (structured) | caller | 0.897 | 0.885 | **[+0.007,+0.017]** helps | ≫ (0.897/0.825) |

→ **Whether token order carries communicative information depends on the vocal
system.** Graded systems (bat) = order irrelevant, multiset is everything (BERT≈bag).
Structured-call species (marmoset; twitter = repeated phrases) = order adds a small
but significant amount, and BERT ≫ bag. This reframes "leverage sequential
structure" (Sarkar) as **species/system-dependent**, established with shuffle
controls + CIs across two species.

**Honest caveats (to resolve):**
1. Marmoset order effect is small (Δ≈0.012–0.013) though significant (CI>0).
2. Preliminary: twin_2 only (caller = 2-class). Full run (10 callers) pending download.
3. **Sequence-length confound:** bat sub-unit sequences are short (~4) vs marmoset (~11).

### CONFOUND CONFIRMED — order effect is length-driven, not species-driven
Length-stratified marmoset call-type (twin_2):

| band (frames) | mean len | Δ(real−shuf) 95%CI |
|---|---|---|
| 2–4 (bat-like) | 3.5 | +0.000 **n.s.** |
| 5–8 | 6.4 | +0.017 **n.s.** |
| 9–48 | 36.9 | +0.050 **ORDER HELPS** |

→ When marmoset sequences are SHORT (bat-like), order is **also n.s.** The apparent
cross-species dissociation is a **length confound**: order only matters once
sequences are long (~37). **Do NOT claim a clean species dissociation.**

→ Real open question: is the bat order-null fundamental (graded system) or just a
consequence of short bat sub-unit sequences?

### Disentangled: order effect is LENGTH-dependent + modest species difference
Bat frame-level (each segment = 21 mel frames; bat frame-seqs are inherently long ≥21):

| species | task | mean len | Δ(real−shuf) 95%CI |
|---|---|---|---|
| bat (frame, long) | context | 78 | +0.014 [−0.001, +0.028] borderline (3 seeds) |
| marmoset (long) | call-type | 37 | +0.050 [+0.035, +0.064] significant |
| marmoset (short, bat-like) | call-type | 3.5 | +0.000 n.s. |
| bat (segment, short) | context | 4 | +0.000 n.s. |

**Honest synthesis (firmed, 5 seeds):**
| species | mean len | Δ(real−shuf) 95%CI |
|---|---|---|
| bat (frame, long) | 78 | +0.008 [−0.005, +0.021] **n.s.** |
| marmoset (long) | 37 | +0.050 [+0.035, +0.064] **sig** |
| both, short (≤4) | 3–4 | ~0 **n.s.** |

1. **Sequence length is a confound the subfield (incl. Sarkar) ignores:** order is n.s.
   at SHORT sequences in BOTH species; the effect appears only at length.
2. **At LONG length the dissociation is genuine, NOT length:** bat sequences are even
   LONGER (78) than marmoset (37) yet show NO order effect (n.s.), while marmosets do
   (+0.050). Length cannot explain the bat null → the difference is the vocal SYSTEM.
3. **Final headline:** token order carries communicative info in marmosets (structured
   calls) but not bats (graded system), even at greater bat sequence length —
   established with shuffle controls, length stratification, cross-individual splits,
   and CIs (controls absent from prior work).

→ Significant + rigorous contribution: (a) "sequential-structure" claims for animal
vocalizations must control for sequence length; (b) once controlled, a genuine
graded-vs-structured vocal-system dissociation remains; (c) constructive SSL tokenizer
(best discrete repertoire); (d) per-context leakage caveat.
NOTE: marmoset side currently twin_2 only — confirming with full 10-caller data.

### Finding 3 — methodological: per-context token-IDs leak the label
Feeding per-context token sequences (disjoint vocab per context) to any sequence
classifier → macro F1 ≈ 1.0 (bag-LR exactly 1.000): the token-ID range reveals the
context. Per-context tokenizers must be evaluated by generative max-likelihood
(as the thesis did), not by token-sequence classifiers.

---

## Contributions
1. First controlled order ablation (shuffle) for tokenized animal vocalizations,
   across tokenizers, sequence-model families, and two tasks, with cross-individual
   splits and CIs — the controls absent from prior work.
2. Behavioral-context decoding from token sequences on a new corpus (bats), beyond
   call-type/caller.
3. Domain-matched SSL tokenizer that surpasses prior generative-density tokenizers.
4. A leakage caveat for per-context tokenization in sequence classifiers.

## Remaining experiments (priority)
- [ ] next-token intrinsic order test (running) — 3rd independent confirmation.
- [ ] SSL vocab sweep V∈{15,30,60,120} — robustness of the SSL-tokenizer advantage.
- [ ] (stretch) SSL + SCAN deep clustering (full SensorSCAN recipe) vs SSL+kmeans.
