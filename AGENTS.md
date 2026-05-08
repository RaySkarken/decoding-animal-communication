# AnimalCommunication – Master's Thesis Project

## Project overview

This project extends the work from the paper **"Associative Syntax and Maximal Repetitions reveal context-dependent complexity in fruit bat communication"** (Luigi Assom, NeurIPS 2025 Workshop on AI for Non-Human Animal Communication). The thesis builds on that paper’s unsupervised pipeline for inferring repertoire and syntax in graded vocal systems (fruit bats as the case study). The **thesis contribution** is **adaptive tokenization with a dynamic vocabulary** and stronger or more interpretable metrics relative to that baseline (see **Thesis goal and current stage** below).

## Key references

- **Paper**: arXiv 2512.01033v1 — Associative Syntax and Maximal Repetitions in fruit bat communication.
- **Baseline analysis**: `baselinePaperAnalysis.md` — structured summary of the paper (objectives, methods, results).
- **Implementation repo**: `decodingNonHumanCommunication/` — cloned from https://github.com/gg4u/decodingNonHumanCommunication (MSc thesis code; unrefactored).
- **Data / download**: `FruitBat-vocalizations-open-data/` — cloned from https://github.com/gg4u/FruitBat-vocalizations-open-data (tutorial for Egyptian fruit bat dataset on AWS S3).

## Dataset

- **Source**: Prat et al. — “An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny” (Sci Data 2017).
- **Access**: FigShare (original) or AWS S3 bucket `s3://egyptian-fruit-bat-vocalizations-open-data/` (see `FruitBat-vocalizations-open-data/README.md` and `Tutorial.md`).
- **Content**: ~300k vocalizations, WAV + CSV metadata (emitter, addressee, context). Contexts include Feeding, Grooming, Kissing, Fighting, Mating protest, Threat-like, Isolation, etc. Ambiguous contexts (Generic, Sleeping, Unknown) are excluded in the paper.

## Paper pipeline (what we want to reproduce / extend)

1. **Repertoire (RQ1)**  
   Raw audio → segmentation (fixed noise floor or dynamic threshold) → mel-spectrograms (or AE latent) → UMAP → HDBSCAN → syllable labels. Evaluation: silhouette, ARI/NMI vs DTW+agglomerative proxy (~27 syllable types per emitter; unsupervised pipeline yields ~7–14).

2. **Syntax & temporal structure (RQ2)**  
   - **HP1 (syntax type)**: Random Forest on 18 sequence features; permutation test (order vs shuffled) → associative syntax (order doesn’t help).  
   - **HP2 (context-dependent usage)**: Wilcoxon on syllable distributions across contexts.  
   - **HP3 (Maximal Repeats)**: Prefix-suffix tree for MRs; exponential vs power-law; MR length and transition networks per context (conflict vs cooperative).

## Repo layout (decodingNonHumanCommunication)

- **Data / DB**: `0.0 - Download-fruitbat-data.ipynb`, `0.2 - Make Json db.ipynb`, `0.3 - *` (Segment Blackboard, Syllables dictionaries, DB -Segment Audio - All), `0.4 - fruit-bat-make-syllable-df.ipynb`, `Thesis-fruit-bat-make-syllable-df.ipynb`.
- **Segmentation (dynamic threshold)**: `fruit_bat_segment-all-2.0.ipynb`, `fruit_bat_isolation-segmentation.ipynb` (use `vocalseg`).
- **Clustering / repertoire**: `UMAP_comparisons.ipynb`, `Experiments with UMAP.ipynb`, `TF_AE.ipynb`, `Exp1 - AE.ipynb`.
- **Syntax / classifier**: `Exp1 - Classifier.ipynb` (RF, 18 features, permutation test).
- **Maximal Repeats / networks**: `Probability_Suffix_Trees.ipynb` (suffix_tree, transition graphs).
- **Single-run baseline**: `notebooks/baseline_pipeline.ipynb` — one notebook that runs data load → mel-specs → UMAP → HDBSCAN → sequence features → RF and permutation test (and optional MRs); use a subset (`MAX_SEGMENTS`) if data is large.
- **Outputs**: `classifiers_bat_215/`, `classifiers_bat_231/` (repertoire and related artifacts; symbolic_sequences, graph, map_complete are produced in `0.3 - Syllables dictionaries.ipynb` and used by Classifier and Probability_Suffix_Trees).

## Setup (virtual environment only)

- **Always use a virtual environment** — never install into system Python. See **`docs/SETUP.md`** for venv + pip or conda steps. Dependencies are in `requirements.txt` and `environment.yml`; `.venv/` is gitignored.

## Paths and dependencies

- Many notebooks use **hardcoded paths** (e.g. `/data0/home/h21/luas6629/Thesis/`). To run locally, replace with project root or use a config (e.g. `DATA_DIR`, `PROJECT_DIR`).
- **External packages**: `avgn` (data paths, download, utils), `vocalseg` (dynamic threshold segmentation), `umap-learn`, `hdbscan`, `tensorflow` / `keras` (Assom-style mel), **`torch` + `torchaudio`** (BEATs / NatureBEATs), `pandas`, `scikit-learn`, `suffix_tree`, `networkx`, `boto3` (if using S3), `safetensors` (NatureLM merge). The download notebook uses `avgn.downloading.download` and FigShare; for S3-only workflow see `FruitBat-vocalizations-open-data/Tutorial.md`.

## Thesis goal and current stage

- **Primary goal**: **Adaptive tokenization with a dynamic vocabulary** (repertoire that can grow/shrink and refine via split / merge / add / prune, plus sequence-aware steps such as BPE-style merges), evaluated against Assom’s fixed-pipeline baseline. Success is framed as **better or more interpretable clustering and downstream metrics** (silhouette, ARI/NMI vs context, noise fraction, HP1 context classification, maximal repeats, network statistics) — not a single number, but consistent improvement where the thesis claims it.
- **Baseline to beat / align with**: Assom’s pipeline (mel features → UMAP → HDBSCAN → optional NCA noise reassignment → 18 sequence features → RF + permutation test; MRs / networks for RQ2).
- **Where we are now**:
  - Core logic lives in **`src/`** (importable from notebooks).
  - **`notebooks/adaptive_tokenization.ipynb`** — main adaptive tokenizer demo vs **UMAP + HDBSCAN + NCA** baseline (same hyperparameters as Assom-style baseline section).
  - **`notebooks/beats_experiment.ipynb`** — compares **Mel-672D**, **Microsoft BEATs**, and **NatureLM-audio–merged “NatureBEATs”** embeddings under the same baseline + adaptive evaluation.
  - Assom-close mel front-end is **TensorFlow `LogMelSpectrogram` + `preprocess_model`** (from `decodingNonHumanCommunication/TF_AE.ipynb`), not `librosa.feature.melspectrogram`, in the reproduction notebooks listed below.

## Reproducing the Assom baseline (file map)

| Role | Location | Notes |
|------|----------|--------|
| Paper summary | `baselinePaperAnalysis.md` | RQ1/RQ2, metrics, narrative. |
| Thesis design brief | `CONTEXT.md` | Adaptive tokenization idea, related work, evaluation checklist. |
| Env / deps | `docs/SETUP.md`, `requirements.txt`, `environment.yml` | Use **`.venv/`** only. |
| **Mel + TF preprocess (Assom-style)** | `src/features.py` | `LogMelSpectrogram`, `build_preprocess_model`, `compute_spectrograms`. |
| **Data load, segmentation, filters** | `src/data.py` | Annotations, zips, `dynamic_segmentation` (vocalseg), `iqr_filter`, `resample_audio`. |
| **Adaptive tokenizer** | `src/tokenizer.py` | `AdaptiveTokenizer` (HDBSCAN seed, refine, BPE), optional `iterative_tokenize`. |
| **Sequences, 18 features, HP1 RF** | `src/sequence.py` | `build_sequences`, `compute_sequence_features`, `classify_context`. |
| **Metrics (silhouette, ARI/NMI, MR, nets, HP2, …)** | `src/eval.py` | `cluster_quality`, `context_alignment`, `full_evaluation`, etc. |
| **Microsoft BEATs (code)** | `src/beats/` | Vendored from [unilm/beats](https://github.com/microsoft/unilm/tree/master/beats); relative imports patched. |
| **BEATs + NatureLM merge** | `src/features.py` | `ensure_naturelm_beats_merged`, `compute_beats_embeddings(encoder=...)`. |
| **Assom reproduction (notebook)** | `notebooks/assom_exact_reproduction.ipynb` | TF mel, UMAP, HDBSCAN, diagnostics. |
| **Single-run baseline (subset-friendly)** | `notebooks/baseline_pipeline.ipynb` | End-to-end baseline with `MAX_SEGMENTS` cap. |
| **Adaptive vs baseline (UMAP+HDBSCAN+NCA)** | `notebooks/adaptive_tokenization.ipynb` | Baseline section uses fractional `min_cluster_size` (2% of N), `cluster_selection_epsilon`, and NCA+KNN noise reassignment; adaptive uses `AdaptiveTokenizer` on the same mel features. |
| **BEATs / NatureBEATs comparison** | `notebooks/beats_experiment.ipynb` | Same baseline + adaptive table for multiple embeddings. |
| **Original Assom / Luigi code (reference)** | `decodingNonHumanCommunication/` | Especially `TF_AE.ipynb` (mel layer), `Exp1 - Classifier.ipynb`, `Probability_Suffix_Trees.ipynb`; paths often hardcoded — use `src/` + configurable `DATA_DIR` instead. |

**Checkpoints (not in git; large files):**

- Microsoft BEATs AS2M: e.g. `BEATs_iter3_plus_AS2M.pt` (user typically stores under **`/Volumes/T7/models/beats/`**).
- NatureLM-audio: Hugging Face `EarthSpeciesProject/NatureLM-audio` → e.g. **`/Volumes/T7/models/NatureLM-audio/`**; first use builds `beats_encoder_merged.pt` (see `ensure_naturelm_beats_merged` in `src/features.py`).

## For AI assistants

- Prefer reading **`baselinePaperAnalysis.md`**, **`CONTEXT.md`**, and this file for paper vs thesis vs implementation context.
- When modifying the pipeline or adding experiments, keep **paths configurable** (`DATA_DIR`, checkpoint paths, `NATURELM_DIR`) and extend this section or add **`docs/pipeline.md`** if the flow grows beyond what fits here.
