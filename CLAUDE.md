# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required reading before doing work

This repo has substantial existing briefing material. Read before making non-trivial changes:

- **`AGENTS.md`** — canonical repo map: paper pipeline, file-to-role table, dataset, thesis goal, checkpoints. **Treat as authoritative; don't duplicate its content here.**
- **`CONTEXT.md`** — thesis design brief (adaptive tokenization idea, evaluation checklist). Written in Russian.
- **`baselinePaperAnalysis.md`** — structured summary of the Assom paper (RQ1/RQ2, metrics).
- **`docs/pipeline.md`** — how the original `decodingNonHumanCommunication/` notebooks map to the paper; run order and hardcoded paths to patch.
- **`docs/SETUP.md`** — venv/conda setup, FigShare download script.

## Environment

- **Always work inside `.venv/` or the `animal-comm` conda env.** Never `pip install` into system Python. `.venv/` is gitignored.
- Python 3.11 (conda) or 3.13 (venv — `tensorflow-metal` wheel may be missing). Core stack: `tensorflow>=2.13`, `torch`+`torchaudio`, `umap-learn`, `hdbscan`, `librosa`, `scikit-learn`, `suffix_tree`, `networkx`, `safetensors`, plus GitHub-only `avgn` and `vocalseg`.
- Activate before running anything: `source .venv/bin/activate` (prompt should show `(.venv)` or `(animal-comm)`).
- If install was interrupted: `bash scripts/finish_install.sh` installs TF + GitHub packages.

## Common commands

```bash
# Dataset: full ~60 GB FigShare download, or first zip only (~3 GB) for testing
python scripts/download_figshare_fruitbat.py
python scripts/download_figshare_fruitbat.py --max-zips 1

# Unzip downloaded archives into the layout notebooks expect
python scripts/unzip_figshare_fruitbat.py

# Notebooks
jupyter notebook   # run from project root with venv active
```

There is no build step, no lint config, and no test suite. Validation is done by re-running the relevant notebook end-to-end (use `MAX_SEGMENTS` in `notebooks/baseline_pipeline.ipynb` for a fast sanity check).

## Architecture: `src/` vs notebooks vs `decodingNonHumanCommunication/`

The repo has three parallel implementations of the same pipeline; know which one to touch:

1. **`src/`** — the thesis codebase. All new logic goes here; notebooks `import` from it.
   - `data.py` — annotation/zip loading, `dynamic_segmentation` (wraps `vocalseg`), `iqr_filter`, `resample_audio`.
   - `features.py` — **Assom-style TF mel front-end** (`LogMelSpectrogram` Keras layer + `preprocess_model`), plus `compute_beats_embeddings` and `ensure_naturelm_beats_merged` which merges NatureLM-audio's BEATs encoder from `safetensors` on first use.
   - `tokenizer.py` — `AdaptiveTokenizer` (thesis contribution). HDBSCAN seed → split/merge/add/prune → sequence-aware BPE merges → iterative refinement. Operates on **full-dimensional embeddings, not 2-D UMAP**.
   - `sequence.py` — `build_sequences`, 18-feature vector, RF classifier + permutation test (HP1).
   - `eval.py` — silhouette, ARI/NMI vs context, MR stats, HP2 Wilcoxon, network metrics. Single entry point: `full_evaluation`.
   - `proxy_labels.py` — DTW+agglomerative proxy labels for ARI/NMI comparison.
   - `beats/` — vendored from `microsoft/unilm/tree/master/beats` with relative imports patched; do not refactor into a package.

2. **`notebooks/`** — experiments that drive `src/`. The active ones:
   - `adaptive_tokenization.ipynb` — main thesis demo. Baseline = UMAP+HDBSCAN+NCA with fractional `min_cluster_size` (2% of N) and `cluster_selection_epsilon`, on the same mel features; adaptive = `AdaptiveTokenizer`.
   - `beats_experiment.ipynb` — same eval across Mel-672D, Microsoft BEATs, and NatureBEATs embeddings.
   - `assom_exact_reproduction.ipynb` — reproduces the paper with TF mel (not librosa).
   - `baseline_pipeline.ipynb` — single-notebook end-to-end baseline with `MAX_SEGMENTS` cap.

3. **`decodingNonHumanCommunication/`** — Luigi Assom's original thesis code, cloned unmodified. **Reference only.** Paths are hardcoded (`/data0/home/h21/luas6629/Thesis/`), the repertoire file is saved as `_reportoire.pkl` (typo) but loaded as `_repertoire.pkl` elsewhere. Don't fix bugs here — mirror the behavior in `src/` with configurable paths.

## Things to watch for

- **Mel front-end must be TF, not librosa.** Reproduction and adaptive notebooks use `LogMelSpectrogram` + `build_preprocess_model` from `src/features.py` (ported from `decodingNonHumanCommunication/TF_AE.ipynb`). Switching to `librosa.feature.melspectrogram` silently breaks comparability with the baseline.
- **Keep paths configurable.** Use `DATA_DIR`, `PROJECT_DIR`, `NATURELM_DIR`, checkpoint paths — don't hardcode. User's checkpoints typically live at `/Volumes/T7/models/beats/` and `/Volumes/T7/models/NatureLM-audio/` (external SSD, not in git).
- **NatureBEATs merge is lazy and cached.** `ensure_naturelm_beats_merged` writes `beats_encoder_merged.pt` into the NatureLM dir on first use; subsequent calls reuse it.
- **Excluded contexts.** The paper drops ambiguous contexts (`Generic`, `Sleeping`, `Unknown`). Keep this filter when replicating HP1/HP2.
- **Scope of success.** Thesis is evaluated as **consistent improvement across silhouette, ARI/NMI, noise fraction, HP1 F1, MR, network stats** — not a single headline metric. When reporting results, show the full evaluation table, not one number.
