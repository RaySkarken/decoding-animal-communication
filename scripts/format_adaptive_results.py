"""
Format the final `results_df` from `adaptive_tokenizer_experiment.ipynb`
into a thesis-ready Markdown table for §3.4.

Usage:
    python scripts/format_adaptive_results.py

Reads: /Volumes/T7/cache/assom_paper_repro/adaptive_tokenizer_results.joblib
Outputs: stdout (paste into docs/thesis/04_chapter3_experiments.md §3.4.2)
         + a decision verdict for which method is best
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable so joblib can unpickle Token / TokenizerState
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd


CHECKPOINT = Path("/Volumes/T7/cache/assom_paper_repro/adaptive_tokenizer_results.joblib")


def _fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{digits}f}"


def _fmt_pair(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and (mean != mean)):
        return "—"
    if std is None or (isinstance(std, float) and (std != std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_thesis_table(df: pd.DataFrame) -> str:
    """Build a markdown table suitable for thesis §3.4.2."""
    rows: list[str] = []
    header = (
        "| method | vocab | atomic | comp | silh | noise | "
        "types/em | ARI proxy | NMI proxy | F1 bos (orig/perm/Δ) | "
        "F1 inv (orig/perm/Δ) |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    rows.append(header)
    rows.append(sep)
    for _, r in df.iterrows():
        row = "| " + " | ".join([
            str(r.get("method", "—")),
            str(int(r.get("vocab_size", 0))),
            str(int(r.get("atomic_vocab_size", 0))),
            str(int(r.get("composite_vocab_size", 0))),
            _fmt(r.get("silhouette"), 3),
            _fmt(r.get("noise_fraction"), 3),
            _fmt_pair(r.get("types_per_emitter"), r.get("types_per_emitter_std"), 1),
            _fmt_pair(r.get("ari_proxy"), r.get("ari_proxy_std"), 3),
            _fmt_pair(r.get("nmi_proxy"), r.get("nmi_proxy_std"), 3),
            (f"{_fmt(r.get('hp1_f1_original_bos'), 3)} / "
             f"{_fmt(r.get('hp1_f1_permuted_bos'), 3)} / "
             f"{_fmt(r.get('hp1_f1_delta_bos'), 3)}"),
            (f"{_fmt(r.get('hp1_f1_original_inv'), 3)} / "
             f"{_fmt(r.get('hp1_f1_permuted_inv'), 3)} / "
             f"{_fmt(r.get('hp1_f1_delta_inv'), 3)}"),
        ]) + " |"
        rows.append(row)
    return "\n".join(rows)


def verdict(df: pd.DataFrame) -> str:
    """Simple rule-based verdict on which method is 'best'.

    Scoring weights (subjective):
      - +2 points for highest silhouette
      - +2 points for highest ARI vs proxy
      - +1 point for types/emitter closest to paper's 27 target
      - +1 point for HP1 F1 at least 95% of max (so not degenerate)
      - +1 point for reasonable vocab (≤ 50 composites)
    """
    if df.empty:
        return "No data."
    lines: list[str] = []
    df = df.copy()
    df["_score"] = 0.0

    # silhouette (higher better, non-nan)
    if "silhouette" in df.columns and df["silhouette"].notna().any():
        top = df["silhouette"].idxmax()
        df.loc[top, "_score"] += 2.0
        lines.append(f"- Best silhouette: **{df.loc[top, 'method']}** "
                      f"({df.loc[top, 'silhouette']:.3f})")

    if "ari_proxy" in df.columns and df["ari_proxy"].notna().any():
        top = df["ari_proxy"].idxmax()
        df.loc[top, "_score"] += 2.0
        lines.append(f"- Best ARI vs proxy: **{df.loc[top, 'method']}** "
                      f"({df.loc[top, 'ari_proxy']:.3f})")

    # Note: `types_per_emitter` is the qt_ward proxy which does NOT depend on
    # the tokenizer, so it cannot discriminate methods. Omit from scoring.

    if "hp1_f1_original_inv" in df.columns and df["hp1_f1_original_inv"].notna().any():
        top = df["hp1_f1_original_inv"].idxmax()
        df.loc[top, "_score"] += 2.0   # HP1 F1 inv is our key downstream metric
        lines.append(f"- Best HP1 F1 (inv features): **{df.loc[top, 'method']}** "
                      f"({df.loc[top, 'hp1_f1_original_inv']:.3f})")

    # HP1 not degenerate
    if "hp1_f1_original_bos" in df.columns and df["hp1_f1_original_bos"].notna().any():
        max_f1 = df["hp1_f1_original_bos"].max()
        cutoff = max_f1 * 0.95
        qualify = df[df["hp1_f1_original_bos"] >= cutoff]
        for i in qualify.index:
            df.loc[i, "_score"] += 1.0
        names = ", ".join([str(x) for x in qualify["method"]])
        lines.append(f"- HP1 F1 bos ≥ 95% of max ({max_f1:.3f}): {names}")

    # Reasonable vocab (not pathologically big)
    if "composite_vocab_size" in df.columns and df["composite_vocab_size"].notna().any():
        reasonable = df[df["composite_vocab_size"] <= 50]
        for i in reasonable.index:
            df.loc[i, "_score"] += 1.0
        names = ", ".join([str(x) for x in reasonable["method"]])
        lines.append(f"- Vocab size reasonable (≤ 50 composites): {names}")

    top_overall = df["_score"].idxmax()
    lines.append("")
    lines.append(f"**Overall winner by weighted score: `{df.loc[top_overall, 'method']}`** "
                   f"(score = {df.loc[top_overall, '_score']:.1f})")
    lines.append("")
    lines.append("Score per method:")
    for _, r in df.sort_values("_score", ascending=False).iterrows():
        lines.append(f"- {r['method']}: {r['_score']:.1f}")
    return "\n".join(lines)


def main():
    if not CHECKPOINT.exists():
        print(f"ERROR: {CHECKPOINT} does not exist. Run the notebook first.",
              file=sys.stderr)
        sys.exit(1)
    data = joblib.load(CHECKPOINT)
    df = data["results_df"]
    print("### §3.4.2 Table — head-to-head")
    print()
    print(build_thesis_table(df))
    print()
    print("### Verdict (rule-based, revisit in thesis prose)")
    print()
    print(verdict(df))


if __name__ == "__main__":
    main()
