"""Why is the order effect near-zero for bats? Within-vocalization token diversity.

Token ORDER can only carry information if a vocalization contains DIVERSE tokens in a
meaningful arrangement. If a vocalization is mostly one repeated token (low diversity),
shuffling changes nothing -> no order effect, regardless of species. We quantify
within-vocalization token diversity for bats vs marmosets and relate it to the observed
order effects. A lower bat diversity would mechanistically EXPLAIN the bat order-null.

Metrics per vocalization: length L, #unique tokens U, normalized entropy H/log(U_voc),
repetition rate 1 - U/L. Reported per species/tokenization. CPU-only.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
BAT = Path('/Volumes/T7/cache/assom_paper_repro')
MARM = Path('/Volumes/T7/datasets/InfantMarmosetsVox/cache')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
V_TOK = 30


def voc_stats(seqs):
    L, U, ent, rep = [], [], [], []
    for s in seqs:
        if len(s) < 2:
            continue
        c = Counter(s); n = len(s); u = len(c)
        p = np.array(list(c.values())) / n
        H = -(p * np.log(p)).sum()
        L.append(n); U.append(u); rep.append(1 - u / n)
        ent.append(H / np.log(V_TOK))     # normalized by max possible (log V)
    return dict(n_voc=len(L), mean_len=np.mean(L), mean_unique=np.mean(U),
                mean_norm_entropy=np.mean(ent), mean_repetition=np.mean(rep),
                frac_singletoken=np.mean([u == 1 for u in U]))


def bat_segment_seqs():
    st = joblib.load(BAT / 'ablation_state_152k_21x32.joblib')
    seg = st['seg_df'].reset_index(drop=True)
    emb = np.load(BAT / 'umap_152k_21x32_md1.0_8d.npy')
    tok = KMeans(V_TOK, n_init=10, random_state=0).fit_predict(emb).astype(np.int32)
    df = pd.DataFrame({'file': seg['file_name'].to_numpy(), 'pos': seg['pos_segment'].to_numpy(),
                       'tok': tok, 'ctx': seg['context'].to_numpy(), 'em': seg['emitter'].to_numpy()})
    df = df[(df.em != 0) & (df.ctx.isin(HP1_CTX))]
    return [g.sort_values('pos').tok.tolist() for _, g in df.groupby('file', sort=False)], st, seg


def bat_frame_seqs(st, seg):
    specs = np.asarray(st['tf_specs']).reshape(len(seg), 21, 32).astype(np.float32)
    df = pd.DataFrame({'file': seg['file_name'].to_numpy(), 'pos': seg['pos_segment'].to_numpy(),
                       'ctx': seg['context'].to_numpy(), 'em': seg['emitter'].to_numpy(),
                       'idx': np.arange(len(seg))})
    df = df[(df.em != 0) & (df.ctx.isin(HP1_CTX))]
    vocs = []
    for _, g in df.sort_values('pos').groupby('file', sort=False):
        fr = specs[g.idx.to_numpy()].reshape(-1, 32)[:96]
        if len(fr) >= 2: vocs.append(fr)
    allf = np.concatenate(vocs); mu, sd = allf.mean(0), allf.std(0) + 1e-6
    samp = allf[np.random.default_rng(0).choice(len(allf), min(200000, len(allf)), replace=False)]
    km = KMeans(V_TOK, n_init=10, random_state=0).fit((samp - mu) / sd)
    return [km.predict((fr - mu) / sd).tolist() for fr in vocs]


def marmoset_frame_seqs():
    calls = []
    for f in sorted(MARM.glob('*.npz')):
        if f.name.startswith('._'): continue
        z = np.load(f, allow_pickle=True)
        for fr in z['frames']:
            fr = np.asarray(fr, dtype=np.float32)
            if fr.ndim == 2 and len(fr) >= 2: calls.append(fr)
    allf = np.concatenate(calls); mu, sd = allf.mean(0), allf.std(0) + 1e-6
    km = KMeans(V_TOK, n_init=10, random_state=0).fit((allf - mu) / sd)
    return [km.predict((fr - mu) / sd).tolist() for fr in calls]


if __name__ == '__main__':
    rows = []
    print('bat segment-level...', flush=True)
    bseg, st, seg = bat_segment_seqs()
    rows.append({'species_tok': 'bat_segment', **voc_stats(bseg)})
    print('bat frame-level...', flush=True)
    rows.append({'species_tok': 'bat_frame', **voc_stats(bat_frame_seqs(st, seg))})
    print('marmoset frame-level...', flush=True)
    rows.append({'species_tok': 'marmoset_frame', **voc_stats(marmoset_frame_seqs())})

    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != 'species_tok':
            df[c] = df[c].round(3)
    print('\n=== Within-vocalization token diversity (V=30) ===', flush=True)
    print(df.to_string(index=False), flush=True)
    df.to_csv(OUT / 'token_diversity.csv', index=False)
    print(f'\nSaved: {OUT/"token_diversity.csv"}', flush=True)
    print('\nInterpretation: lower mean_unique / mean_norm_entropy and higher '
          'mean_repetition -> less room for order to matter.', flush=True)
