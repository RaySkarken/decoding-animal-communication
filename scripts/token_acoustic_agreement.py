"""Label-free validation: do SSL tokens reflect acoustic structure better than
mel-UMAP tokens? Agreement of each tokenizer with the independent DTW+agglomerative
acoustic reference (seg_df.proxy_label), via ARI / AMI / NMI. No context labels used.

If SSL tokens agree more with the acoustic proxy, that is evidence — independent of
the downstream context task — that the SSL tokenizer better captures the acoustic
repertoire, corroborating its higher context macro F1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
VS = [30, 120]

print('Loading state + SSL encoder...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
proxy = seg_df['proxy_label'].to_numpy()
from ssl_token_order import Encoder, embed_all
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
mu, sd = mel.mean(0), mel.std(0) + 1e-6
enc = Encoder().to(DEVICE); enc.load_state_dict(torch.load(CACHE / 'ssl_encoder_token.pt', map_location=DEVICE))
Z = embed_all(enc, (mel - mu) / sd)

valid = proxy >= 0
print(f'segments with acoustic proxy: {valid.sum():,} / {len(proxy):,}', flush=True)

sources = {'mel_umap': emb, 'ssl': Z}
rows = []
for name, X in sources.items():
    for V in VS:
        tok = KMeans(V, n_init=10, random_state=0).fit_predict(X).astype(np.int32)
        m = valid
        ari = adjusted_rand_score(proxy[m], tok[m])
        ami = adjusted_mutual_info_score(proxy[m], tok[m])
        nmi = normalized_mutual_info_score(proxy[m], tok[m])
        # silhouette of the tokenization in its own embedding space (sample)
        ridx = np.random.default_rng(0).choice(len(X), 20000, replace=False)
        sil = silhouette_score(X[ridx], tok[ridx])
        print(f'{name:8s} V={V:3d}: ARI(proxy)={ari:.3f} AMI={ami:.3f} NMI={nmi:.3f} silhouette={sil:.3f}', flush=True)
        rows.append({'source': name, 'V': V, 'ARI_proxy': round(ari, 4),
                     'AMI_proxy': round(ami, 4), 'NMI_proxy': round(nmi, 4), 'silhouette': round(sil, 4)})
pd.DataFrame(rows).to_csv(OUT / 'token_acoustic_agreement.csv', index=False)
print(f'\nSaved: {OUT/"token_acoustic_agreement.csv"}', flush=True)
print('\n(Higher ARI/AMI/NMI with the acoustic proxy = tokenizer better reflects '
      'independent acoustic structure, no context labels involved.)', flush=True)
