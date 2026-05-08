"""Per-class F1 (5-seed mean) + macro F1 for two pipelines vs baseline.

Saves bar chart and CSV.
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2,3,4,5,6,7,9,10]
CTX_NAME = {2:'Biting', 3:'Feeding', 4:'Fighting', 5:'Grooming',
            6:'Isolation', 7:'Kissing', 9:'Mating', 10:'Threat'}

def run_pipeline(state_path, umap_path, hdb_nca_path):
    st = joblib.load(state_path)
    seg_df = st['seg_df'].reset_index(drop=True)
    emb = np.load(umap_path)
    hdb_nca = np.load(hdb_nca_path)
    ctx = seg_df['context'].to_numpy()
    em_arr = seg_df['emitter'].to_numpy()
    vocs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        if dom_em_signed == 0: continue
        dom_em_abs = abs(dom_em_signed)
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        vocs.append({'seg_ids':seg_ids,'ctx':dom_ctx,'em':dom_em_abs})
    all_bats = sorted(set(v['em'] for v in vocs))

    def seqf(seq, V):
        cnt = Counter(seq); n = len(seq)
        bos = np.zeros(V, dtype=np.float32)
        for k, c in cnt.items():
            if 0<=k<V: bos[k]=c/max(n,1)
        rich = len(cnt)/max(n,1)
        p = np.array(list(cnt.values()), dtype=np.float32)/max(n,1)
        ent = float(-(p*np.log(p+1e-12)).sum())
        rep = max(cnt.values())/max(n,1) if cnt else 0
        return np.concatenate([bos,[n,rich,ent,rep]]).astype(np.float32)

    rows = []
    for s in range(5):
        rng = np.random.default_rng(s); ba = np.array(all_bats); rng.shuffle(ba)
        test_b = set(ba[:11].tolist()); train_b = set(ba[11:41].tolist())
        train_v = [v for v in vocs if v['em'] in train_b]
        test_v  = [v for v in vocs if v['em'] in test_b]
        train_mask = np.zeros(len(emb), dtype=bool)
        for v in train_v: train_mask[v['seg_ids']] = True
        n_tv = len(train_v)
        log_prior = {c: np.log(max(sum(1 for v in train_v if v['ctx']==c),1)/n_tv) for c in HP1_CTX}

        toks = {}
        for c in HP1_CTX:
            m = train_mask & (ctx==c)
            if m.sum()<30: continue
            toks[c] = BayesianGaussianMixture(n_components=15,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.1, covariance_type='full',
                max_iter=150, random_state=s).fit(emb[m])

        yt_pc, yp_pc = [], []
        for v in test_v:
            X = emb[v['seg_ids']]
            if len(X)==0: continue
            best, bs = None, -np.inf
            for c, t in toks.items():
                ll = t.score_samples(X).sum() + log_prior[c]
                if ll>bs: bs=ll; best=c
            if best is None: continue
            yt_pc.append(v['ctx']); yp_pc.append(best)

        V = int(np.max(hdb_nca))+1
        Xtr=[]; ytr=[]
        for v in train_v:
            L = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i]>=0]
            if L: Xtr.append(seqf(L,V)); ytr.append(v['ctx'])
        Xte=[]; yte=[]
        for v in test_v:
            L = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i]>=0]
            if L: Xte.append(seqf(L,V)); yte.append(v['ctx'])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=s, n_jobs=-1).fit(Xtr,ytr)
        yp_bl = rf.predict(Xte)

        # per-class F1 + macro
        yt_pc_a = np.array(yt_pc); yp_pc_a = np.array(yp_pc)
        yte_a = np.array(yte); yp_bl_a = np.array(yp_bl)
        per_cls_pc = {}; per_cls_bl = {}
        for c in HP1_CTX:
            mask_pc = yt_pc_a == c
            mask_bl = yte_a == c
            per_cls_pc[c] = f1_score(yt_pc_a==c, yp_pc_a==c, zero_division=0) if mask_pc.sum() else 0
            per_cls_bl[c] = f1_score(yte_a==c, yp_bl_a==c, zero_division=0) if mask_bl.sum() else 0
        macro_pc = np.mean([per_cls_pc[c] for c in HP1_CTX])
        macro_bl = np.mean([per_cls_bl[c] for c in HP1_CTX])
        weighted_pc = f1_score(yt_pc_a, yp_pc_a, average='weighted', labels=HP1_CTX, zero_division=0)
        weighted_bl = f1_score(yte_a, yp_bl_a, average='weighted', labels=HP1_CTX, zero_division=0)
        rows.append({'seed':s, 'method':'DP-GMM', 'macro_F1':macro_pc, 'weighted_F1':weighted_pc,
                     **{f'F1_{CTX_NAME[c]}':per_cls_pc[c] for c in HP1_CTX}})
        rows.append({'seed':s, 'method':'Baseline', 'macro_F1':macro_bl, 'weighted_F1':weighted_bl,
                     **{f'F1_{CTX_NAME[c]}':per_cls_bl[c] for c in HP1_CTX}})
    return pd.DataFrame(rows)


print('Pipeline A: Paper-faithful + standard seg + 8D UMAP')
dfA = run_pipeline(
    CACHE / 'ablation_state_152k_21x32.joblib',
    CACHE / 'umap_152k_21x32_md1.0_8d.npy',
    CACHE / 'hdb_nca_labels_152k_21x32.npy',
)

print('Pipeline B: Paper-faithful + per-context seg + 8D UMAP')
dfB = run_pipeline(
    CACHE / 'ablation_state_152k_21x32_pcseg.joblib',
    CACHE / 'umap_152k_21x32_pcseg_md1.0_8d.npy',
    CACHE / 'hdb_nca_labels_152k_21x32_pcseg.npy',
)

# Aggregate (5-seed mean ± std)
def agg(df, name):
    a = df.groupby('method').agg(['mean', 'std']).reset_index()
    a.columns = ['_'.join(c).strip('_') for c in a.columns]
    a['pipeline'] = name
    return a

aggA = agg(dfA, 'standard_seg')
aggB = agg(dfB, 'pc_seg')

# Print formatted summary
print('\n=== Pipeline A: STANDARD seg + 8D UMAP (5 seeds) ===')
print(f'{"context":12s} {"DP-GMM mean":>11s} {"baseline mean":>13s} {"Δ":>7s}')
for c in HP1_CTX:
    col = f'F1_{CTX_NAME[c]}'
    pc_mean = dfA[(dfA.method == 'DP-GMM')][col].mean()
    bl_mean = dfA[(dfA.method == 'Baseline')][col].mean()
    print(f'{CTX_NAME[c]:12s} {pc_mean:>11.3f} {bl_mean:>13.3f} {pc_mean-bl_mean:>+7.3f}')
print(f'{"macro F1":12s} {dfA[dfA.method=="DP-GMM"].macro_F1.mean():>11.3f} {dfA[dfA.method=="Baseline"].macro_F1.mean():>13.3f}')
print(f'{"weighted F1":12s} {dfA[dfA.method=="DP-GMM"].weighted_F1.mean():>11.3f} {dfA[dfA.method=="Baseline"].weighted_F1.mean():>13.3f}')

print('\n=== Pipeline B: PER-CONTEXT seg + 8D UMAP (5 seeds) ===')
print(f'{"context":12s} {"DP-GMM mean":>11s} {"baseline mean":>13s} {"Δ":>7s}')
for c in HP1_CTX:
    col = f'F1_{CTX_NAME[c]}'
    pc_mean = dfB[(dfB.method == 'DP-GMM')][col].mean()
    bl_mean = dfB[(dfB.method == 'Baseline')][col].mean()
    print(f'{CTX_NAME[c]:12s} {pc_mean:>11.3f} {bl_mean:>13.3f} {pc_mean-bl_mean:>+7.3f}')
print(f'{"macro F1":12s} {dfB[dfB.method=="DP-GMM"].macro_F1.mean():>11.3f} {dfB[dfB.method=="Baseline"].macro_F1.mean():>13.3f}')
print(f'{"weighted F1":12s} {dfB[dfB.method=="DP-GMM"].weighted_F1.mean():>11.3f} {dfB[dfB.method=="Baseline"].weighted_F1.mean():>13.3f}')

# Save CSV
dfA.to_csv('docs/thesis/figures/per_class_f1_standard_seg.csv', index=False)
dfB.to_csv('docs/thesis/figures/per_class_f1_pc_seg.csv', index=False)

# Build comparison chart
fig, ax = plt.subplots(figsize=(13, 6))
contexts = HP1_CTX + ['macro_F1', 'weighted_F1']
labels = [CTX_NAME[c] if c in CTX_NAME else c for c in contexts]

# Aggregate per-class means
def col_means(df, method, contexts):
    means, stds = [], []
    for c in contexts:
        col = f'F1_{CTX_NAME[c]}' if c in CTX_NAME else c
        sub = df[df.method == method]
        means.append(sub[col].mean()); stds.append(sub[col].std())
    return np.array(means), np.array(stds)

A_pc_m, A_pc_s = col_means(dfA, 'DP-GMM', contexts)
A_bl_m, A_bl_s = col_means(dfA, 'Baseline', contexts)
B_pc_m, B_pc_s = col_means(dfB, 'DP-GMM', contexts)
B_bl_m, B_bl_s = col_means(dfB, 'Baseline', contexts)

x = np.arange(len(contexts))
w = 0.21

ax.bar(x - 1.5*w, A_bl_m, w, yerr=A_bl_s, label='Baseline RF (std seg)', color='#cccccc', edgecolor='black', linewidth=0.5)
ax.bar(x - 0.5*w, A_pc_m, w, yerr=A_pc_s, label='DP-GMM (std seg, +0.107 SIG+)', color='#5b9bd5', edgecolor='black', linewidth=0.5)
ax.bar(x + 0.5*w, B_bl_m, w, yerr=B_bl_s, label='Baseline RF (per-context seg)', color='#9d9d9d', edgecolor='black', linewidth=0.5)
ax.bar(x + 1.5*w, B_pc_m, w, yerr=B_pc_s, label='DP-GMM (per-context seg, +0.156 SIG+)', color='#ed7d31', edgecolor='black', linewidth=0.5)

ax.axvline(7.5, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('F1 score')
ax.set_title('Per-class F1, macro F1, weighted F1 — paper-faithful 153k corpus, 8D UMAP, 5 seeds (mean ± SD)')
ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
ax.set_ylim(0, max(B_pc_m) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('docs/thesis/figures/per_class_f1_chart.png', dpi=160, bbox_inches='tight')
plt.savefig('docs/thesis/figures/per_class_f1_chart.pdf', bbox_inches='tight')
print('\nSaved: docs/thesis/figures/per_class_f1_chart.{png,pdf}')
