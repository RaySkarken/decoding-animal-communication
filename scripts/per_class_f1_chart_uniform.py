"""Per-class F1 (5-seed mean) — DP-GMM with UNIFORM prior, vs baseline.

Recommended config in thesis: per-context DP-GMM, full covariance, UMAP-8D,
classification by maximum likelihood with UNIFORM context prior.

Saves bar chart and CSV. Outputs to per_class_f1_chart_uniform.{png,pdf,csv}.
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
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
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

        # UNIFORM prior over contexts (key change vs original chart)
        log_prior = {c: -np.log(len(HP1_CTX)) for c in HP1_CTX}

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
        rows.append({'seed':s, 'method':'DP-GMM (uniform prior)', 'macro_F1':macro_pc, 'weighted_F1':weighted_pc,
                     **{f'F1_{CTX_NAME[c]}':per_cls_pc[c] for c in HP1_CTX}})
        rows.append({'seed':s, 'method':'Baseline RF', 'macro_F1':macro_bl, 'weighted_F1':weighted_bl,
                     **{f'F1_{CTX_NAME[c]}':per_cls_bl[c] for c in HP1_CTX}})
    return pd.DataFrame(rows)


print('Running DP-GMM (full covariance, UMAP-8D) + uniform prior + max-likelihood...')
df = run_pipeline(
    CACHE / 'ablation_state_152k_21x32.joblib',
    CACHE / 'umap_152k_21x32_md1.0_8d.npy',
    CACHE / 'hdb_nca_labels_152k_21x32.npy',
)

df.to_csv('docs/thesis/figures/per_class_f1_uniform.csv', index=False)

print('\n=== Per-class F1 (5-seed mean), uniform prior ===')
print(f'{"context":12s} {"DP-GMM":>9s} {"baseline":>9s} {"Δ":>7s}')
for c in HP1_CTX:
    col = f'F1_{CTX_NAME[c]}'
    pc_mean = df[df.method.str.startswith('DP-GMM')][col].mean()
    bl_mean = df[df.method=='Baseline RF'][col].mean()
    print(f'{CTX_NAME[c]:12s} {pc_mean:>9.3f} {bl_mean:>9.3f} {pc_mean-bl_mean:>+7.3f}')
print(f'{"macro":12s} {df[df.method.str.startswith("DP-GMM")].macro_F1.mean():>9.3f} {df[df.method=="Baseline RF"].macro_F1.mean():>9.3f}')
print(f'{"weighted":12s} {df[df.method.str.startswith("DP-GMM")].weighted_F1.mean():>9.3f} {df[df.method=="Baseline RF"].weighted_F1.mean():>9.3f}')

# Bar chart
fig, ax = plt.subplots(figsize=(11, 5.5))
contexts = HP1_CTX + ['macro_F1', 'weighted_F1']
labels = [CTX_NAME[c] if c in CTX_NAME else ('macro F1' if c=='macro_F1' else 'weighted F1') for c in contexts]


def col_means(df, method, contexts):
    means, stds = [], []
    for c in contexts:
        col = f'F1_{CTX_NAME[c]}' if c in CTX_NAME else c
        sub = df[df.method == method]
        means.append(sub[col].mean()); stds.append(sub[col].std())
    return np.array(means), np.array(stds)


pc_m, pc_s = col_means(df, 'DP-GMM (uniform prior)', contexts)
bl_m, bl_s = col_means(df, 'Baseline RF', contexts)

x = np.arange(len(contexts))
w = 0.38
ax.bar(x - w/2, bl_m, w, yerr=bl_s, label='Baseline (RF на токенах)', color='#bbbbbb', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, pc_m, w, yerr=pc_s, label='Per-context DP-GMM (равн. приор)', color='#5b9bd5', edgecolor='black', linewidth=0.5)
ax.axvline(7.5, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('F1')
ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
ax.set_ylim(0, max(pc_m.max(), bl_m.max()) * 1.20)
ax.grid(axis='y', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('docs/thesis/figures/per_class_f1_uniform.png', dpi=160, bbox_inches='tight')
plt.savefig('docs/thesis/figures/per_class_f1_uniform.pdf', bbox_inches='tight')
print('\nSaved: docs/thesis/figures/per_class_f1_uniform.{png,pdf,csv}')
