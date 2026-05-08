"""Main experiment on per-context segmentation pipeline + UMAP 8D."""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd, joblib, hdbscan
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, HDBSCAN as sk_HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CACHE / 'ablation_state_152k_21x32_pcseg.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_pcseg_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
HP1_CTX = [2,3,4,5,6,7,9,10]
print(f'pcseg state: {len(seg_df)} segments, embedding: {emb.shape}')

# Need global HDBSCAN labels for baseline. Fit once with Assom defaults.
HDB_PATH = CACHE / 'hdb_nca_labels_152k_21x32_pcseg.npy'
if HDB_PATH.exists():
    hdb_nca = np.load(HDB_PATH)
    print(f'Loaded cached HDBSCAN labels: {len(set(hdb_nca.tolist()))} syllables')
else:
    print('Fitting global HDBSCAN (Assom defaults: frac=0.02, ms=20, eps=0.1, leaf)...')
    mcs = max(10, int(0.02 * len(emb)))
    h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                        cluster_selection_epsilon=0.1,
                        cluster_selection_method='leaf',
                        metric='euclidean', core_dist_n_jobs=-1).fit(emb)
    raw = h.labels_
    nn = raw >= 0
    knn = KNeighborsClassifier(n_neighbors=30, n_jobs=-1).fit(emb[nn], raw[nn])
    hdb_nca = raw.copy(); hdb_nca[~nn] = knn.predict(emb[~nn])
    np.save(HDB_PATH, hdb_nca)
    n_cl = len(set(hdb_nca.tolist()))
    print(f'  vocab: {n_cl} syllables, raw noise: {(raw==-1).mean():.1%}')

# Group vocs
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
print(f'Vocs: {len(vocs)}, bats: {len(all_bats)}')


def _gauss(X,mu,S):
    D=X.shape[1]; S=S+1e-6*np.eye(D)
    _,ld=np.linalg.slogdet(S); inv=np.linalg.inv(S)
    diff=X-mu; mahal=np.einsum('ij,jk,ik->i',diff,inv,diff)
    return -0.5*(D*np.log(2*np.pi)+ld+mahal)
def _logmix(comps,X):
    if not comps: return np.full(len(X),-1e10)
    logs=np.stack([np.log(p+1e-12)+_gauss(X,mu,S) for (p,mu,S) in comps],axis=1)
    m=logs.max(axis=1); return m+np.log(np.exp(logs-m[:,None]).sum(axis=1))
def fit_dp(X,s):
    return BayesianGaussianMixture(n_components=15,weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1,covariance_type='full',max_iter=150,random_state=s).fit(X)
def fit_hdb(X,s):
    mcs = max(20, int(0.02*len(X)))
    h = sk_HDBSCAN(min_cluster_size=mcs,min_samples=10,cluster_selection_epsilon=0.05).fit(X)
    comps = []
    for k in sorted(set(h.labels_)):
        if k < 0: continue
        m = h.labels_ == k
        if m.sum() < 5: continue
        Xi = X[m]; mu = Xi.mean(0); Sg = np.cov(Xi.T) if Xi.shape[0]>1 else np.eye(X.shape[1])*0.1
        comps.append((m.sum()/len(X), mu, Sg))
    if not comps:
        comps = [(1.0, X.mean(0), np.cov(X.T) if X.shape[0]>1 else np.eye(X.shape[1])*0.1)]
    return comps
def fit_km(X,s,K=15):
    K = min(K, max(2, len(X)//10))
    km = KMeans(n_clusters=K,random_state=s,n_init=10).fit(X)
    comps = []
    for k in range(K):
        m = km.labels_ == k
        if m.sum() < 2: continue
        Xi = X[m]; mu = Xi.mean(0); Sg = np.cov(Xi.T) if Xi.shape[0]>1 else np.eye(X.shape[1])*0.1
        comps.append((m.sum()/len(X), mu, Sg))
    return comps
def classify(toks,log_prior,test_vocs,score):
    yt,yp=[],[]
    for v in test_vocs:
        X = emb[v['seg_ids']]
        if len(X)==0: continue
        best,bs = None,-np.inf
        for c, t in toks.items():
            ll = score(t,X).sum()+log_prior[c]
            if ll>bs: bs=ll; best=c
        if best is None: continue
        yt.append(v['ctx']); yp.append(best)
    return np.array(yt), np.array(yp)
def seqf(seq,V):
    cnt = Counter(seq); n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k,c in cnt.items():
        if 0<=k<V: bos[k]=c/max(n,1)
    rich = len(cnt)/max(n,1); p = np.array(list(cnt.values()),dtype=np.float32)/max(n,1)
    ent = float(-(p*np.log(p+1e-12)).sum()); rep = max(cnt.values())/max(n,1) if cnt else 0
    return np.concatenate([bos,[n,rich,ent,rep]]).astype(np.float32)


rows = []
for s in range(5):
    print(f'\nseed {s}...', flush=True)
    rng = np.random.default_rng(s); ba = np.array(all_bats); rng.shuffle(ba)
    test_b = set(ba[:11].tolist()); train_b = set(ba[11:41].tolist())
    train_v = [v for v in vocs if v['em'] in train_b]
    test_v  = [v for v in vocs if v['em'] in test_b]
    train_mask = np.zeros(len(emb), dtype=bool)
    for v in train_v: train_mask[v['seg_ids']] = True
    n_tv = len(train_v); log_prior = {}
    for c in HP1_CTX:
        nc = sum(1 for v in train_v if v['ctx']==c)
        log_prior[c] = np.log(max(nc,1)/n_tv)
    tdp,thdb,tkm={},{},{}
    for c in HP1_CTX:
        m = train_mask & (ctx==c)
        if m.sum()<30: continue
        Xc = emb[m]
        tdp[c]=fit_dp(Xc,s); thdb[c]=fit_hdb(Xc,s); tkm[c]=fit_km(Xc,s)
    yt_dp,yp_dp = classify(tdp,log_prior,test_v,lambda t,X: t.score_samples(X))
    yt_hd,yp_hd = classify(thdb,log_prior,test_v,lambda t,X: _logmix(t,X))
    yt_km,yp_km = classify(tkm,log_prior,test_v,lambda t,X: _logmix(t,X))
    V = int(np.max(hdb_nca))+1
    Xtr=[]; ytr=[]
    for v in train_v:
        L = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i]>=0]
        if L: Xtr.append(seqf(L,V)); ytr.append(v['ctx'])
    Xte=[]; yte=[]
    for v in test_v:
        L = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i]>=0]
        if L: Xte.append(seqf(L,V)); yte.append(v['ctx'])
    rf = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=s,n_jobs=-1).fit(Xtr,ytr)
    yp_bl = rf.predict(Xte)
    f = lambda y,p: round(f1_score(y,p,average='weighted',labels=HP1_CTX,zero_division=0),3) if len(y) else 0
    rows.append({'seed':s,
                 'pc_dpgmm':f(yt_dp,yp_dp),
                 'pc_hdb':f(yt_hd,yp_hd),
                 'pc_km':f(yt_km,yp_km),
                 'baseline':f(np.array(yte),yp_bl)})
    print(f'  DP-GMM {rows[-1]["pc_dpgmm"]:.3f}  HDB-tok {rows[-1]["pc_hdb"]:.3f}  k-means {rows[-1]["pc_km"]:.3f}  base {rows[-1]["baseline"]:.3f}')


df = pd.DataFrame(rows)
print('\n=== AGG (UMAP 8D + per-context segmentation) ===')
for col in ['pc_dpgmm','pc_hdb','pc_km','baseline']:
    print(f'  {col:12s} {df[col].mean():.3f} ± {df[col].std():.3f}')
for col in ['pc_dpgmm','pc_hdb','pc_km']:
    g = df[col] - df['baseline']
    m, sd = g.mean(), g.std()
    sig = 'SIG+' if m-2*sd>0 else ('SIG-' if m+2*sd<0 else 'NS')
    print(f'  gain_{col:8s} {m:+.3f} ± {sd:.3f}  CI[{m-2*sd:+.3f},{m+2*sd:+.3f}]  {sig}')
df.to_csv('docs/thesis/figures/percontext_152k_21x32_pcseg_8d_results.csv', index=False)
print('Saved.')
