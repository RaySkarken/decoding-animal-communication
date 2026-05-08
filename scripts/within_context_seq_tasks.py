"""Three within-context sequence tasks:
   T1: Next-token prediction (bigram, already done; here as reference)
   T2: Masked-token prediction (BERT-style cloze, bidirectional bigram)
   T3: Sequence completion: given prefix, predict tail; BLEU-2 + edit distance + NLL

For each context: per-context vocab vs global vocab. 5 random splits 80/20.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter, defaultdict
from itertools import pairwise
from sklearn.mixture import BayesianGaussianMixture

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
HP1_CTX = [2,3,4,5,6,7,9,10]
CTX = {2:'Biting',3:'Feeding',4:'Fighting',5:'Grooming',6:'Isolation',7:'Kissing',9:'Mating',10:'Threat'}

# Per-context labels
labels_pc = np.full(len(seg_df), -1, dtype=np.int32)
offset = 0
for c in HP1_CTX:
    mc = ctx == c
    if mc.sum() < 30: continue
    bgm = BayesianGaussianMixture(n_components=15,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=200, random_state=0).fit(emb[mc])
    labels_pc[mc] = bgm.predict(emb[mc]) + offset
    offset += bgm.n_components

def build_seqs(label_arr):
    seqs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        if dom_em == 0: continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        seq = [int(label_arr[i]) for i in seg_ids if label_arr[i] >= 0]
        if len(seq) < 4: continue   # need ≥4 for masking + completion
        seqs.append({'file': fname, 'context': dom_ctx, 'seq': seq})
    return pd.DataFrame(seqs)

seqs_global = build_seqs(hdb_nca)
seqs_pc = build_seqs(labels_pc)


def fit_bigram(train_seqs, vocab_set, alpha=1.0):
    V = len(vocab_set)
    vlist = sorted(vocab_set); v_to_i = {v:i for i,v in enumerate(vlist)}
    N = np.zeros((V, V)) + alpha
    N0 = np.zeros(V) + alpha
    Nback = np.zeros((V, V)) + alpha   # P(t_i | t_{i+1}) — backward bigram for cloze
    for seq in train_seqs:
        if not seq: continue
        if seq[0] in v_to_i: N0[v_to_i[seq[0]]] += 1
        for a, b in pairwise(seq):
            if a in v_to_i and b in v_to_i:
                N[v_to_i[a], v_to_i[b]] += 1
                Nback[v_to_i[b], v_to_i[a]] += 1
    P0 = N0 / N0.sum()
    P_fwd = N / N.sum(axis=1, keepdims=True)
    P_bwd = Nback / Nback.sum(axis=1, keepdims=True)
    P_uni = N0 / N0.sum()
    return {'V':V, 'vlist':vlist, 'v_to_i':v_to_i, 'P0':P0, 'P_fwd':P_fwd, 'P_bwd':P_bwd, 'P_uni':P_uni}


def task1_nexttok_eval(model, test_seqs):
    total_bits, total_tokens = 0.0, 0
    correct = 0
    v_to_i = model['v_to_i']; P0 = model['P0']; P = model['P_fwd']
    for seq in test_seqs:
        if seq[0] in v_to_i:
            total_bits += -np.log2(max(P0[v_to_i[seq[0]]], 1e-12))
            pred0 = int(np.argmax(P0))
            if pred0 == v_to_i.get(seq[0], -1): correct += 1
        else:
            total_bits += np.log2(model['V'])
        total_tokens += 1
        for a, b in pairwise(seq):
            if a in v_to_i and b in v_to_i:
                pa = P[v_to_i[a]]
                total_bits += -np.log2(max(pa[v_to_i[b]], 1e-12))
                if int(np.argmax(pa)) == v_to_i[b]: correct += 1
            else:
                total_bits += np.log2(model['V'])
            total_tokens += 1
    return total_bits / max(total_tokens,1), correct / max(total_tokens,1)


def task2_masked_eval(model, test_seqs, mask_rate=0.15, rng=None):
    """Mask interior tokens, predict using P(t_i | t_{i-1}) * P(t_{i+1} | t_i) normalized."""
    if rng is None: rng = np.random.default_rng(0)
    v_to_i = model['v_to_i']; vlist = model['vlist']
    P_fwd = model['P_fwd']; P_bwd = model['P_bwd']; V = model['V']
    P_uni = model['P_uni']
    total_bits, n_masked, correct = 0.0, 0, 0
    for seq in test_seqs:
        if len(seq) < 3: continue
        # mask interior positions
        n_mask = max(1, int((len(seq)-2) * mask_rate))
        positions = rng.choice(np.arange(1, len(seq)-1), n_mask, replace=False)
        for pos in positions:
            left, right = seq[pos-1], seq[pos+1]
            true = seq[pos]
            if true not in v_to_i:
                total_bits += np.log2(V); n_masked += 1; continue
            # P(t | left, right) ∝ P(t|left) * P(right|t) ≈ unnormalized
            if left in v_to_i and right in v_to_i:
                p_left = P_fwd[v_to_i[left]]
                # P(right | t) for each t — that's P_fwd[t][right]
                p_right_given_t = P_fwd[:, v_to_i[right]]
                p_combined = p_left * p_right_given_t
                p_combined = p_combined / max(p_combined.sum(), 1e-12)
            elif left in v_to_i:
                p_combined = P_fwd[v_to_i[left]]
            elif right in v_to_i:
                p_right_given_t = P_fwd[:, v_to_i[right]]
                p_combined = p_right_given_t / max(p_right_given_t.sum(), 1e-12)
            else:
                p_combined = P_uni
            total_bits += -np.log2(max(p_combined[v_to_i[true]], 1e-12))
            if int(np.argmax(p_combined)) == v_to_i[true]: correct += 1
            n_masked += 1
    return total_bits / max(n_masked,1), correct / max(n_masked,1), n_masked


def edit_dist(a, b):
    """Levenshtein distance between two token sequences."""
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    dp[:,0] = np.arange(len(a)+1); dp[0,:] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[len(a), len(b)]


def bleu2(true_seq, pred_seq):
    """Simple BLEU-2: bigram precision with brevity penalty."""
    if len(pred_seq) < 2 or len(true_seq) < 2: return 0.0
    pred_bigrams = list(pairwise(pred_seq))
    true_bigrams = Counter(pairwise(true_seq))
    matches = 0
    for bg in pred_bigrams:
        if true_bigrams[bg] > 0:
            matches += 1; true_bigrams[bg] -= 1
    precision = matches / len(pred_bigrams)
    bp = min(1.0, np.exp(1 - len(true_seq)/len(pred_seq)))
    return bp * precision


def task3_completion_eval(model, test_seqs, prefix_len=3, rng=None):
    """Given prefix of length L, autoregressively complete using argmax."""
    if rng is None: rng = np.random.default_rng(0)
    v_to_i = model['v_to_i']; vlist = model['vlist']
    P_fwd = model['P_fwd']; V = model['V']
    nlls, accs, eds, bleus = [], [], [], []
    for seq in test_seqs:
        if len(seq) < prefix_len + 2: continue
        prefix = seq[:prefix_len]; tail = seq[prefix_len:]
        # Autoregressive argmax completion
        current = prefix[-1]
        completion = []
        for step in range(len(tail)):
            if current in v_to_i:
                pa = P_fwd[v_to_i[current]]
                next_tok = vlist[int(np.argmax(pa))]
            else:
                next_tok = vlist[0]
            completion.append(next_tok)
            current = next_tok
        # NLL of true continuation
        nll = 0; n = 0
        cur = prefix[-1]
        for t in tail:
            if cur in v_to_i and t in v_to_i:
                nll += -np.log2(max(P_fwd[v_to_i[cur]][v_to_i[t]], 1e-12))
            else:
                nll += np.log2(V)
            n += 1; cur = t
        nlls.append(nll / n)
        # acc per token
        accs.append(sum(1 for a, b in zip(completion, tail) if a == b) / max(len(tail),1))
        # edit distance (normalized by max length)
        ed = edit_dist(completion, tail)
        eds.append(ed / max(len(tail),1))
        # BLEU-2
        bleus.append(bleu2(tail, completion))
    return np.mean(nlls), np.mean(accs), np.mean(eds), np.mean(bleus), len(nlls)


# === Run all 3 tasks per context ===
N_SPLITS = 5
results = []
print(f'{"context":12s} {"|V|_g":>6s} {"|V|_pc":>7s} {"":<3s}'
      f'{"T1 bits/tok":>12s} {"T1 acc":>8s} '
      f'{"T2 cloze bits":>14s} {"T2 acc":>8s} '
      f'{"T3 NLL":>8s} {"T3 acc":>8s} {"T3 BLEU2":>10s} {"T3 ED":>7s}'
)
for c in HP1_CTX:
    sg = seqs_global[seqs_global.context==c]
    sp = seqs_pc[seqs_pc.context==c]
    common = set(sg.file) & set(sp.file)
    sg_dict = {r['file']:r['seq'] for _,r in sg.iterrows() if r['file'] in common}
    sp_dict = {r['file']:r['seq'] for _,r in sp.iterrows() if r['file'] in common}
    files = sorted(common)
    if len(files) < 50: continue

    metrics = {'g':defaultdict(list), 'p':defaultdict(list)}
    for s in range(N_SPLITS):
        rng = np.random.default_rng(s)
        files_p = list(files); rng.shuffle(files_p)
        n_train = int(len(files_p) * 0.8)
        tr, te = files_p[:n_train], files_p[n_train:]
        tr_g = [sg_dict[f] for f in tr]; te_g = [sg_dict[f] for f in te]
        tr_p = [sp_dict[f] for f in tr]; te_p = [sp_dict[f] for f in te]
        vg = set(t for sq in tr_g for t in sq); vp = set(t for sq in tr_p for t in sq)
        if not vg or not vp: continue
        mg = fit_bigram(tr_g, vg); mp = fit_bigram(tr_p, vp)
        # T1
        bg, ag = task1_nexttok_eval(mg, te_g); bp, ap = task1_nexttok_eval(mp, te_p)
        metrics['g']['t1_bits'].append(bg); metrics['g']['t1_acc'].append(ag)
        metrics['p']['t1_bits'].append(bp); metrics['p']['t1_acc'].append(ap)
        # T2
        rng2 = np.random.default_rng(s+100)
        bg2, ag2, _ = task2_masked_eval(mg, te_g, rng=rng2); bp2, ap2, _ = task2_masked_eval(mp, te_p, rng=rng2)
        metrics['g']['t2_bits'].append(bg2); metrics['g']['t2_acc'].append(ag2)
        metrics['p']['t2_bits'].append(bp2); metrics['p']['t2_acc'].append(ap2)
        # T3
        nllg, accg, edg, blg, _ = task3_completion_eval(mg, te_g)
        nllp, accp, edp, blp, _ = task3_completion_eval(mp, te_p)
        metrics['g']['t3_nll'].append(nllg); metrics['g']['t3_acc'].append(accg); metrics['g']['t3_ed'].append(edg); metrics['g']['t3_bleu'].append(blg)
        metrics['p']['t3_nll'].append(nllp); metrics['p']['t3_acc'].append(accp); metrics['p']['t3_ed'].append(edp); metrics['p']['t3_bleu'].append(blp)

    Vg = mg['V']; Vp = mp['V']
    def m(key, side): return np.mean(metrics[side][key])
    print(f'{CTX[c]:12s} {Vg:>6d} {Vp:>7d}  '
          f'g={m("t1_bits","g"):.2f}/{m("t1_acc","g"):.2f}  p={m("t1_bits","p"):.2f}/{m("t1_acc","p"):.2f} | '
          f'g={m("t2_bits","g"):.2f}/{m("t2_acc","g"):.2f}  p={m("t2_bits","p"):.2f}/{m("t2_acc","p"):.2f} | '
          f'g={m("t3_nll","g"):.2f}/acc{m("t3_acc","g"):.2f}/bleu{m("t3_bleu","g"):.2f}/ed{m("t3_ed","g"):.2f}  '
          f'p={m("t3_nll","p"):.2f}/acc{m("t3_acc","p"):.2f}/bleu{m("t3_bleu","p"):.2f}/ed{m("t3_ed","p"):.2f}')
    results.append({
        'context':CTX[c], 'V_g':Vg, 'V_pc':Vp,
        't1_bits_g':m('t1_bits','g'), 't1_bits_p':m('t1_bits','p'),
        't1_acc_g':m('t1_acc','g'), 't1_acc_p':m('t1_acc','p'),
        't2_bits_g':m('t2_bits','g'), 't2_bits_p':m('t2_bits','p'),
        't2_acc_g':m('t2_acc','g'), 't2_acc_p':m('t2_acc','p'),
        't3_nll_g':m('t3_nll','g'), 't3_nll_p':m('t3_nll','p'),
        't3_acc_g':m('t3_acc','g'), 't3_acc_p':m('t3_acc','p'),
        't3_bleu_g':m('t3_bleu','g'), 't3_bleu_p':m('t3_bleu','p'),
        't3_ed_g':m('t3_ed','g'), 't3_ed_p':m('t3_ed','p'),
    })

df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/within_context_seq_tasks.csv', index=False)

# Aggregate wins
print('\n=== WINS PER TASK (lower bits/NLL/ED/higher acc/BLEU is better) ===')
print(f'  T1 next-token cardinality-fair (compr ratio):')
df['t1_compr_g'] = df.t1_bits_g / np.log2(df.V_g)
df['t1_compr_p'] = df.t1_bits_p / np.log2(df.V_pc)
print(f'    PC wins: {(df.t1_compr_p < df.t1_compr_g).sum()}/{len(df)}')
print(f'  T1 next-token accuracy:')
print(f'    PC wins: {(df.t1_acc_p > df.t1_acc_g).sum()}/{len(df)}')
print(f'  T2 masked cloze accuracy:')
print(f'    PC wins: {(df.t2_acc_p > df.t2_acc_g).sum()}/{len(df)}')
print(f'  T3 completion accuracy:')
print(f'    PC wins: {(df.t3_acc_p > df.t3_acc_g).sum()}/{len(df)}')
print(f'  T3 completion BLEU-2:')
print(f'    PC wins: {(df.t3_bleu_p > df.t3_bleu_g).sum()}/{len(df)}')
print(f'  T3 completion edit distance (normalized):')
print(f'    PC wins (lower ED): {(df.t3_ed_p < df.t3_ed_g).sum()}/{len(df)}')

print('\nSaved: docs/thesis/figures/within_context_seq_tasks.csv')
