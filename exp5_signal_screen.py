"""Experiment 5: signal-strength screen (see exp5_preregistration.md).
Usage: python exp5_signal_screen.py --out results_exp5 --stage screen      (seeds 0-2, full grid)
       python exp5_signal_screen.py --out results_exp5 --stage confirm --configs "<config id>" ...  (seeds 10-14)
"""
import argparse, itertools, os, random, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results_exp5'); ap.add_argument('--stage', choices=['screen', 'confirm'], default='screen')
ap.add_argument('--configs', nargs='*', default=None); ap.add_argument('--data_root', default='data')
cli = ap.parse_args(); os.makedirs(cli.out, exist_ok=True)
torch.set_num_threads(max(4, os.cpu_count() or 4))
SEEDS = [0, 1, 2] if cli.stage == 'screen' else [10, 11, 12, 13, 14]
GRID = list(itertools.product(['Cora', 'CiteSeer'], [64, 128], [0.0, 0.5], ['many', 'few']))
cid = lambda c: f'{c[0]}|h{c[1]}|do{c[2]}|{c[3]}'
if cli.configs: GRID = [c for c in GRID if cid(c) in cli.configs]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class GCN(nn.Module):
    def __init__(self, i, h, o, p):
        super().__init__(); self.c1, self.c2, self.p = GCNConv(i, h), GCNConv(h, o), p
    def forward(self, x, ei):
        x = F.dropout(x, self.p, self.training); x = F.relu(self.c1(x, ei))
        x = F.dropout(x, self.p, self.training); return self.c2(x, ei)

def train(data, ei, mask, hidden, p, seed):
    set_seed(seed)
    m = GCN(data.num_features, hidden, int(data.y.max()) + 1, p)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        m.train(); opt.zero_grad(); F.cross_entropy(m(data.x, ei)[mask], data.y[mask]).backward(); opt.step()
    m.eval(); return m

_DS = {}
OUTF = f'{cli.out}/{cli.stage}.csv'
done = set(pd.read_csv(OUTF).config) if os.path.exists(OUTF) else set()
for (ds, hidden, p, labels) in GRID:
    if cid((ds, hidden, p, labels)) in done: continue
    rows = []
    if ds not in _DS: _DS[ds] = Planetoid(cli.data_root, ds, transform=T.NormalizeFeatures())[0]
    data = _DS[ds]; N = data.num_nodes; y = data.y.numpy()
    for seed in SEEDS:
        t0 = time.time(); rng = np.random.default_rng(seed)
        perm = rng.permutation(N); n_tr = int(0.9 * N) if labels == 'many' else int(0.1 * N)
        tr, held = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        D = np.sort(rng.choice(tr, int(0.1 * len(tr)), replace=False)); retained = np.setdiff1d(tr, D)
        hp = rng.permutation(held); H_cal, H_eval = hp[: len(hp) // 2], hp[len(hp) // 2:]
        keep = torch.ones(N, dtype=torch.bool); keep[torch.from_numpy(D)] = False
        ei_red, _ = subgraph(keep, data.edge_index, num_nodes=N)          # deleted nodes' edges removed
        m_full = torch.zeros(N, dtype=torch.bool); m_full[torch.from_numpy(tr)] = True
        m_red = torch.zeros(N, dtype=torch.bool); m_red[torch.from_numpy(retained)] = True
        models = {'Original': train(data, data.edge_index, m_full, hidden, p, 1000 + seed),
                  'Retrain-A': train(data, ei_red, m_red, hidden, p, 2000 + seed),
                  'Retrain-B': train(data, ei_red, m_red, hidden, p, 3000 + seed)}
        R_cal = rng.choice(retained, size=min(len(retained), len(H_cal)), replace=False)
        res = {}
        for name, m in models.items():
            with torch.no_grad():
                P = F.softmax(m(data.x, data.edge_index), 1).numpy()      # same full graph for every model
            conf = P[np.arange(N), y]
            auc = roc_auc_score(np.r_[np.ones(len(D)), np.zeros(len(H_eval))], np.r_[conf[D], conf[H_eval]])
            cand = np.unique(np.r_[conf[R_cal], conf[H_cal]])
            thr = cand[int(np.argmax([((conf[R_cal] >= t).mean() + (conf[H_cal] < t).mean()) / 2 for t in cand]))]
            res[name] = dict(auc=auc, adv=2 * abs(auc - 0.5), tpr_minus_fpr=(conf[D] >= thr).mean() - (conf[H_eval] >= thr).mean(),
                             test_acc=float((P.argmax(1)[held] == y[held]).mean()), train_acc=float((P.argmax(1)[tr] == y[tr]).mean()))
        rows.append(dict(config=cid((ds, hidden, p, labels)), dataset=ds, hidden=hidden, dropout=p, labels=labels, seed=seed,
                         n_deleted=len(D), n_heldout_eval=len(H_eval),
                         auc_orig=res['Original']['auc'], auc_retA=res['Retrain-A']['auc'], auc_retB=res['Retrain-B']['auc'],
                         gap=res['Original']['auc'] - res['Retrain-A']['auc'],
                         retrain_noise=abs(res['Retrain-A']['auc'] - res['Retrain-B']['auc']),
                         adv_orig=res['Original']['adv'], adv_retA=res['Retrain-A']['adv'],
                         tpr_fpr_orig=res['Original']['tpr_minus_fpr'], tpr_fpr_retA=res['Retrain-A']['tpr_minus_fpr'],
                         test_acc_orig=res['Original']['test_acc'], train_acc_orig=res['Original']['train_acc'],
                         test_acc_retA=res['Retrain-A']['test_acc']))
        print(f"[{cid((ds, hidden, p, labels))} seed {seed}] gap {rows[-1]['gap']:+.3f} noise {rows[-1]['retrain_noise']:.3f} "
              f"test acc {rows[-1]['test_acc_orig']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUTF, mode='a', header=not os.path.exists(OUTF), index=False)   # saved after every configuration
