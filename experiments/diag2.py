"""
diag2.py — decisive cheap test of WHY the model stalls at 0.5 for M>=2.

Hypothesis: the in-context skill is gated by a flat optimization plateau, made
worse by 1-cell ("hard") negatives that give almost no early gradient. If so,
EASY negatives (corrupt many cells) should let the SAME small model learn the
in-context comparison quickly at M=8 -- in-world AND out-of-world. If even easy
negatives fail, it's an architecture/plumbing bug instead.

We run M=8 under three negative difficulties and log IW/OOW vs steps.
Same safe envelope: CPU, 2 threads, niced.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try:
    os.nice(10)
except Exception:
    pass
import time, random, pickle, sys
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from worlds import is_admissible
from baselines import frequency_learner_batch
from model import InContextT1
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(__file__); CACHE = os.path.join(HERE, "cache")
Q, N, RHO, K = 3, 6, 0.3, 4
EVAL_HOLD, M = 12, 8
D_MODEL, LAYERS, NHEAD = 64, 3, 4
STEPS, BATCH, LR, LOG_EVERY, N_EVAL = 1000, 16, 3e-4, 200, 300


def view(w, sl):
    return {'F': w['F'], 'key': w['key'], 'configs': w['configs'][sl]}


def make_negative(config, F, q, rng, n_cells):
    """Corrupt exactly up to n_cells cells; resample until inadmissible."""
    Nn = config.shape[0]
    for _ in range(60):
        y = config.copy()
        k = max(1, n_cells)
        for t in rng.sample(range(Nn * Nn), k):
            i, j = divmod(t, Nn)
            ch = [a for a in range(q) if a != config[i, j]]
            y[i, j] = ch[rng.randrange(len(ch))]
        if not is_admissible(y, F, q):
            return y
    return None


def batch(worlds, idxs, B, rng, n_cells):
    Nn = worlds[0]['configs'].shape[1]
    ctx = np.zeros((B, K, Nn, Nn), np.int8); qry = np.zeros((B, Nn, Nn), np.int8); lab = np.zeros((B,), np.int64)
    for b in range(B):
        w = worlds[idxs[rng.randrange(len(idxs))]]
        cfgs = w['configs']; sel = rng.sample(range(cfgs.shape[0]), K + 1)
        ctx[b] = cfgs[sel[:K]]; base = cfgs[sel[K]]
        if rng.random() < 0.5:
            qry[b] = base; lab[b] = 1
        else:
            neg = make_negative(base, w['F'], Q, rng, n_cells)
            qry[b] = neg if neg is not None else base; lab[b] = 0 if neg is not None else 1
    return ctx, qry, lab


def probs(model, ctx, qry):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(ctx.astype(np.int64)),
                                 torch.from_numpy(qry.astype(np.int64))), -1)[:, 1].numpy()
    model.train(); return p


def run(label, n_cells, tr_train, tr_eval, oo_all, idxs):
    torch.manual_seed(0); rng = random.Random(0)
    model = InContextT1(Q, N, K, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=4*D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LR); lf = torch.nn.CrossEntropyLoss()
    ic, iq, il = batch(tr_eval, idxs, N_EVAL, random.Random(555), n_cells)
    oc, oq, ol = batch(oo_all, list(range(len(oo_all))), N_EVAL, random.Random(2), n_cells)
    print(f"\n[{label}] negatives corrupt up to {n_cells} cell(s)", flush=True)
    for step in range(1, STEPS + 1):
        ctx, qry, lab = batch(tr_train, idxs, BATCH, rng, n_cells)
        opt.zero_grad()
        lf(model(torch.from_numpy(ctx.astype(np.int64)), torch.from_numpy(qry.astype(np.int64))),
           torch.from_numpy(lab)).backward(); opt.step()
        if step % LOG_EVERY == 0:
            iw = balanced_accuracy_score(il, (probs(model, ic, iq) >= 0.5).astype(int))
            oo = balanced_accuracy_score(ol, (probs(model, oc, oq) >= 0.5).astype(int))
            print(f"  [{label}] step {step:4d}  IW {iw:.3f}  OOW {oo:.3f}", flush=True)


def main():
    pools = pickle.load(open(os.path.join(CACHE, f"minipool_q{Q}_N{N}_rho{RHO}.pkl"), "rb"))
    tr_all, oo_all = pools['train'], pools['oow']
    tr_train = [view(w, slice(0, w['configs'].shape[0] - EVAL_HOLD)) for w in tr_all]
    tr_eval = [view(w, slice(w['configs'].shape[0] - EVAL_HOLD, None)) for w in tr_all]
    order = list(range(len(tr_all))); random.Random(42).shuffle(order); idxs = order[:M]
    print(f"M={M} d={D_MODEL} L={LAYERS} K={K} steps={STEPS}", flush=True)
    # easy -> medium -> hard
    run("easy",   N*N//2, tr_train, tr_eval, oo_all, idxs)   # corrupt ~half the grid
    run("medium", 3,      tr_train, tr_eval, oo_all, idxs)
    run("hard",   1,      tr_train, tr_eval, oo_all, idxs)   # 1-cell (original task)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
