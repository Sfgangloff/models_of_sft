"""
emerge.py — single curve-logged run to test whether in-context rule inference
emerges with enough training. Same safe envelope as mini.py (CPU, 2 threads,
niced). Trains at M = all training worlds and logs IW/OOW vs steps.
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
from episodes import build_batch
from baselines import frequency_learner_batch
from model import InContextT1
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(__file__); CACHE = os.path.join(HERE, "cache")
Q, N, RHO, K = 3, 6, 0.3, 5
EVAL_HOLD = 12
D_MODEL, LAYERS, NHEAD = 80, 4, 4
STEPS, BATCH, LR = 4000, 24, 3e-4
LOG_EVERY, N_EVAL = 250, 300


def view(w, sl):
    return {'F': w['F'], 'key': w['key'], 'configs': w['configs'][sl]}


def probs(model, ctx, qry):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(ctx.astype(np.int64)),
                                 torch.from_numpy(qry.astype(np.int64))), -1)[:, 1].numpy()
    model.train(); return p


def main():
    pools = pickle.load(open(os.path.join(CACHE, f"minipool_q{Q}_N{N}_rho{RHO}.pkl"), "rb"))
    tr_all, oo_all = pools['train'], pools['oow']
    tr_train = [view(w, slice(0, w['configs'].shape[0] - EVAL_HOLD)) for w in tr_all]
    tr_eval = [view(w, slice(w['configs'].shape[0] - EVAL_HOLD, None)) for w in tr_all]
    print(f"M={len(tr_train)} d={D_MODEL} L={LAYERS} K={K} steps={STEPS} batch={BATCH}", flush=True)

    ic, iq, il = build_batch(tr_eval, N_EVAL, K, Q, random.Random(555))
    oc, oq, ol = build_batch(oo_all, N_EVAL, K, Q, random.Random(2))
    freq = balanced_accuracy_score(ol, frequency_learner_batch(oc, oq, Q))
    print(f"frequency-learner OOW = {freq:.3f}", flush=True)

    torch.manual_seed(0); rng = random.Random(0)
    model = InContextT1(Q, N, K, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=4*D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=LR); lf = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        ctx, qry, lab = build_batch(tr_train, BATCH, K, Q, rng)
        opt.zero_grad()
        lf(model(torch.from_numpy(ctx.astype(np.int64)), torch.from_numpy(qry.astype(np.int64))),
           torch.from_numpy(lab)).backward()
        opt.step()
        if step % LOG_EVERY == 0:
            iw = balanced_accuracy_score(il, (probs(model, ic, iq) >= 0.5).astype(int))
            oo = balanced_accuracy_score(ol, (probs(model, oc, oq) >= 0.5).astype(int))
            print(f"step {step:4d}  IW {iw:.3f}  OOW {oo:.3f}  (freq {freq:.3f})  "
                  f"{step/(time.time()-t0):.1f} st/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
