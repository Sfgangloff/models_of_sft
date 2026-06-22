"""
t4_curve.py — tuning curve for transparent T4 at higher constraint density.
Trains the rule-conditioned completer at M = all training worlds and logs
IW/OOW completion success vs steps, against the majority baseline and a final
rule-blind comparison. Picks the density/step budget for the full ablation.
Safe envelope: CPU, 2 threads, niced.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try: os.nice(10)
except Exception: pass
import random, pickle, sys, time
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from worlds import build_world_pool
from t4_data import build_batch_t4, completion_success
from t4_model import RuleCompleter
from t4_run import masked_loss, eval_success, majority_success, view

CACHE = os.path.join(os.path.dirname(__file__), "cache")
Q, N, RHO, P = 3, 6, 0.5, 0.4
N_TRAIN, N_OOW, N_CFG, EVAL_HOLD = 48, 16, 40, 10
D_MODEL, LAYERS, NHEAD, DIMFF = 96, 4, 4, 256
STEPS, BATCH, LR, LOG, N_EVAL = 3000, 32, 3e-4, 250, 400


def get_pool():
    path = os.path.join(CACHE, f"t4curve_q{Q}_N{N}_rho{RHO}.pkl")
    if os.path.exists(path): print("load", path, flush=True); return pickle.load(open(path, "rb"))
    print(f"building rho={RHO} pool...", flush=True)
    rng = random.Random(99); minc = EVAL_HOLD + 8
    tr = build_world_pool(Q, RHO, N, N_TRAIN, N_CFG, rng, min_configs=minc, label="train", log_every=16)
    oo = build_world_pool(Q, RHO, N, N_OOW, N_CFG, rng, exclude_keys={w['key'] for w in tr},
                          min_configs=minc, label="oow", log_every=8)
    pools = {'train': tr, 'oow': oo}; pickle.dump(pools, open(path, "wb")); print("saved", flush=True)
    return pools


def main():
    pools = get_pool(); tr_all, oo_all = pools['train'], pools['oow']
    tr_train = [view(w, slice(0, w['configs'].shape[0] - EVAL_HOLD)) for w in tr_all]
    tr_eval = [view(w, slice(w['configs'].shape[0] - EVAL_HOLD, None)) for w in tr_all]
    cfg = dict(q=Q, N=N, p_mask=P)
    ir, im, it, ih, iF = build_batch_t4(tr_eval, N_EVAL, Q, P, random.Random(1))
    orr, om, ot, oh, oF = build_batch_t4(oo_all, N_EVAL, Q, P, random.Random(2))
    maj = majority_success(om, oh, oF, Q)
    print(f"M={len(tr_train)} rho={RHO} p={P}  majority OOW success = {maj:.3f}", flush=True)

    def train(blind, steps):
        torch.manual_seed(0); rng = random.Random(0)
        m = RuleCompleter(Q, N, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=DIMFF)
        opt = torch.optim.Adam(m.parameters(), lr=LR); m.train(); t0 = time.time()
        for step in range(1, steps + 1):
            rule, masked, target, holes, _ = build_batch_t4(tr_train, BATCH, Q, P, rng)
            if blind: rule = np.zeros_like(rule)
            logits = m(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
            opt.zero_grad(); masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward(); opt.step()
            if not blind and (step % LOG == 0):
                iw = eval_success(m, ir, im, ih, iF, Q)
                oo = eval_success(m, orr, om, oh, oF, Q)
                print(f"  step {step:4d}  IW {iw:.3f}  OOW {oo:.3f}  (maj {maj:.3f})  {step/(time.time()-t0):.1f} st/s", flush=True)
        return m

    print("[rule-conditioned]", flush=True)
    train(blind=False, steps=STEPS)
    print("[rule-blind] (final only)", flush=True)
    mb = train(blind=True, steps=STEPS)
    bo = eval_success(mb, orr, om, oh, oF, Q, blind=True)
    print(f"  rule-blind OOW success = {bo:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
