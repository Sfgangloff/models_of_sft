"""
diag.py — lightweight learning-curve diagnostic (thermally gentle).

Goal: confirm the in-context model actually learns T1, and estimate how many
steps it needs, BEFORE running the full M-ablation. Runs on MPS, caps CPU
threads to 2, nices itself, and throttles each step so the GPU is not pinned at
100% duty cycle.
"""
from __future__ import annotations
import os
# --- keep the machine cool: cap threads BEFORE importing torch/numpy ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try:
    os.nice(10)
except Exception:
    pass

import time, random, pickle, sys
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from worlds import build_world_pool
from episodes import build_batch
from baselines import frequency_learner_batch
from model import InContextT1
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)

# ---- light config (CPU-only, tiny model, low memory) ----
Q, N, RHO, K = 3, 7, 0.3, 5
N_TRAIN, N_OOW, N_CFG = 48, 16, 60
EVAL_HOLD = 15
D_MODEL, LAYERS, NHEAD = 48, 2, 4
STEPS, BATCH, LR = 1200, 16, 3e-4
LOG_EVERY = 100
THROTTLE = 0.0           # CPU at 2 threads is already light
N_EVAL = 200
DEVICE = "cpu"           # avoid MPS memory reservation (machine is swapping)


def view(w, sl):
    return {'F': w['F'], 'key': w['key'], 'configs': w['configs'][sl]}


def get_pool():
    path = os.path.join(CACHE, f"diagpool_q{Q}_N{N}_rho{RHO}.pkl")
    if os.path.exists(path):
        print(f"load {path}", flush=True)
        return pickle.load(open(path, "rb"))
    print("building N=7 pool (single-threaded SAT)...", flush=True)
    rng = random.Random(7)
    minc = EVAL_HOLD + K + 8
    tr = build_world_pool(Q, RHO, N, N_TRAIN, N_CFG, rng, min_configs=minc, label="train", log_every=16)
    excl = {w['key'] for w in tr}
    oo = build_world_pool(Q, RHO, N, N_OOW, N_CFG, rng, exclude_keys=excl, min_configs=minc, label="oow", log_every=8)
    pools = {'train': tr, 'oow': oo}
    pickle.dump(pools, open(path, "wb"))
    print(f"saved {path}", flush=True)
    return pools


def bacc(model, ctx, qry, lab):
    model.eval()
    with torch.no_grad():
        c = torch.from_numpy(ctx.astype(np.int64)).to(DEVICE)
        q = torch.from_numpy(qry.astype(np.int64)).to(DEVICE)
        p = torch.softmax(model(c, q), -1)[:, 1].cpu().numpy()
    model.train()
    return balanced_accuracy_score(lab, (p >= 0.5).astype(int))


def main():
    print(f"device={DEVICE} N={N} K={K} d={D_MODEL} L={LAYERS} steps={STEPS} throttle={THROTTLE}", flush=True)
    pools = get_pool()
    tr_all, oo_all = pools['train'], pools['oow']
    tr_train = [view(w, slice(0, w['configs'].shape[0] - EVAL_HOLD)) for w in tr_all]
    tr_eval = [view(w, slice(w['configs'].shape[0] - EVAL_HOLD, None)) for w in tr_all]

    rng = random.Random(0)
    iw_ctx, iw_qry, iw_lab = build_batch(tr_eval, N_EVAL, K, Q, random.Random(1))
    oo_ctx, oo_qry, oo_lab = build_batch(oo_all, N_EVAL, K, Q, random.Random(2))
    freq = frequency_learner_batch(oo_ctx, oo_qry, Q)
    freq_bacc = balanced_accuracy_score(oo_lab, freq)
    print(f"frequency-learner OOW bacc = {freq_bacc:.3f}", flush=True)

    torch.manual_seed(0)
    model = InContextT1(Q, N, K, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=4*D_MODEL).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        ctx, qry, lab = build_batch(tr_train, BATCH, K, Q, rng)
        c = torch.from_numpy(ctx.astype(np.int64)).to(DEVICE)
        q = torch.from_numpy(qry.astype(np.int64)).to(DEVICE)
        y = torch.from_numpy(lab).to(DEVICE)
        opt.zero_grad(); loss = lossfn(model(c, q), y); loss.backward(); opt.step()
        if THROTTLE:
            time.sleep(THROTTLE)
        if step % LOG_EVERY == 0 or step == 1:
            iw = bacc(model, iw_ctx, iw_qry, iw_lab)
            oo = bacc(model, oo_ctx, oo_qry, oo_lab)
            rate = step / (time.time() - t0)
            print(f"step {step:4d}  loss {loss.item():.3f}  IW {iw:.3f}  OOW {oo:.3f}  "
                  f"(freq {freq_bacc:.3f})  {rate:.1f} steps/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
