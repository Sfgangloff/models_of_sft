"""
t4_ablation.py — forced-cell M-ablation for transparent T4.

Trains the rule-conditioned completer on M worlds for M in a ladder, and measures
forced-cell accuracy in-world (IW) and out-of-world (OOW), plus the mismatched-rule
control. Headline question: does OOW rule-use improve as the number of training
worlds grows? Writes results after EACH M so a drive drop can't wipe progress.
Safe envelope: CPU, 2 threads, niced.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try: os.nice(10)
except Exception: pass
import json, random, pickle, sys, time, gc
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from t4_data import build_batch_t4, forced_cell_stats
from t4_model import RuleCompleter
from t4_run import masked_loss, view

HERE = os.path.dirname(__file__); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
Q, N, RHO, P = 3, 6, 0.5, 0.4
EVAL_HOLD = 10
D_MODEL, LAYERS, NHEAD, DIMFF = 80, 3, 4, 256
M_LIST = [1, 4, 16, 48]
STEPS, BATCH, LR, N_EVAL = 1500, 32, 3e-4, 500
POOL = os.path.join(CACHE, f"t4curve_q{Q}_N{N}_rho{RHO}.pkl")
OUT = os.path.join(RES, "t4_ablation.json")


@torch.no_grad()
def facc(model, rule, masked, holes, Fs, blind=False):
    model.eval()
    r = torch.zeros_like(torch.from_numpy(rule)) if blind else torch.from_numpy(rule)
    pred = model(r, torch.from_numpy(masked.astype(np.int64))).argmax(-1).numpy().astype(np.int8)
    model.train()
    a, _ = forced_cell_stats(pred, masked, holes, Fs, Q)
    return a


def train(train_views, widx, seed):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    m = RuleCompleter(Q, N, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=DIMFF)
    opt = torch.optim.Adam(m.parameters(), lr=LR); m.train()
    for _ in range(STEPS):
        rule, masked, target, holes, _ = build_batch_t4(train_views, BATCH, Q, P, rng, world_indices=widx)
        logits = m(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
        opt.zero_grad(); masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward(); opt.step()
    return m


def main():
    pools = pickle.load(open(POOL, "rb")); tr_all, oo_all = pools['train'], pools['oow']
    tr = [view(w, slice(0, w['configs'].shape[0]-EVAL_HOLD)) for w in tr_all]
    tev = [view(w, slice(w['configs'].shape[0]-EVAL_HOLD, None)) for w in tr_all]
    orr, om, ot, oh, oF = build_batch_t4(oo_all, N_EVAL, Q, P, random.Random(2))
    mmr, mmm, mmt, mmh, mmF = build_batch_t4(oo_all, N_EVAL, Q, P, random.Random(9), mismatch=True)
    nt = len(tr); order = list(range(nt)); random.Random(424242).shuffle(order)
    print(f"forced-cell M-ablation  Ms={M_LIST}  d={D_MODEL} L={LAYERS} steps={STEPS}", flush=True)

    res = {"config": dict(q=Q, N=N, rho=RHO, p=P, d_model=D_MODEL, layers=LAYERS, steps=STEPS, M_list=M_LIST),
           "chance": 1.0/Q, "runs": []}
    for M in M_LIST:
        if M > nt: continue
        t0 = time.time(); widx = order[:M]
        model = train(tr, widx, seed=0)
        iw_views = [tev[i] for i in widx]
        ir, im, it, ih, iF = build_batch_t4(iw_views, N_EVAL, Q, P, random.Random(555))
        iw = facc(model, ir, im, ih, iF)
        oo = facc(model, orr, om, oh, oF)
        mm = facc(model, mmr, mmm, mmh, mmF)
        rec = dict(M=M, forced_iw=iw, forced_oow=oo, forced_mismatch_oow=mm, secs=time.time()-t0)
        res["runs"].append(rec)
        json.dump(res, open(OUT, "w"), indent=2)          # incremental save
        print(f"M={M:3d}  forced IW {iw:.3f}  OOW {oo:.3f}  mismatch {mm:.3f}  ({rec['secs']:.0f}s) [saved]", flush=True)
        del model; gc.collect()

    # plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ms = [r['M'] for r in res['runs']]
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(Ms, [r['forced_iw'] for r in res['runs']], 'o-', label='In-world (correct rules)')
        plt.plot(Ms, [r['forced_oow'] for r in res['runs']], 's-', label='Out-of-world (correct rules)')
        plt.plot(Ms, [r['forced_mismatch_oow'] for r in res['runs']], '^--', c='orange', label='OOW, wrong rules (control)')
        plt.axhline(res['chance'], ls=':', c='k', label='Chance (1/q)')
        plt.xscale('log', base=2); plt.xlabel('Number of training worlds  M'); plt.ylabel('Forced-cell accuracy')
        plt.title(f'Transparent T4 — rule-use vs world diversity (q={Q}, N={N}, ρ={RHO})')
        plt.ylim(0, 1.02); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(HERE, "figs", "t4_ablation.png"), dpi=140)
        print("saved figs/t4_ablation.png", flush=True)
    except Exception as e:
        print("plot skipped:", e, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
