"""
t4_forced.py — sharp test of genuine rule-use via the forced-cell metric.

Trains a rule-conditioned and a rule-blind completer on the same worlds and
compares accuracy on *locally-forced* masked cells (cells whose value is uniquely
determined by the rules + observed neighbours). A rule-blind model cannot know
these, so a large rule-conditioned >> rule-blind gap = the model genuinely uses
the rules and that use generalizes out-of-world. Safe envelope: CPU/2 threads.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
try: os.nice(10)
except Exception: pass
import json, random, pickle, sys, time
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from t4_data import build_batch_t4, completion_success, forced_cell_stats
from t4_model import RuleCompleter
from t4_run import masked_loss, view

HERE = os.path.dirname(__file__); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
Q, N, RHO, P = 3, 6, 0.5, 0.4
EVAL_HOLD = 10
D_MODEL, LAYERS, NHEAD, DIMFF = 96, 4, 4, 256
STEPS, BATCH, LR, LOG, N_EVAL = 2500, 32, 3e-4, 500, 500
POOL = os.path.join(CACHE, f"t4curve_q{Q}_N{N}_rho{RHO}.pkl")


@torch.no_grad()
def evaluate(model, rule, masked, holes, Fs, blind):
    model.eval()
    r = torch.zeros_like(torch.from_numpy(rule)) if blind else torch.from_numpy(rule)
    pred = model(r, torch.from_numpy(masked.astype(np.int64))).argmax(-1).numpy().astype(np.int8)
    model.train()
    succ = completion_success(pred, masked, holes, Fs, Q)
    facc, nf = forced_cell_stats(pred, masked, holes, Fs, Q)
    return succ, facc, nf


def train(blind, ev):
    torch.manual_seed(0); rng = random.Random(0)
    m = RuleCompleter(Q, N, d_model=D_MODEL, nhead=NHEAD, num_layers=LAYERS, dim_ff=DIMFF)
    opt = torch.optim.Adam(m.parameters(), lr=LR); m.train(); t0 = time.time()
    tag = "blind" if blind else "rule"
    for step in range(1, STEPS + 1):
        rule, masked, target, holes, _ = build_batch_t4(ev['tr'], BATCH, Q, P, rng)
        if blind: rule = np.zeros_like(rule)
        logits = m(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
        opt.zero_grad(); masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward(); opt.step()
        if step % LOG == 0:
            s_iw, f_iw, _ = evaluate(m, *ev['iw'], blind)
            s_oo, f_oo, nf = evaluate(m, *ev['oo'], blind)
            print(f"  [{tag}] step {step:4d}  forced-acc IW {f_iw:.3f} OOW {f_oo:.3f} | "
                  f"compl OOW {s_oo:.3f}  (nforced~{nf})  {step/(time.time()-t0):.1f}st/s", flush=True)
    return m


def main():
    pools = pickle.load(open(POOL, "rb")); tr_all, oo_all = pools['train'], pools['oow']
    tr = [view(w, slice(0, w['configs'].shape[0]-EVAL_HOLD)) for w in tr_all]
    tev = [view(w, slice(w['configs'].shape[0]-EVAL_HOLD, None)) for w in tr_all]
    ir, im, it, ih, iF = build_batch_t4(tev, N_EVAL, Q, P, random.Random(1))
    orr, om, ot, oh, oF = build_batch_t4(oo_all, N_EVAL, Q, P, random.Random(2))
    ev = {'tr': tr, 'iw': (ir, im, ih, iF), 'oo': (orr, om, oh, oF)}
    print(f"M={len(tr)} rho={RHO} | training rule-conditioned then rule-blind", flush=True)

    print("[rule-conditioned]", flush=True); mr = train(False, ev)
    print("[rule-blind]", flush=True); mb = train(True, ev)

    sr_oo, fr_oo, nf = evaluate(mr, *ev['oo'], False)
    sr_iw, fr_iw, _ = evaluate(mr, *ev['iw'], False)
    sb_oo, fb_oo, _ = evaluate(mb, *ev['oo'], True)
    # mismatched-rule control on rule-conditioned model
    mmr, mmm, mmt, mmh, mmF = build_batch_t4(oo_all, N_EVAL, Q, P, random.Random(9), mismatch=True)
    sm_oo, fm_oo, _ = evaluate(mr, mmr, mmm, mmh, mmF, False)

    out = dict(rho=RHO, n_forced=nf,
               rule_forced_iw=fr_iw, rule_forced_oow=fr_oo, blind_forced_oow=fb_oo,
               mismatch_forced_oow=fm_oo, chance=1.0/Q,
               rule_compl_oow=sr_oo, blind_compl_oow=sb_oo)
    print("\n=== FORCED-CELL SUMMARY ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    json.dump(out, open(os.path.join(RES, "t4_forced.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
