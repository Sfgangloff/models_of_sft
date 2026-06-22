"""
t4_run.py — TRANSPARENT T4 (completion) cross-world ablation.

Headline: given the rules F as input + a partially-masked admissible grid, fill
the holes admissibly. Train on M worlds, test in-world (IW) and out-of-world
(OOW). Metric = completion success rate (fraction of completed grids that are
admissible). The question: does training on many rule-sets teach the model to
USE the given rules, so it completes correctly for rule-sets never seen (OOW)?

Key baselines:
  - rule-blind neural: identical model fed an all-zeros rule vector (told nothing
    about F). If rules matter, its OOW success is far lower.
  - majority: fill each hole with the most common observed symbol.
  - mismatched-rule control: feed another world's rules; success should drop.

Safe envelope: CPU, 2 threads, niced. Short sequences (rule + grid tokens), so
this is much cheaper than the opaque in-context design.
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
import argparse, json, random, pickle, sys, time, gc
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(__file__))
from worlds import build_world_pool, is_admissible
from t4_data import build_batch_t4, completion_success
from t4_model import RuleCompleter

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figs"); CACHE = os.path.join(HERE, "cache")
for d in (RES, FIG, CACHE): os.makedirs(d, exist_ok=True)


def get_config(smoke):
    if smoke:
        return dict(q=3, N=6, rho=0.3, p_mask=0.4, n_train=16, n_oow=8, n_cfg=40,
                    eval_hold=12, M_list=[1, 4, 16], seeds=[0], steps=200, batch=32,
                    lr=3e-4, d_model=64, layers=3, nhead=4, dim_ff=128, n_eval=300)
    return dict(q=3, N=6, rho=0.3, p_mask=0.4, n_train=64, n_oow=24, n_cfg=50,
                eval_hold=14, M_list=[1, 2, 4, 8, 16, 32, 64], seeds=[0, 1], steps=2500,
                batch=32, lr=3e-4, d_model=96, layers=4, nhead=4, dim_ff=256, n_eval=500)


def view(w, sl):
    return {'F': w['F'], 'key': w['key'], 'configs': w['configs'][sl]}


def get_pool(cfg):
    tag = f"t4pool_q{cfg['q']}_N{cfg['N']}_rho{cfg['rho']}_tr{cfg['n_train']}_oow{cfg['n_oow']}"
    path = os.path.join(CACHE, f"{tag}.pkl")
    if os.path.exists(path):
        print(f"load {path}", flush=True); return pickle.load(open(path, "rb"))
    print("building T4 world pool (single-threaded SAT)...", flush=True)
    rng = random.Random(2024); minc = cfg['eval_hold'] + 10
    tr = build_world_pool(cfg['q'], cfg['rho'], cfg['N'], cfg['n_train'], cfg['n_cfg'], rng,
                          min_configs=minc, label="train", log_every=16)
    excl = {w['key'] for w in tr}
    oo = build_world_pool(cfg['q'], cfg['rho'], cfg['N'], cfg['n_oow'], cfg['n_cfg'], rng,
                          exclude_keys=excl, min_configs=minc, label="oow", log_every=12)
    pools = {'train': tr, 'oow': oo}; pickle.dump(pools, open(path, "wb")); print("saved", flush=True)
    return pools


def masked_loss(logits, target, holes):
    B, N, _, q = logits.shape
    lf = nn.CrossEntropyLoss()
    flat = logits.reshape(-1, q); tgt = target.reshape(-1); m = holes.reshape(-1)
    return lf(flat[m], tgt[m])


@torch.no_grad()
def eval_success(model, rule, masked, holes, Fs, q, blind=False):
    model.eval()
    r = torch.zeros_like(torch.from_numpy(rule)) if blind else torch.from_numpy(rule)
    logits = model(r, torch.from_numpy(masked.astype(np.int64)))
    pred = logits.argmax(-1).numpy().astype(np.int8)
    model.train()
    return completion_success(pred, masked, holes, Fs, q)


def majority_success(masked, holes, Fs, q):
    """Fill each hole with the most common observed symbol in that grid."""
    B = masked.shape[0]; ok = 0
    for b in range(B):
        obs = masked[b][~holes[b]]
        if obs.size:
            vals, cnts = np.unique(obs, return_counts=True); fill = vals[cnts.argmax()]
        else:
            fill = 0
        g = masked[b].copy(); g[holes[b]] = fill
        if is_admissible(g, Fs[b], q):
            ok += 1
    return ok / B


def train_model(train_views, world_idx, cfg, seed, blind=False):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    model = RuleCompleter(cfg['q'], cfg['N'], d_model=cfg['d_model'], nhead=cfg['nhead'],
                          num_layers=cfg['layers'], dim_ff=cfg['dim_ff'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr']); model.train()
    for step in range(cfg['steps']):
        rule, masked, target, holes, _ = build_batch_t4(train_views, cfg['batch'], cfg['q'],
                                                        cfg['p_mask'], rng, world_indices=world_idx)
        if blind:
            rule = np.zeros_like(rule)
        logits = model(torch.from_numpy(rule), torch.from_numpy(masked.astype(np.int64)))
        opt.zero_grad()
        masked_loss(logits, torch.from_numpy(target.astype(np.int64)), torch.from_numpy(holes)).backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    cfg = get_config(args.smoke)
    print(f"config={json.dumps(cfg)}", flush=True)
    pools = get_pool(cfg)
    tr_all, oo_all = pools['train'], pools['oow']
    H = cfg['eval_hold']
    tr_train = [view(w, slice(0, w['configs'].shape[0] - H)) for w in tr_all]
    tr_eval = [view(w, slice(w['configs'].shape[0] - H, None)) for w in tr_all]
    q = cfg['q']

    # fixed OOW eval set
    oc_rule, oc_mask, oc_tgt, oc_holes, oc_Fs = build_batch_t4(oo_all, cfg['n_eval'], q, cfg['p_mask'], random.Random(777))
    maj_oow = majority_success(oc_mask, oc_holes, oc_Fs, q)
    print(f"OOW majority-fill success = {maj_oow:.3f}", flush=True)

    results = {'config': cfg, 'baselines': {'majority_oow': maj_oow}, 'runs': []}
    nt = len(tr_train)
    order = list(range(nt)); random.Random(424242).shuffle(order)

    for M in cfg['M_list']:
        if M > nt:
            continue
        for seed in cfg['seeds']:
            t0 = time.time()
            widx = order[:M]
            model = train_model(tr_train, widx, cfg, seed)
            iw_views = [tr_eval[i] for i in widx]
            ir, im, it, ih, iF = build_batch_t4(iw_views, cfg['n_eval'], q, cfg['p_mask'], random.Random(555 + seed))
            iw = eval_success(model, ir, im, ih, iF, q)
            oow = eval_success(model, oc_rule, oc_mask, oc_holes, oc_Fs, q)
            rec = dict(M=M, seed=seed, iw=iw, oow=oow, secs=time.time() - t0)
            results['runs'].append(rec)
            print(f"M={M:3d} seed={seed}  IW {iw:.3f}  OOW {oow:.3f}  ({rec['secs']:.0f}s)", flush=True)
            del model; gc.collect()

    # rule-blind neural baseline + mismatched-rule control at largest M
    Mmax = max(m for m in cfg['M_list'] if m <= nt); widx = order[:Mmax]
    blind = train_model(tr_train, widx, cfg, cfg['seeds'][0], blind=True)
    blind_oow = eval_success(blind, oc_rule, oc_mask, oc_holes, oc_Fs, q, blind=True)
    results['baselines']['rule_blind_oow'] = blind_oow
    print(f"rule-blind neural OOW success = {blind_oow:.3f}", flush=True)

    full = train_model(tr_train, widx, cfg, cfg['seeds'][0])
    mr, mm, mt, mh, mF = build_batch_t4(oo_all, cfg['n_eval'], q, cfg['p_mask'], random.Random(888), mismatch=True)
    mismatch_oow = eval_success(full, mr, mm, mh, mF, q)
    results['mismatched_rule_oow'] = mismatch_oow
    print(f"mismatched-rule control OOW success = {mismatch_oow:.3f} (should drop vs OOW)", flush=True)

    out = os.path.join(RES, "t4_smoke.json" if args.smoke else "t4.json")
    json.dump(results, open(out, "w"), indent=2); print(f"saved {out}", flush=True)
    make_plot(results, cfg, args.smoke)
    print("DONE", flush=True)


def make_plot(results, cfg, smoke):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from collections import defaultdict
    iw = defaultdict(list); oow = defaultdict(list)
    for r in results['runs']:
        iw[r['M']].append(r['iw']); oow[r['M']].append(r['oow'])
    Ms = sorted(iw)
    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(Ms, [np.mean(iw[m]) for m in Ms], yerr=[np.std(iw[m]) for m in Ms], marker='o', capsize=3, label='In-world (IW)')
    plt.errorbar(Ms, [np.mean(oow[m]) for m in Ms], yerr=[np.std(oow[m]) for m in Ms], marker='s', capsize=3, label='Out-of-world (OOW)')
    plt.axhline(results['baselines'].get('rule_blind_oow', np.nan), ls='--', c='red', label='Rule-blind model (OOW)')
    plt.axhline(results['baselines']['majority_oow'], ls=':', c='gray', label='Majority-fill (OOW)')
    plt.xscale('log', base=2); plt.xlabel('Number of training worlds  M'); plt.ylabel('Completion success rate')
    plt.title(f"Transparent T4 completion (q={cfg['q']}, N={cfg['N']}, ρ={cfg['rho']}, mask p={cfg['p_mask']})")
    plt.ylim(0, 1.02); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(FIG, "t4_smoke.png" if smoke else "t4.png")
    plt.savefig(out, dpi=140); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
