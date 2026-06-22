# Cross-World Generalization on SFT Worlds — Experimental Design

Pre-registration for the empirical section of *Worlds as Local Rules Systems*.
Scope: **headline result + core controls** (one task, one (q, ρ, N) regime).

## Scientific question

The paper's central, currently-unsupported claim:

> *Out-of-world (OOW) performance measures inference rather than memorization; we
> expect this regime to distinguish systems that infer laws from those that memorize.*

We test it directly: **does training a single model on a diverse set of SFT worlds
teach it to infer the rules of new, unseen worlds from observation alone?**

## Paradigm: opaque, in-context meta-learning

- A *world* is a random nearest-neighbour domino SFT `F` over alphabet `A` (|A| = q),
  on an `N x N` window. Dominoes = forbidden horizontal (1x2) and vertical (2x1) pairs.
- **Opaque mode**: the model never sees `F`. Its only window onto the rules is a set of
  `K` admissible *context* configurations sampled from the world.
- **In-context**: an episode = (`K` admissible context configs, 1 query config) → binary
  label. **No test-time weight updates.** To answer OOW the model *must* infer the
  constraints from the context; it cannot store them in weights, because each world's
  symbols and rules are independent.

## Task: T1 validity classification

- Query is admissible (label 1) or inadmissible (label 0), balanced 50/50.
- **Positives**: held-out admissible configs from the world (never identical to a context
  config — no grid leakage; they share dominoes, which is the point).
- **Hard negatives**: take an admissible config, resample cells until the result contains
  ≥1 forbidden domino (verified against `F`). Negatives are genuinely inadmissible and lie
  near the admissible manifold, so the task is non-trivial.
- **Metric**: balanced accuracy and AUROC on held-out queries. Behavioral throughout —
  we never score recovery of `F` itself (non-identifiability makes that meaningless).

## Headline experiment: task-diversity (M) ablation

Train identical models on `M ∈ {1,2,4,8,16,32,64,128,256,512}` worlds, drawn as nested
random subsets of a fixed 512-world master pool. **Hold total training episodes and model
capacity constant** across M, so only world *diversity* varies, not data volume or compute.

Evaluate each trained model on a fixed eval set:
- **IW** (in-world): unseen queries from the M *training* worlds.
- **OOW** (out-of-world): queries from a disjoint held-out pool of worlds never trained on.

**Predicted finding** (cf. Raventós et al. 2023, task diversity → emergent in-context
learning): IW accuracy is high even at small M (memorization is enough); OOW accuracy
rises with M and saturates. The **closing IW–OOW gap as M grows** is the empirical
signature of the paper's thesis — generalization across world families, not memorization.

## Baselines and controls (rigor)

1. **Majority-class** baseline → 0.5 balanced accuracy (sanity floor).
2. **Unseen-domino frequency learner** (non-neural, parameter-free): collect the set of
   dominoes appearing in the `K` context configs; predict the query admissible iff every
   domino in it was observed. This is the Bayes-optimal *behavioral* rule-inducer in the
   infinite-context limit, and a strong finite-K reference the neural model must match/beat.
   It is independent of `M` (uses only context), so it appears as a flat reference line.
3. **Mismatched-context control**: feed the best (M=512) model context from a *different*
   world than the query. Accuracy must collapse toward chance — proof the model uses the
   context, not memorized weights.
4. **SAT oracle** defines ground-truth labels (the 1.0 ceiling).

## Fixed regime and hyperparameters

- `q = 3`, `N = 8`, `ρ = 0.3` (α ≈ 0.96 from the paper's table: non-empty, diverse,
  non-trivial). `K = 8` context configs.
- World filters: non-empty AND ≥2 distinct admissible configs (diversity). Worlds deduped
  by `F`; OOW pool disjoint from training master pool by `F`.
- Configs sampled per world via SAT with randomized variable phases + distinctness
  rejection (decorrelated, non-uniform — bias declared, per the paper's solver protocol).
- Model: in-context Transformer encoder. Symbol + shared 2D-cell-position + context/query
  role embeddings (no per-config index ⇒ invariance to context ordering). A `[READ]`
  token over the query produces the label.
- 3 random seeds per M for error bars. Total training episodes per run held fixed.

## Outputs

- `results/headline.json` — per (M, seed, regime) balanced-acc + AUROC, plus baselines
  and the mismatched-context control.
- `figs/headline.png` — balanced accuracy vs M, IW and OOW curves + baseline reference
  lines, error bars over seeds. The paper's main empirical figure.

## Deliberately out of scope this round (the "full study")

Generalization across ρ / q / N, context-size (K) ablation, CNN-vs-Transformer
architecture comparison, and task T4 (completion). All reuse this harness.
