# Experiment Log — cross-world generalization on SFT worlds

Chronological lab notebook. Each entry: setup, config, result, takeaway.
Companion to `REPORT.md` (the narrative writeup). Numbers are single-seed unless
noted. All runs CPU-only, 2 threads, niced (see `memory/machine-constraints`).

Common regime knobs: q = alphabet size, N = grid size, ρ = forbidden-domino
density, M = number of training worlds, K = in-context examples (opaque only).

---

## Phase A — OPAQUE in-context, task T1 (validity classification)  [superseded]

The model infers rules from K example grids; classifies a query valid/invalid.

**A1. mini M-ablation** (`mini.py`, `results/mini.json`)
- q3, N6, K4, ρ0.3; d48/L2; 700 steps; M∈{1,8,32}.
- Frequency-learner baseline OOW = **0.979** (task is solvable & calibrated).
- M=1: IW **0.808**, OOW **0.500**. M=8: 0.500/0.500. M=32: 0.500/0.500.
- Mismatched-context control: 0.500.
- Takeaway: M=1 = pure memorization (high IW, chance OOW). M≥8 collapses to chance
  even IW, because the same query has different labels in different worlds → no
  query-only shortcut; model must learn in-context inference and did not.

**A2. emergence run** (`emerge.py`)
- q3, N6, K5; d80/L4; up to 2000+ steps; M=32. IW & OOW pinned at **0.500** throughout.
- Takeaway: in-context inference did not emerge at this scale.

**A3. difficulty sweep** (`diag2.py`) — why A1/A2 stalled
- M=8; vary negative difficulty (cells corrupted): 
  - easy (18/36 cells): IW 0.50→**0.807**, OOW 0.50→**0.647**.
  - medium (3 cells): ~0.50.  hard (1 cell, the real task): ~0.50.
- Takeaway: plumbing is fine (easy learns); the 1-cell negatives give almost no
  early gradient → optimization plateau. Genuine in-context rule inference needs
  far more scale. **Decision: pivot to transparent mode (rules given as input).**

---

## Phase B — TRANSPARENT, task T4 (completion); rules given as input  [headline]

Model gets rules F (binary vector over all dominoes) + masked grid; fills holes.

**B1. completion-metric smoke** (`t4_run.py --smoke`, `results/t4_smoke.json`)
- q3, N6, ρ0.3, mask p0.4; d64/L3; 200 steps; M∈{1,4,16}.
- Majority-fill OOW success = 0.473.
- M=1: IW 0.013 / OOW 0.003. M=4: 0.270 / 0.083. M=16: 0.413 / **0.450**.
- rule-blind OOW 0.457; mismatch 0.360.
- Takeaway: OOW completion-success rises with M — but ρ0.3 is too weak (majority
  already 0.47), so rules barely matter under this metric.

**B2. tuning curve at ρ0.5** (`t4_curve.py`)
- q3, N6, ρ0.5, p0.4; d96/L4; 3000 steps; M=48. Majority OOW 0.338.
- Rule-conditioned: IW →**0.565**, OOW →**0.430**. Rule-blind OOW **0.427**.
- Takeaway: under the lenient completion metric, rule-conditioned ≈ rule-blind OOW
  → metric doesn't isolate rule-use. **Decision: switch to forced-cell metric.**

**B3. forced-cell comparison** (`t4_forced.py`, `results/t4_forced.json`)
- Forced cell = masked cell whose value is uniquely determined by rules + observed
  neighbours. q3, N6, ρ0.5; d96/L4; 2500 steps; M=48; ~3600 forced cells.
- Rule-conditioned: IW **0.895**, OOW **0.844**.  Wrong rules (mismatch): **0.616**.
  Rule-blind: **0.794**.  Chance: 0.333.
- Takeaways: (1) generalizes OOW≈IW ≫ chance; (2) wrong rules < blind ⇒ rules are
  causally used; (3) blind still 0.794 ⇒ strong cross-world statistical shortcut at
  ρ0.5, so rule-use is proven by the *mismatch* control, not the blind comparison.

**B4. forced-cell M-ablation** (`t4_ablation.py`, `results/t4_ablation.json`,
`figs/t4_ablation.png`)  ← **headline figure**
- q3, N6, ρ0.5; d80/L3; 1500 steps; M∈{1,4,16,48}. Forced-cell accuracy:

  | M | IW | OOW | OOW wrong-rules |
  |---|----|-----|-----------------|
  | 1  | 0.998 | 0.521 | 0.522 |
  | 4  | 0.982 | 0.551 | 0.567 |
  | 16 | 0.871 | 0.621 | 0.547 |
  | 48 | 0.842 | **0.773** | 0.693 |

- Takeaway: OOW rises monotonically with M; IW–OOW gap closes (memorization →
  generalization); correct-vs-wrong-rules gap widens with M (rule-use emerges with
  world diversity).

---

## Standing conclusions

1. Transparent T4: a model **generalizes across worlds** and **genuinely uses the
   given rules**; both improve with the number of training worlds.
2. Opaque T1 at this scale: only memorization (M=1); in-context inference needs scale.
3. Metric choice matters: completion-admissibility is too lenient; forced-cell
   isolates rule-use.
4. The rule-blind baseline measures the task's statistical shortcut; the mismatch
   control is the clean causal test of rule-use.

---

## Phase C — Coverage / mechanism (Exp C)  [`t4_coverage.py`, `results/t4_coverage.json`, `figs/t4_coverage.png`]

Hold out U = 6 of 18 dominoes; training worlds forbid only the other 12 (S). Test
worlds forbid k=4 dominoes with j drawn from U (j=0..4). Forced-cell accuracy split
by whether a cell's forcing depends on a held-out (U) domino. q3, N6; d80/L3; 2000
steps; 64 training worlds, 14 test worlds/j. Chance = 0.333.

| j | S-sufficient cells | U-critical cells | rule-blind, U-critical |
|---|---|---|---|
| 0 | 0.698 (n=1907) | — | — |
| 1 | 0.742 (n=948) | **0.367** (n=1440) | 0.409 |
| 2 | 0.882 (n=390) | **0.299** (n=1596) | 0.316 |
| 3 | — | **0.328** (n=1325) | 0.320 |
| 4 | — | **0.233** (n=1548) | 0.238 |

**Result: strict coverage confirmed — the model learned PER-DOMINO LOOKUPS, not a
general "avoid forbidden pair" operation.** On cells whose forcing needs a held-out
domino, the rule-conditioned model sits at chance and is **indistinguishable from
rule-blind** (red ≈ gray in the figure), even though it is *given* the U-bit at test
and the architecture shares the forbidden-bit embedding across dominoes. On
S-sufficient cells it is high. (j=4 dips below chance: with only-U constraints the
model's S-tuned statistics actively mislead — same for blind.)

**Implication:** a training set's value is governed by its **coverage of the rule
space** — each domino must be *exercised* (forbidden somewhere) in training for the
model to use it. This is a concrete characterization of "good training worlds" and a
mechanism result. Caveat: shown at this scale/architecture; more scale or a
constraint-applying architecture might learn the general operation.

## Open threads (see REPORT §9 and discussion)
- Which training worlds help most, and how to characterize them (coverage of
  forbidden-domino "bits"? dynamical invariants?).
- Robustness across task (T1/T3/T5), mask geometry, model type (CNN/GNN/transformer),
  q, N, ρ. Scaling in M / steps / model size. Mechanism (does it do propagation?).
