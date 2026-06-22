# Research Program — what makes a world informative for cross-world generalization

Status: design under discussion (2026-06). Companion to `REPORT.md` (results) and
`LOG.md` (run-by-run notebook).

## Unifying thesis

Every task in the benchmark (validity, completion, repair, emptiness) forces a
model to recover the same latent object: the **causal structure** of the world —
which local configurations force which others. We have shown (transparent T4,
forced-cell metric) that a model *can* learn to use given rules and generalize to
unseen worlds, and that this improves with the number of training worlds.

The program below asks the next-order questions:
1. **Coverage** — does generalization decompose over the rule's components?
2. **Influence** — which training worlds drive generalization (data-attribution)?
3. **Complexity** — is a world's informativeness ∝ its causal complexity?
4. **Mechanism** — *how* are the rules used (does the model do propagation)?
5. **Cross-task** — do tasks share the causal-structure representation?

Hypothesis (Gangloff): **a world's informativeness for training is governed by the
complexity of its causal structure**, with the deepest form being the minimal
number of generators of its forcing relation (defined below).

---

## 0. Formalizing causal complexity κ (to be validated)

Configurations are colorings of the grid. A **located pattern** `p` is a partial
assignment (a set of (cell, symbol) pairs). The **forcing (causal) relation**:
`p ⇒ q` iff every admissible configuration that contains `p` also contains `q`.

Closure operations generating new forcing relations from given ones:
- **transitivity/composition**: `p⇒q, q⇒r ⊢ p⇒r`;
- **conjunction**: `p⇒q, p'⇒q' ⊢ (p∧p') ⇒ (q∧q')` (when consistent);
- **shift**: `p⇒q ⊢ (σp) ⇒ (σq)` for any translation σ.

A **generating set** `G` is a set of forcing relations whose closure under the
above equals the world's full forcing relation. **Causal complexity**
`κ(X) = min |G|`.

- Full shift: `κ = 0` (no nontrivial forcing).
- Two-uniform-configs SFT ({all-a, all-b}): `κ = 4` — the horizontal and vertical
  `a⇒a`, `b⇒b` generate everything.

**Connection (makes κ computable).** Ignoring the shift operation, `min |G|` is the
cardinality of the **Duquenne–Guigues canonical (stem) basis** of the closure
system defined by `⇒` — the unique minimum implication base in Formal Concept
Analysis. κ is its **shift-invariant analogue** (generators counted modulo
translation). Exact κ on Z² may be intractable; we use a **bounded-radius proxy
κ_r**: restrict patterns to radius ≤ r, compute the DG basis of the resulting
forcing closure on a window, and quotient by translation. κ_r is monotone in r and
computable for small r,q,N.

*Note on monotonicity:* κ is **not** monotone in entropy. The full shift (max
entropy) has κ=0; rigid worlds can have high κ (two-uniform: κ=4). So "informative =
high κ" and "informative = intermediate entropy" are *different* hypotheses; the
experiments below are designed to discriminate them rather than assume either.

**Related work to cite (provided by author, not yet read):** "epiplexity",
arXiv:2601.03220 — a complexity notion plausibly related to world-complexity vs.
informativeness; to fetch, read, and place in intro/related work.

---

## Experiment C — Coverage (does generalization decompose over rule components?)

**Question.** The rule is a binary vector over the `D = 2q²` possible dominoes.
Does the model generalize to a new world `F'` only on the parts of `F'` built from
dominoes it saw *forbidden* during training?

**Justification.** A model can only learn the causal effect of forbidding domino
`d` if `d` is forbidden (its bit varies, with consequences) somewhere in training.
A domino held at "allowed" throughout training is a constant input feature → its
effect is unlearnable. The forced-cell metric makes the consequence measurable.
*But* the rule-token architecture embeds each domino by its `(a,b,orient)`
structure, so the model *might* instead learn a single general "forbidden ⇒ avoid
this pair" operation that transfers to unseen dominoes. **C distinguishes these**,
which is also a mechanism result (per-domino lookup vs. general operation).

**Design.**
- Partition the `D` dominoes into SEEN `S` and HELD-OUT `U` (e.g. |S|=12, |U|=6).
- Training worlds: forbidden sets drawn **only from `S`** (U-dominoes never
  forbidden in training). Train one rule-conditioned model.
- Test worlds (held-out): **fix the number of forbidden dominoes `k`** (controls
  difficulty), **vary `j` = how many of them lie in `U`** (the author's refinement:
  same |F'|, varying "already-seen" count), `j = 0..k`.
- **Primary metric:** forced-cell accuracy, split into cells whose forcing rule
  involves **only S-dominoes** vs **at least one U-domino**.

**Predictions / readout.**
- *Strict coverage (per-domino learning):* accuracy on U-involving forced cells ≈
  rule-blind/chance; S-only cells stay high; overall OOW falls with `j`.
- *General-operation learning:* U-involving cells also high → coverage violated →
  the model learned a transferable "avoid forbidden pair" operation; "which worlds
  to train on" matters less (only need to exercise the operation, not every domino).
- Either outcome is informative and mechanistic.

**Cost:** cheap (reuses small models + SAT generation with constrained `F`).

---

## Experiment I — Influence / data attribution (which worlds drive generalization?)

**Question.** At fixed budget M, which training worlds most improve OOW? Can we
characterize them?

**Design.**
- *Influence estimate:* leave-one-out (ΔOOW from removing world `w`) is the gold
  standard but O(M) retrains; use cheaper proxies and validate on a subset:
  (a) **add-one-in** ΔOOW over a fixed base set; (b) many random M-subsets,
  regress OOW on per-world indicator (Shapley-style); (c) gradient/representation
  influence (TracIn-style) if needed.
- *Characterize top-influencers:* correlate per-world influence with features:
  `|F|`, #admissible configs (have it), empirical 2-block entropy, rigidity, and
  **κ_r** (Exp 0). Inspect the highest-influence worlds and form a hypothesis.

**Prediction (thesis):** influence rises with κ_r; top influencers are
causally-complex worlds (rich, composable forcing), not the highest-entropy
(near-full-shift, low forcing) nor the most trivially rigid.

**Cost:** moderate (many small trainings); start with random-subset regression.

---

## Experiment D — Causal complexity ↔ informativeness (the deep hypothesis)

**Question.** Is a single world's informativeness a function of κ (and how does it
relate to entropy)?

**Design.**
- Compute κ_r and entropy `H` for a bank of worlds.
- Informativeness of world `w`: train on a base set + `w` (or on `{w}` augmented),
  measure ΔOOW on a fixed held-out test bank. (Single-world training overfits, so
  use base+w influence from Exp I rather than M=1.)
- Plot informativeness vs κ_r and vs H; fit; test monotone-in-κ vs peaked-in-H.

**Prediction:** informativeness monotone increasing in κ_r; non-monotone
(possibly peaked) in entropy, because κ peaks away from the full-shift extreme.

**Cost:** builds on Exp I; main new cost is the κ_r computation (implement DG-basis
of bounded-radius forcing closure).

---

## Experiment M — Mechanism (how are the rules used?)

**Question.** Does the model implement constraint propagation, and which rule
elements does each prediction depend on?

**Probes.**
- **Attention attribution:** for a forced cell, does the model attend (from that
  cell's grid token) to exactly the rule tokens for the dominoes that force it?
- **Rule-token ablation:** zero/scramble individual rule tokens; a correct
  propagator's prediction at a cell should change iff the ablated domino is one
  that forces it.
- **Propagation depth:** accuracy vs. distance from observed cells / vs. required
  propagation chain length (forced-by-1-step vs multi-step). A genuine propagator
  degrades gracefully with chain length; a lookup table fails sharply.
- Connects to the paper's Dennett/IIT claim: *which features a correct world-model
  relies on* becomes directly measurable.

**Cost:** cheap-moderate (analysis on a trained model; no new training).

---

## Experiment X — Cross-task transfer (shared causal structure)

**Question.** Do tasks share one world-model? Does competence on one predict
another, and does a representation trained on one transfer?

**Design.**
- Implement transparent **T3 (repair)** alongside T4 (completion) on the same
  worlds (T3: given rules + grid + a forced cell value, produce a minimal-change
  admissible grid).
- (a) *Correlation:* per-world T3 vs T4 OOW competence — correlated ⇒ shared model.
- (b) *Representation transfer:* freeze the encoder trained on T4, fit a small T3
  head (and vice versa); strong transfer ⇒ a shared causal-structure representation.

**Prediction (thesis):** strong positive correlation and transfer — all tasks are
windows onto the same causal structure.

**Cost:** moderate (new T3 head + training).

---

## Execution plan

Order by value-per-compute and dependency:
1. **Exp C (Coverage)** — cheap, decisive, and doubles as a first mechanism result.
2. **Exp 0 tooling: κ_r computation** — needed for D and I's characterization.
3. **Exp I (Influence)** — random-subset regression first.
4. **Exp D (Complexity ↔ informativeness)** — the deep test, needs 0 + I.
5. **Exp M (Mechanism)** and **Exp X (Cross-task)** — parallelizable analyses/extensions.

Compute discipline unchanged (see `memory/machine-constraints`): CPU, 2 threads,
niced, memory+drive watchdog, small models/worlds, results saved incrementally.
Heavy sweeps → GPU.
