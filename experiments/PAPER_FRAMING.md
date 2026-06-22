# Paper Framing (planning scaffold — NOT the paper; do not edit the paper repo)

Draft language and structure for reframing *Worlds as Local Rules Systems* around
the causal-structure thesis. Lift into the paper as desired.

---

## The story (the spine)

A central puzzle of intelligence is understanding an unfamiliar world from the
outside: inferring the laws that govern it and reasoning about their consequences.
ARC-style benchmarks chase this but are laced with human visual convention. We
strip that away by taking worlds to be **finite windows of two-dimensional
subshifts of finite type (SFTs)**: grids over a finite alphabet that avoid a
finite set of forbidden local patterns. The forbidden patterns are the world's
physics; nothing else is imposed.

What does it *mean* to understand such a world? Our answer — the **thesis** of the
paper — is that understanding a world is **recovering its causal structure**: the
relation of local forcing that determines which configurations compel which
others. Every act of inquiry we can pose about a world — Is this state valid?
Complete this partial state. Repair this intervention. Could such a world be
non-empty? — is a *window onto that one structure*. To understand the world is to
hold the structure that all these questions secretly share.

This reframes "world-model inference" as something formal and measurable. The
causal structure is a mathematical object with a complexity, and the tasks are its
shadows. The rest of the paper makes this precise and tests it.

## The thesis, made precise

The **causal (forcing) relation**: a located pattern `p` forces `q`, written
`p ⇒ q`, if every admissible configuration containing `p` also contains `q`. Closed
under composition, conjunction, and shift, these relations form the world's causal
structure. Its **complexity** `κ` is the minimal number of generators — a
**shift-invariant analogue of the Duquenne–Guigues canonical implication base**
from Formal Concept Analysis (so κ is, in principle, computable). κ(full shift)=0;
κ(two-uniform-configs)=4. (See `RESEARCH_PROGRAM.md` §0 for the full definition and
the bounded-radius proxy κ_r used in experiments.)

Two readings of the same structure, which is why the tasks cohere:
- **observational** forcing → *completion* (what observations compel);
- **interventional** forcing → *repair* (how a forced change propagates).

## The refutable law (the apex)

If understanding is recovering causal structure, then **how much a world can teach
a learner should be governed by how much causal structure it has** — i.e. by κ.
We state this as a sharp, falsifiable law:

> **Informativeness Law.** A world's usefulness as training data for cross-world
> generalization is monotone in its causal complexity κ.

The law is meaningful *because of* the thesis: it predicts that the quantity which
governs learning is an intrinsic invariant of the world's causal structure, not an
incidental statistic. It is falsifiable — κ might simply fail to correlate — which
is the point.

## How we keep ourselves honest (methodological contribution)

Local statistics can *mimic* causal forcing (a model can guess a forced cell from
neighbour frequencies without using the rule). So the paper must — and does —
**separate genuine causal understanding from statistical shortcut**:
- the **forced-cell** metric (cells whose value is determined by the rules);
- the **mismatch control** (give the wrong rules: a true user is misled);
- the **coverage** split (rule components never exercised in training).
This is a reusable recipe for measuring structural understanding without being
fooled by correlation.

## Draft abstract

> We propose a formal, culturally neutral framework for studying a core facet of
> general intelligence: understanding an unfamiliar world. A world is a finite
> window of a two-dimensional subshift of finite type — configurations over a
> finite alphabet avoiding a finite set of forbidden local patterns — together with
> a suite of tasks (validity, completion, repair, emptiness) probing an agent's
> grasp of it. Our central thesis is that **understanding a world is recovering its
> causal structure**: the relation of local forcing that determines which
> configurations compel which others. We make this precise via κ, a shift-invariant
> analogue of the Duquenne–Guigues canonical implication base, and argue that all
> tasks are windows onto this single object. We test the thesis three ways:
> (i) **out-of-world generalization** — a model that recovered the causal structure
> applies it to unseen worlds; (ii) **cross-task transfer** — competence transfers
> across tasks that share the structure; and (iii) a **refutable law** — a world's
> usefulness as training data is governed by its causal complexity κ. We also give
> metrics (forced-cell accuracy, a rule-mismatch control, a coverage split) that
> separate genuine structural understanding from statistical shortcut. [Results
> sentence.] By grounding "understanding" in a measurable formal invariant, we
> obtain a difficulty-tunable substrate for world-model inference and a sharp,
> falsifiable account of what such inference recovers.

## Contributions

1. **Framework.** Worlds as SFT windows; a generator parameterized by (q, ρ, N);
   a task suite — a culture-free, difficulty-tunable AGI substrate. *(original)*
2. **Thesis.** Understanding a world = recovering its causal structure, formalized
   as the forcing relation and its complexity κ (shift-invariant DG base). *(new)*
3. **Methodology.** Metrics that separate causal understanding from statistical
   shortcut (forced-cell, mismatch, coverage). *(new)*
4. **Evidence.** Three tests of the thesis: OOW generalization *(established)*,
   cross-task transfer, and the κ-informativeness law. *(new)*

## Claims → experiments → status

| Claim | Experiment | Status |
|---|---|---|
| Causal structure is the transferable object | OOW forced-cell + mismatch (T4) | **established** (REPORT §6) |
| Metrics separate understanding from statistics | rule-blind vs mismatch vs forced-cell | **established** (REPORT §6.3) |
| Generalization decomposes over rule components; per-domino vs general operation | Coverage (Exp C) | to run |
| Causal structure is the common substrate of all tasks | Cross-task transfer T3↔T4 (Exp X) | to run |
| Informativeness ∝ κ (the refutable law) | Influence (Exp I) + κ↔informativeness (Exp D) | to run |

## Honest logical status

The three pillars **corroborate** the thesis; they do not prove it. The
κ-informativeness law is the sharpest, most falsifiable element and should be
presented as such — a refutable prediction the thesis motivates, not an assumption.

## Related work to integrate

- Lind & Marcus (symbolic dynamics); Formal Concept Analysis / Duquenne–Guigues
  basis (for κ); the existing ARC / universal-intelligence framing.
- arXiv:2601.03220 "epiplexity" — read and connect to κ / world-complexity (see
  `memory/epiplexity-paper-ref`).
