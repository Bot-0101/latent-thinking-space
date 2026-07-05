# Q-COMMIT-B — experiment dossier (living, append-only)

**READ ME FIRST.** This is the single place to understand this experiment cold. Everything ABOVE the `=== RESULTS ===` divider is **frozen pre-registration** — written before we saw any data, never edited afterwards. Results + interpretation are **appended below, dated.** That boundary is what keeps "what we predicted" honest against "what we found." For the precise spec, see `problem-statement.md`; for the (unit-tested) math, `geometry.py` / `test_geometry.py`.

---

## TL;DR
When a small model reasons step by step, its guess about the next word traces a *path*. We ask one thing: **does the *bendiness* of that path tell us whether the answer will be right — more than the model's plain uncertainty already does?** Yes → the geometry of "thinking" carries real signal. No → uncertainty was the whole story (a clean negative).

## The big picture — why this experiment exists
The grand goal ("map the model's thinking space") is too vague to test. This is **one concrete, falsifiable tile** of that map. As the model generates a chain-of-thought, at every step it holds a probability distribution over the next token — essentially "what it's leaning towards saying next." Stack those distributions in order and you get a **trajectory**: a path the model's belief walks as it reasons. We're asking whether the *shape* of that path (specifically how sharply it turns) is meaningful, or just noise on top of a simpler quantity (uncertainty).

## Objective (plain)
Test whether **curvature of the belief-path predicts answer correctness beyond an entropy (uncertainty) baseline** on Qwen2.5-1.5B, GSM8K math problems.

## The hypothesis, in plain words + the intuition
**Hypothesis:** the path bends *differently* on problems the model gets right vs wrong, in a way that plain uncertainty misses.

**Intuition:** imagine the model "homing in" on an answer. If it reasons cleanly, the path might sweep smoothly toward a conclusion (low curvature). If it hesitates, second-guesses, or switches strategy mid-stream, the path makes sharp turns (high curvature). *Maybe* sharp turns flag shaky reasoning → wrong answers. That's the bet. It might be false — hesitation could equally signal careful correction. Either way we learn something.

## What we measure, and why each piece is there
- **Belief trajectory:** the raw next-token probability distribution at each generated step (captured directly from the model — verified nnsight API).
- **Fisher–Rao distance** between consecutive steps = "how much the model's leaning *changed* from one step to the next" (the natural distance between probability distributions).
- **Menger curvature** = "how sharply the path *turns*" at each point (from three consecutive steps). This is the star quantity. Note: it's curvature of the *output distributions*, deliberately different from prior work that measured curvature of internal hidden states.
- **Entropy** at each step = "how *unsure* the model is" right there. This is the **baseline** — the thing curvature must beat.
- **Why the baseline is non-negotiable:** curvature and uncertainty are related (an unsure model may also wobble). If curvature only predicts correctness *because* it's secretly tracking uncertainty, it tells us nothing new. So the real question isn't "does curvature predict correctness?" — it's **"does curvature predict correctness *beyond what entropy already explains*?"** That's the falsifier.

## The single pre-registered test (unbiased)
- Curvature summary = `curv_per_len` (total turning, normalized by path length so it isn't just "longer path = more turning").
- Entropy summary = `mean_H` (average uncertainty).
- Compute each one's Spearman correlation with correctness, then **bootstrap the difference** |ρ_curv| − |ρ_ent|.
- **FALSIFIER:** curvature only "wins" if the bootstrap 95% CI of that difference is **entirely above 0.** (We use *one* scalar per side, chosen in advance, so nobody can cherry-pick the best of many features after the fact.)

## Method in plain steps
1. **Sanity first** — print the chat template + a tokenization; confirm the format is right (a silent format bug ruins everything).
2. Run 100 GSM8K problems; capture the per-step next-token distributions of each chain-of-thought.
3. Compute curvature features + entropy features from those distributions.
4. Label each answer right/wrong.
5. **Primary test** (above) → the falsifier.
6. **Tier-2 check** on 30 problems: also beat the *real published* entropy baseline (2603.18940), not just our cheap one.

## What each result MEANS + how it changes direction
| Outcome | What it means (plain) | What we do next |
|---|---|---|
| **Curvature wins** (CI entirely > 0) | The path's shape carries correctness-signal uncertainty misses. A real, promotable finding. | Deepen: (a) rule out the length confound hard, (b) test *causally* (perturb curvature, does correctness move?), (c) scale to 7B, (d) chase the mechanism. |
| **Null** (CI straddles 0) | Curvature adds nothing beyond uncertainty. This tile of the map is flat (for this metric/model/data). | Log the **rigorous negative** honestly, update the map, pivot — a different geometry, or off-frontier (`negative-space`/`detok-converge`). |
| **Entropy clearly wins** | Plain uncertainty is the better simple predictor; geometry is redundant. | Same pivot as null, with a cleaner "uncertainty dominates" story. |
| **Beats cheap entropy but NOT the published baseline (Tier-2)** | Weak, partial — beats a naive baseline but not the strong one. | Not enough for a novelty claim; needs a stronger feature or more data before promoting. |
| **Accuracy ~0 or ~100%** | No variance in correctness → nothing to predict; result is *inconclusive, not negative*. | Re-pick difficulty (harder/easier problems) or more questions, rerun. |
| **Too few usable CoTs (<~30)** | Underpowered; CI will be too wide to conclude. | Expand n before believing anything. |

## How this could fool us (what `/analyze` will hunt)
- **Length confound (biggest risk):** longer CoTs accumulate more turning *and* may track problem difficulty/correctness. We normalize by path length to fight this, but `/analyze` must check curvature isn't just proxying CoT length/step-count. Check `corr(path_len, correct)` directly.
- **Answer-parsing errors:** our regex could mislabel right/wrong → fake signal or noise. Eyeball random examples.
- **Small-n / wide CI:** ~100 problems, maybe ~60–80 usable; a borderline CI is inconclusive, not a result.
- **Degenerate curvature spikes:** near-identical low-entropy steps blow up Menger curvature; we clip at the 99th percentile — check the distribution isn't dominated by clipping.
- **Greedy decoding:** the primary path is greedy (one deterministic trajectory); fine, but note Tier-2 uses sampling.

## "How could this be false?" — 5-minute checks (run before believing a positive)
1. **Permutation null:** shuffle the correctness labels → the curvature ρ should collapse to ≈0. If it doesn't, the pipeline is leaking.
2. **Length check:** if `curv_per_len` correlates with correctness but so does raw `path_len` equally, suspect a length artifact.
3. **Read 8 random CoTs:** do the high-curvature ones actually *look* more hesitant/wrong? If not, be very suspicious.

## Pointers
- Precise pre-registration + amendments (two-tier baseline; single-feature fix): `problem-statement.md`
- Config (model/library/seed): `experiment.config.md`
- Math, unit-tested: `geometry.py`, `test_geometry.py` (21/21 pass)
- Parent research directions this descends from: `research-loop/directions/001/`, `research-loop/directions/x001/`
- Novelty framing / what to cite-and-differentiate: `problem-statement.md` (GeoFaith 2605.26893, 2603.18940, 2606.09287, 2605.22007)

=== RESULTS === (appended after the Colab run — NOTHING above this line is edited post-data)

## Results (date: ____)
_empty until the run returns. Paste summary.json numbers + which outcome-row fired._

## Interpretation (date: ____)
_empty. Written after `/analyze`, not before._

## Decision (date: ____)
_empty. Deepen / pivot / drop — with the reason._
