# problem-statement — q-commit-b (PRE-REGISTERED)

## One line
Does the **curvature of a reasoning model's raw next-token output-distribution trajectory** predict answer correctness **beyond a scalar-entropy baseline** — on Qwen2.5-1.5B?

## Why (maps onto the thinking-space goal)
"Thinking space" is vague; this is one sharp, measurable tile of it: as the model reasons, its per-step output distribution traces a path on the categorical simplex. If the *bendiness* (curvature) of that path carries information about whether the reasoning is going to be right — information that raw uncertainty (entropy) does not — then the *geometry* of the belief path is a real, readable structure. If not, geometry is decorative and entropy is the whole story. Either answer advances the map.

## Hypothesis
The Fisher–Rao/Menger **curvature** of the raw next-token output-distribution trajectory over CoT positions is a **non-redundant** predictor of final-answer correctness — it beats the published entropy-trajectory baseline.

## Falsifier (single, clean — Signal A dropped so no AND-clause escape hatch)
Curvature-feature Spearman ρ with correctness does **NOT** exceed the entropy-trajectory-monotonicity ρ of **arXiv 2603.18940**, i.e. the **bootstrap 95% CI of (ρ_curvature − ρ_entropy) includes 0** → geometry adds nothing beyond entropy → **drop** (logged as a rigorous negative).

## Method
- Subject: `Qwen/Qwen2.5-1.5B-Instruct` via nnsight. Data: **GSM8K first 100 test**, seed 42.
- **Slice 1 = tokenizer-first check** (print chat-template + sample tokenization; locate first generated token).
- Generate CoT per question; capture the **raw per-step next-token softmax** pₜ over the full vocab.
- Fisher–Rao geodesic step distance `d = 2·arccos(Σ√(pₜ pₜ₊₁))` (clamp Bhattacharyya before arccos).
- **Menger curvature** = 3-point circumradius on the sphere (NOT 2-vector turning angle — that's 2606.09287's; distinguish explicitly).
- Correctness label per question (parsed final answer vs gold).

## Baseline (REAL, not strawman)
**arXiv 2603.18940** entropy-trajectory-monotonicity metric (their entropy-shape predictor: monotone 68.8% vs non-monotone 46.8% on GSM8K). Curvature must beat *this specific* published baseline.

## Ablation
curvature-only vs entropy-only vs joint; with/without arc-length normalisation; per-position vs aggregate.

## Novelty framing (honest — narrow combination-novelty)
Must **cite and out-perform**, not resemble: GeoFaith 2605.26893 (Fisher–Rao on VAE latents, no curvature), 2606.09287 (turning-angle curvature on hidden states, same model), 2605.22007 (commitment = distribution collapse — reuse their definition), 2603.18940 (the baseline), ENIGMA 2510.11278 (Fisher–Rao-on-token-dists as theory).

## Gates
- **Pre-read before ANY compute:** 2603.18940 (baseline) + 2605.22007 (confirm no pre-empt).
- **STOP for human approval before running compute** (per research-loop rule).

## AMENDMENT 2026-07-05 (approved) — two-tier baseline
Pre-read revealed 2603.18940's baseline is NOT a cheap same-forward-pass entropy: it samples **m=5 completions-to-the-end per CoT step** (τ=0.7), takes entropy over the empirical **final-answer** distribution, then a **binary** monotonicity classifier (ε=0.01; any step with H_{k+1} > H_k + ε → non-monotone), reported as an accuracy GAP (monotone 68.8% vs 46.8%). Different quantity, expensive, and binary (not a ρ). Amended design:
- **Tier-1 (PRIMARY falsifier, cheap):** entropy of the SAME per-step next-token distributions we compute curvature on (free from the same forward pass). Falsifier = curvature ρ does not beat this entropy ρ (bootstrap 95% CI of the difference includes 0) → drop.
- **Tier-2 (published-baseline robustness, expensive):** implement 2603.18940's answer-entropy monotonicity FAITHFULLY on a **30-question subset**; confirm curvature also beats it (accuracy-gap / AUROC framing).
- Pre-empt check CLEAR: 2605.22007 has no geometry/curvature and no geometry-vs-entropy race. Citation fix: its "commitment" = mass concentrated on a **surface form** (not "a single token"). Also cite-and-differentiate **2604.15400** (attractor dynamics on Qwen2.5-1.5B — different construct).

## AMENDMENT 2026-07-05b (dry-run) — pre-registered single primary features
Local geometry dry-run (21/21 tests pass) caught a **column-count bias**: comparing "best |ρ| over k columns" is unfair when curvature has 3 features and entropy has 4 (more columns → higher max by chance), and it re-opens cherry-picking. **Pre-registered primary scalars (1 vs 1, unbiased):** curvature = **`curv_per_len`** (arc-length-normalised total Menger curvature); entropy = **`mean_H`**. Falsifier unchanged: bootstrap 95% CI of |ρ_curv|−|ρ_ent| must exclude 0. All other features are EXPLORATORY and cannot rescue a null primary. Pure functions live in `geometry.py` (tested by `test_geometry.py`).
