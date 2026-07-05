"""Pure geometry + stats for Q-COMMIT-B — numpy only, no torch/nnsight, so it is unit-testable locally.
The notebook imports these; tests live in test_geometry.py."""
import numpy as np


def d_fr(p, q):
    """Fisher-Rao geodesic distance between two categorical distributions: 2*arccos(sum sqrt(p q))."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    bc = float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))
    return 2.0 * np.arccos(np.clip(bc, 0.0, 1.0))          # clamp guards fp bc slightly > 1


def menger_curv(p0, p1, p2):
    """3-point Menger curvature using Fisher-Rao side lengths (NOT hidden-state turning angle).
    Geodesic-collinear points have additive distances -> Heron area 0 -> curvature 0."""
    a, b, c = d_fr(p0, p1), d_fr(p1, p2), d_fr(p0, p2)
    s = (a + b + c) / 2.0
    area2 = s * (s - a) * (s - b) * (s - c)
    if area2 <= 1e-18:
        return 0.0
    R = (a * b * c) / (4.0 * np.sqrt(area2))
    return 0.0 if R <= 1e-12 else 1.0 / R


def geom_features(probs):
    """probs: list of (vocab,) arrays along the CoT. Returns curvature summary or None if < 3 steps."""
    if len(probs) < 3:
        return None
    dists = [d_fr(probs[t], probs[t + 1]) for t in range(len(probs) - 1)]
    curvs = [menger_curv(probs[t], probs[t + 1], probs[t + 2]) for t in range(len(probs) - 2)]
    if len(curvs):
        curvs = list(np.clip(curvs, 0, np.percentile(curvs, 99)))   # tame degenerate spikes
    path_len = float(np.sum(dists)) + 1e-9
    return dict(mean_curv=float(np.mean(curvs)), max_curv=float(np.max(curvs)),
                curv_per_len=float(np.sum(curvs) / path_len), path_len=path_len)


def entropy_features(probs):
    """Tier-1 baseline: entropy of the SAME per-step distributions (fair, same forward pass)."""
    if len(probs) < 3:
        return None
    H = np.array([float(-np.sum(np.asarray(p, float) * np.log(np.asarray(p, float) + 1e-12))) for p in probs])
    dH = np.diff(H)
    return dict(mean_H=float(H.mean()), slope_H=float(np.polyfit(np.arange(len(H)), H, 1)[0]),
                frac_decreasing=float(np.mean(dH < 0)), final_H=float(H[-1]))


def _rank(x):
    """Average-tie ranks (avoids a scipy dependency)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_diff(g, e, y, seed=42, n=2000):
    """PRIMARY test on PRE-REGISTERED SINGLE features (1 vs 1 -> unbiased by feature count).
    Returns (CI95, mean) of |rho(curv,y)| - |rho(ent,y)|. Falsifier: CI lower bound must exceed 0."""
    g = np.asarray(g, float); e = np.asarray(e, float); y = np.asarray(y, float)
    idx = np.arange(len(y)); rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True); yb = y[b]
        if yb.sum() in (0, len(yb)):
            continue
        diffs.append(abs(spearman(g[b], yb)) - abs(spearman(e[b], yb)))
    diffs = np.array(diffs)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), float(diffs.mean())
