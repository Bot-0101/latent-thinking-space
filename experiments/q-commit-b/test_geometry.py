"""Local dry-run: validate the pure geometry/stats on SYNTHETIC data (no model/GPU needed)."""
import numpy as np
from geometry import d_fr, menger_curv, geom_features, entropy_features, spearman, bootstrap_diff

fails = []
def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        fails.append(name)
def approx(a, b, t=1e-6):
    return abs(a - b) <= t

# --- Fisher-Rao distance ---
p = np.array([0.2, 0.3, 0.5]); q = np.array([0.5, 0.4, 0.1])
oh0 = np.array([1., 0, 0]); oh1 = np.array([0, 1., 0]); u5 = np.ones(5) / 5
check("d_fr(p,p) == 0", approx(d_fr(p, p), 0, 1e-9))
check("d_fr(disjoint one-hots) == pi", approx(d_fr(oh0, oh1), np.pi, 1e-9))
check("d_fr symmetric", approx(d_fr(p, q), d_fr(q, p), 1e-12))
check("d_fr in [0, pi]", 0 <= d_fr(p, q) <= np.pi + 1e-12)
check("d_fr(uniform,onehot) in (0,pi)", 0 < d_fr(u5, np.eye(5)[0]) < np.pi)
check("d_fr no NaN on identical", not np.isnan(d_fr(u5, u5)))

# --- Menger curvature ---
def geodesic(a, b, t):  # great-circle interpolation via sqrt-parameterisation
    sa, sb = np.sqrt(a), np.sqrt(b)
    om = np.arccos(np.clip(np.sum(sa * sb), -1, 1))
    if om < 1e-9:
        return np.asarray(a, float)
    return ((np.sin((1 - t) * om) * sa + np.sin(t * om) * sb) / np.sin(om)) ** 2
g0, g1, g2 = geodesic(oh0, oh1, 0.2), geodesic(oh0, oh1, 0.5), geodesic(oh0, oh1, 0.8)
sharp = menger_curv([0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9])
gentle = menger_curv([0.6, 0.2, 0.2], [0.5, 0.25, 0.25], [0.4, 0.3, 0.3])
check("menger(identical) == 0", approx(menger_curv(p, p, p), 0))
check("menger non-negative", sharp >= 0 and gentle >= 0)
check("menger finite (no NaN/inf)", np.isfinite(sharp) and np.isfinite(gentle))
check("menger(geodesic-collinear) ~ 0", menger_curv(g0, g1, g2) < 1e-6)
check("menger sharp-turn > gentle-turn", sharp > gentle)

# --- entropy features ---
check("entropy(uniform) ~ ln5", approx(entropy_features([u5] * 4)["mean_H"], np.log(5), 1e-6))
decr = [u5, np.array([0.4, 0.3, 0.15, 0.1, 0.05]), np.array([0.7, 0.15, 0.1, 0.03, 0.02]),
        np.array([0.95, 0.02, 0.01, 0.01, 0.01])]
ef = entropy_features(decr)
check("entropy slope < 0 for decreasing seq", ef["slope_H"] < 0)
check("entropy frac_decreasing high", ef["frac_decreasing"] >= 0.9)

# --- geom_features on a synthetic trajectory ---
gf = geom_features([np.eye(5)[i % 5] * 0.8 + 0.05 for i in range(6)])
check("geom_features returns keys", gf is not None and "curv_per_len" in gf)
check("geom_features < 3 steps -> None", geom_features([u5, u5]) is None)

# --- spearman ---
check("spearman monotonic == 1", approx(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, 1e-9))
check("spearman reversed == -1", approx(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, 1e-9))
check("spearman constant -> 0", approx(spearman([1, 2, 3, 4], [5, 5, 5, 5]), 0.0))

# --- PRIMARY test end-to-end (single pre-registered feature each: unbiased) ---
rng = np.random.default_rng(0); n = 80
y = rng.integers(0, 2, n).astype(float)
g_corr = y * 2 + rng.normal(0, 1, n)          # curvature-analog correlated with correctness
e_noise = rng.normal(0, 1, n)                 # entropy-analog uncorrelated
(lo, hi), m = bootstrap_diff(g_corr, e_noise, y, seed=42, n=1500)
check("primary detects curvature-wins (CI lower > 0)", lo > 0)
g_noise = rng.normal(0, 1, n)                 # NULL: both uninformative
(lo0, hi0), m0 = bootstrap_diff(g_noise, e_noise, y, seed=42, n=1500)
check("primary NULL: CI includes 0 (unbiased)", lo0 <= 0 <= hi0)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
