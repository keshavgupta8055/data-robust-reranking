# Joint bootstrap over queries AND judge seeds. Run on CPU after runs A, B, C (any machine, no GPU needed):
#   python merge_seeds.py perq_j1.5B_s0.npz perq_j1.5B_s1.npz perq_j1.5B_s2.npz
# For every (re-ranker, scenario) it averages the per-query correctness over the seeds, then bootstraps the QUERIES
# (same resampled queries for every seed), so the interval reflects test-set size, and it also reports the spread over seeds.
import sys, numpy as np, pandas as pd
files = sys.argv[1:]; assert len(files) >= 2, "give at least two perq_*.npz files"
Z = [np.load(f, allow_pickle=True) for f in files]
t0 = Z[0]["_test_titles"]
for z in Z[1:]: assert (z["_test_titles"] == t0).all(), "different test sets: runs must use the same N_QUERIES / HOLD10 / N_DISTR / PASS_SET"
keys = sorted(k for k in Z[0].files if not k.startswith("_") and all(k in z.files for z in Z))
rng = np.random.default_rng(0); n = len(t0); B = 5000; idx = rng.integers(0, n, (B, n))
mean_vec = {k: np.mean([z[k] for z in Z], 0) for k in keys}
seed_means = {k: np.array([z[k].mean() for z in Z]) for k in keys}
def ci(v): m = v[idx].mean(1); return 100 * v.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)
rows = []; groups = sorted({tuple(k.split("@")[:2]) for k in keys})
refs = ["metadata_only", "metadata_thresh", "decay", "content_ft", "nogate_ft", "gated_zs"]
for rn, sc in groups:
    g = f"{rn}@{sc}@gated_ft"
    for fam in ["gated_ft"] + refs + ["date_free"]:
        k = f"{rn}@{sc}@{fam}"
        if k not in mean_vec: continue
        m, lo, hi = ci(mean_vec[k]); row = dict(reranker=rn, scenario=sc, family=fam, mean=round(m, 1), ci=f"[{lo:.1f},{hi:.1f}]", seed_sd=round(100 * seed_means[k].std(ddof=1), 2), seed_min=round(100 * seed_means[k].min(), 1), seed_max=round(100 * seed_means[k].max(), 1))
        if fam != "gated_ft" and g in mean_vec:
            d, dlo, dhi = ci(mean_vec[g] - mean_vec[k]); row["gated_ft_minus_this"] = f"{d:+.1f} [{dlo:+.1f},{dhi:+.1f}]"
        rows.append(row)
df = pd.DataFrame(rows); df.to_csv("pooled_seed_intervals.csv", index=False); print(df.to_string(index=False)); print("saved pooled_seed_intervals.csv")
