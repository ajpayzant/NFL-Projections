"""Which outcome distribution prices NFL props best? Walk-forward 2022-2025 on the blended projections.

    nb      negative binomial, var = c1 mu + c2 mu^2
    ecdf    the empirical distribution of y / mu among training rows with a similar projection
            (log mu in 20 quantile bins, the neighbouring bins blended), read at line / mu

Scored at the lines books post: mean gap |predicted - actual| over-rate and log loss.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
P = pl.read_parquet(ROOT / "research" / "audit_engine_pred.parquet")
BLEND = json.loads((ROOT / "data" / "models" / "blend.json").read_text())
import importlib.util
spec = importlib.util.spec_from_file_location("bc", ROOT / "research" / "blend_calibrate.py")
LINES = {"passing_yards": [199.5, 224.5, 249.5, 274.5], "attempts": [29.5, 34.5], "completions": [19.5, 22.5],
         "passing_tds": [0.5, 1.5, 2.5], "interceptions": [0.5], "carries": [9.5, 14.5, 17.5],
         "rushing_yards": [19.5, 39.5, 59.5, 79.5], "receptions": [2.5, 3.5, 4.5, 5.5], "targets": [4.5, 6.5],
         "receiving_yards": [24.5, 39.5, 54.5, 69.5], "receiving_tds": [0.5], "rushing_tds": [0.5]}
BINS = 20


class Ecdf:
    """P(Y > line | mu) from the training rows whose projection is near mu."""

    def __init__(self, mu, y):
        lm = np.log(np.clip(mu, 1e-3, None))
        self.edges = np.quantile(lm, np.linspace(0, 1, BINS + 1))
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(self.edges, lm, side="right") - 1, 0, BINS - 1)
        self.groups = []
        for k in range(BINS):
            sel = (b >= k - 1) & (b <= k + 1)          # neighbours too, for a smoother table
            self.groups.append(np.sort(y[sel] / np.clip(mu[sel], 1e-3, None)))
        self.zero = [np.mean(y[(b >= k - 1) & (b <= k + 1)] == 0) for k in range(BINS)]

    def p_over(self, mu, line):
        lm = np.log(np.clip(mu, 1e-3, None))
        b = np.clip(np.searchsorted(self.edges, lm, side="right") - 1, 0, BINS - 1)
        out = np.empty(len(mu))
        for k in np.unique(b):
            i = b == k
            r = self.groups[k]
            thr = line / np.clip(mu[i], 1e-3, None)
            out[i] = 1 - np.searchsorted(r, thr, side="right") / len(r)
        return np.clip(out, 1e-4, 1 - 1e-4)


def nb_over(mu, c, line):
    mu = np.clip(mu, 1e-6, None)
    var = np.maximum(c[0] * mu + c[1] * mu ** 2, mu * 1.0001)
    n = mu ** 2 / (var - mu)
    return stats.nbinom.sf(np.floor(line), n, n / (n + mu))


def ll(p, o):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


rows = []
for (pos, stat), g in P.group_by(["positions", "stat"]):
    w = BLEND[f"{pos}|{stat}"]["w_engine"]
    g = g.with_columns((w * pl.col("engine") + (1 - w) * pl.col("lgbm")).alias("mu"))
    for s in (2022, 2023, 2024, 2025):
        tr, te = g.filter(pl.col("season") < s), g.filter(pl.col("season") == s)
        mtr, ytr, mte, yte = tr["mu"].to_numpy(), tr["truth"].to_numpy(), te["mu"].to_numpy(), te["truth"].to_numpy()
        X = np.column_stack([mtr, mtr ** 2]); c = np.linalg.lstsq(X, (ytr - mtr) ** 2, rcond=None)[0]; c = [max(c[0], 1.0), c[1]]
        E = Ecdf(mtr, ytr)
        for line in LINES.get(stat, []):
            o = (yte > line).astype(float)
            pn, pe = nb_over(mte, c, line), E.p_over(mte, line)
            rows.append({"pos": pos, "stat": stat, "line": line, "season": s, "n": len(o), "actual": o.mean(),
                         "nb": pn.mean(), "ecdf": pe.mean(), "ll_nb": ll(pn, o), "ll_ecdf": ll(pe, o)})
R = pl.DataFrame(rows)
wm = lambda c: (pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()
S = R.group_by("pos", "stat", "line").agg(wm("actual"), wm("nb"), wm("ecdf"), wm("ll_nb"), wm("ll_ecdf"), pl.col("n").sum()).sort("pos", "stat", "line")
S = S.with_columns((pl.col("nb") - pl.col("actual")).alias("gap_nb"), (pl.col("ecdf") - pl.col("actual")).alias("gap_ecdf"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(S.select("pos", "stat", "line", "actual", "gap_nb", "gap_ecdf", "ll_nb", "ll_ecdf").with_columns(pl.col(pl.Float64).round(4)))
print("mean |gap|: nb", round(S["gap_nb"].abs().mean(), 4), " ecdf", round(S["gap_ecdf"].abs().mean(), 4),
      "| log loss better with ecdf on", int((S["ll_ecdf"] < S["ll_nb"]).sum()), "of", S.height, "lines")
