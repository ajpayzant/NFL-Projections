"""Blend weights and outcome distributions, from the audit's out-of-sample predictions.

For each line: the engine / direct-model weight minimising Poisson deviance, chosen walk-forward
(season s uses weights fitted on the seasons before it; 2021 has none and takes 0.5); then the
variance model var = c1 mu + c2 mu^2 fitted the same way, and the calibration of P(over) at the
lines books post and of P(1+) for touchdowns. Prints a table; writes data/models/blend.json with the
weights and variance coefficients fitted on every season (what the live tool uses).
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
P = pl.read_parquet(ROOT / "research" / "audit_engine_pred.parquet")
W_GRID = np.linspace(0, 1, 21)
BOOK_LINES = {"passing_yards": [199.5, 224.5, 249.5, 274.5], "attempts": [29.5, 34.5], "completions": [19.5, 22.5],
              "passing_tds": [0.5, 1.5, 2.5], "interceptions": [0.5], "carries": [9.5, 14.5, 17.5],
              "rushing_yards": [19.5, 39.5, 59.5, 79.5], "receptions": [2.5, 3.5, 4.5, 5.5], "targets": [4.5, 6.5],
              "receiving_yards": [24.5, 39.5, 54.5, 69.5], "receiving_tds": [0.5], "rushing_tds": [0.5]}


def dev(mu, y):
    mu = np.clip(mu, 1e-6, None)
    return float(np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))))


def var_fit(mu, y):
    X = np.column_stack([mu, mu ** 2])
    c = np.linalg.lstsq(X, (y - mu) ** 2, rcond=None)[0]
    return [float(max(c[0], 1.0)), float(c[1])]


def p_over(mu, c, line):
    mu = np.clip(mu, 1e-6, None)
    var = np.maximum(c[0] * mu + c[1] * mu ** 2, mu * 1.0001)
    n = mu ** 2 / (var - mu)
    return stats.nbinom.sf(np.floor(line), n, n / (n + mu))


out, rows = {}, []
for (pos, stat), g in P.group_by(["positions", "stat"]):
    g = g.sort("season")
    seasons = sorted(g["season"].unique().to_list())
    cal = []
    for s in seasons:
        tr, te = g.filter(pl.col("season") < s), g.filter(pl.col("season") == s)
        if tr.height:
            w = min(W_GRID, key=lambda w: dev(w * tr["engine"].to_numpy() + (1 - w) * tr["lgbm"].to_numpy(), tr["truth"].to_numpy()))
            mtr = w * tr["engine"].to_numpy() + (1 - w) * tr["lgbm"].to_numpy()
            c = var_fit(mtr, tr["truth"].to_numpy())
        else:
            w, c = 0.5, None
        mte = w * te["engine"].to_numpy() + (1 - w) * te["lgbm"].to_numpy()
        y = te["truth"].to_numpy()
        r = {"pos": pos, "stat": stat, "season": s, "w_engine": w, "dev_engine": dev(te["engine"].to_numpy(), y),
             "dev_blend": dev(mte, y), "n": te.height}
        if c is not None:
            for ln in BOOK_LINES.get(stat, []):
                p = p_over(mte, c, ln)
                o = (y > ln).astype(float)
                cal.append({"pos": pos, "stat": stat, "line": ln, "pred": p.mean(), "actual": o.mean(),
                            "ll": float(-np.mean(o * np.log(np.clip(p, 1e-6, 1)) + (1 - o) * np.log(np.clip(1 - p, 1e-6, 1)))), "n": len(y)})
        rows.append(r)
    w_all = min(W_GRID, key=lambda w: dev(w * g["engine"].to_numpy() + (1 - w) * g["lgbm"].to_numpy(), g["truth"].to_numpy()))
    m_all = w_all * g["engine"].to_numpy() + (1 - w_all) * g["lgbm"].to_numpy()
    out[f"{pos}|{stat}"] = {"w_engine": float(w_all), "var": var_fit(m_all, g["truth"].to_numpy())}
    if cal:
        C = pl.DataFrame(cal).group_by("pos", "stat", "line").agg((pl.col("pred") * pl.col("n")).sum() / pl.col("n").sum(),
                                                                  (pl.col("actual") * pl.col("n")).sum() / pl.col("n").sum(), pl.col("n").sum())
        out[f"{pos}|{stat}"]["calibration"] = C.sort("line").to_dicts()

R = pl.DataFrame(rows).filter(pl.col("season") > 2021).group_by("pos", "stat").agg(
    pl.col("w_engine").mean(), ((pl.col("dev_engine") - pl.col("dev_blend")) / pl.col("dev_engine") * 100).mean().alias("blend_gain_%"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print("walk-forward blend (2022-2025): mean engine weight and deviance gain vs engine")
print(R.sort("pos", "stat").with_columns(pl.col(pl.Float64).round(3)))
print("\ncalibration of P(over) at book lines, walk-forward 2022-2025 (predicted vs actual):")
for k, v in sorted(out.items()):
    for c in v.get("calibration", []):
        print(f"  {k:24s} o{c['line']:<6g} pred {c['pred']:.3f} actual {c['actual']:.3f}  gap {c['pred'] - c['actual']:+.3f}  n {c['n']}")
(ROOT / "data" / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "models" / "blend.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "calibration"} for k, v in out.items()}, indent=1))
print("\nfinal weights:", {k: round(v["w_engine"], 2) for k, v in sorted(out.items())})
