"""Score audit_engine.py's saved predictions with mean-sensitive metrics.

MAE is minimised by the median, so on skewed lines (touchdowns, yards) a model that projects low
wins it without forecasting better. Here: RMSE, Poisson deviance and bias, plus for the lines books
price as 'anytime' (touchdowns) the log loss of P(1+) under a Poisson around each projection.
"""
from pathlib import Path

import numpy as np
import polars as pl

P = pl.read_parquet(Path(__file__).with_name("audit_engine_pred.parquet"))


def dev(mu, y):
    mu = np.clip(mu, 1e-6, None)
    return np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu)))


def ll1(mu, y):
    p = np.clip(1 - np.exp(-np.clip(mu, 1e-6, None)), 1e-6, 1 - 1e-6)
    o = (y >= 1).astype(float)
    return -np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))


rows = []
for (pos, stat), g in P.group_by(["positions", "stat"]):
    y = g["truth"].to_numpy().astype(float)
    r = {"positions": pos, "stat": stat, "n": len(y), "mean": y.mean()}
    for m in ("engine", "lgbm"):
        mu = g[m].to_numpy()
        r[f"rmse_{m}"] = np.sqrt(np.mean((mu - y) ** 2))
        r[f"dev_{m}"] = dev(mu, y)
        r[f"bias_{m}"] = (mu - y).mean()
    b = (g["engine"].to_numpy() + g["lgbm"].to_numpy()) / 2
    r["rmse_blend"], r["dev_blend"] = np.sqrt(np.mean((b - y) ** 2)), dev(b, y)
    if stat.endswith("_tds"):
        r["ll1_engine"], r["ll1_lgbm"], r["ll1_blend"] = ll1(g["engine"].to_numpy(), y), ll1(g["lgbm"].to_numpy(), y), ll1(b, y)
    rows.append(r)
R = pl.DataFrame(rows).sort("positions", "stat")
R = R.with_columns(((pl.col("dev_engine") - pl.col("dev_lgbm")) / pl.col("dev_engine") * 100).alias("lgbm_dev_gain_%"),
                   ((pl.col("dev_engine") - pl.col("dev_blend")) / pl.col("dev_engine") * 100).alias("blend_dev_gain_%"),
                   ((pl.col("rmse_engine") - pl.col("rmse_blend")) / pl.col("rmse_engine") * 100).alias("blend_rmse_gain_%"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
print(R.select("positions", "stat", "mean", "rmse_engine", "rmse_lgbm", "rmse_blend", "bias_engine", "bias_lgbm",
               "lgbm_dev_gain_%", "blend_dev_gain_%", "blend_rmse_gain_%").with_columns(pl.col(pl.Float64).round(3)))
print(R.filter(pl.col("stat").str.ends_with("_tds")).select("positions", "stat", "ll1_engine", "ll1_lgbm", "ll1_blend")
      .with_columns(pl.col(pl.Float64).round(4)))
