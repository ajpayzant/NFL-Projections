"""P(1+ TD): the empirical table (plateaus at high projections) vs a logistic-calibrated Poisson,
walk-forward 2022-2025 on the blended out-of-sample projections. Also by projection decile at the top."""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dist
ROOT = Path(__file__).resolve().parents[1]
P = pl.read_parquet(ROOT / "research" / "audit_engine_pred.parquet")
B = json.loads((ROOT / "data" / "models" / "blend.json").read_text())
P = P.with_columns(pl.struct("positions", "stat").map_elements(lambda r: B[f"{r['positions']}|{r['stat']}"]["w_engine"], return_dtype=pl.Float64).alias("w"))
P = P.with_columns((pl.col("w") * pl.col("engine") + (1 - pl.col("w")) * pl.col("lgbm")).alias("mu"))
mu = P.pivot(on="stat", index=["positions", "season", "game_id", "player_id"], values="mu")
y = P.pivot(on="stat", index=["positions", "season", "game_id", "player_id"], values="truth")
J = mu.join(y, on=["positions", "season", "game_id", "player_id"], suffix="_y")
def any_td(df, sfx=""):
    cols = [c for c in (f"rushing_tds{sfx}", f"receiving_tds{sfx}") if c in df.columns]
    return df.select(pl.sum_horizontal([pl.col(c).fill_null(0) for c in cols])).to_series().to_numpy()
def ll(p, o): p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))
for grp in ("QB", "RB", "WR+TE"):
    g = J.filter(pl.col("positions") == grp)
    m, o, s = any_td(g), (any_td(g, "_y") >= 1).astype(float), g["season"].to_numpy()
    res = {"empirical": [], "poisson": [], "calibrated": []}; tops = []
    for t in (2022, 2023, 2024, 2025):
        tr, te = s < t, s == t
        E = dist.Empirical(m[tr], (any_td(g.filter(pl.Series(tr)), "_y")).astype(float))
        c = dist.fit_td_calibration(m[tr], o[tr])
        pe, pp, pc = E.p_over(m[te], 0.5), 1 - np.exp(-m[te]), dist.td_probability(c, m[te])
        for k, p in (("empirical", pe), ("poisson", pp), ("calibrated", pc)): res[k].append((p, o[te]))
        hi = m[te] >= np.quantile(m[te], 0.9)
        tops.append((pe[hi].mean(), pp[hi].mean(), pc[hi].mean(), o[te][hi].mean()))
    print(grp, {k: round(float(np.mean([ll(p, q) for p, q in v])), 4) for k, v in res.items()},
          "| top decile pred emp/poisson/cal vs actual:", np.round(np.mean(tops, axis=0), 3))
