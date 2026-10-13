"""Audit: is ~/nfl-projection-system's chain the best player model available to the NFL tool?

    cd ~/nfl-projection-system && PYTHONUTF8=1 .venv/Scripts/python <this file>

The engine's walk-forward replay (data/projections/backtest_replay.parquet, seasonal cadence,
2021-2025) is scored against a direct challenger: one LightGBM per position and statistic, fitted
on the same pregame rows the engine trains on (src.features.store.load_training_frame, every
non-target numeric column, market context included) from 2016 through the season before the test
season, and scored on exactly the replay's player-games and truth values. A 50/50 blend of the two
is scored as well. Output: research/audit_engine_out.csv and a table on stdout.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ENGINE = Path.home() / "nfl-projection-system"
sys.path.insert(0, str(ENGINE))
from src.features import store  # noqa: E402

OUT = Path(__file__).with_name("audit_engine_out.csv")
PRED = Path(__file__).with_name("audit_engine_pred.parquet")
TEST = [2021, 2022, 2023, 2024, 2025]
FIRST = 2016
# (frame kind, positions, replay population that owns the line, statistic)
LINES = [("qb", ("QB",), "quarterback", s) for s in
         ("attempts", "completions", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards",
          "rushing_tds")] + \
        [("skill", ("RB",), "running_back", s) for s in
         ("carries", "rushing_yards", "targets", "receptions", "receiving_yards", "receiving_tds")] + \
        [("skill", ("RB",), "touchdown", "rushing_tds")] + \
        [("skill", ("WR", "TE"), "receiver", s) for s in ("targets", "receptions", "receiving_yards", "receiving_tds")]
YARDS = {"passing_yards", "rushing_yards", "receiving_yards"}
PARAMS = {"learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.3,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "num_threads": 12}
ROUNDS = 500

frames: dict[str, pl.DataFrame] = {}


def frame(kind: str) -> pl.DataFrame:
    if kind not in frames:
        frames[kind] = store.load_training_frame(kind, list(range(FIRST, max(TEST) + 1)), played_only=True)
    return frames[kind]


def features(f: pl.DataFrame) -> list[str]:
    return [c for c, t in f.schema.items() if t.is_numeric() and not c.startswith("y_") and "_y_" not in c
            and c not in ("season",)]


replay = pl.read_parquet(ENGINE / "data/projections/backtest_replay.parquet").filter(pl.col("cadence") == "seasonal")
rows, preds = [], []
t0 = time.time()
for kind, pos, popn, stat in LINES:
    f = frame(kind).filter(pl.col("projected_position").is_in(list(pos)))
    X_cols = features(f)
    rp = replay.filter((pl.col("population") == popn) & pl.col("projected_position").is_in(list(pos))).select(
        "game_id", "player_id", "season", pl.col(f"p_{stat}_chain").alias("engine"), pl.col(f"y_{stat}").alias("truth"))
    rp = rp.drop_nulls(["engine", "truth"])
    obj = {"objective": "tweedie", "tweedie_variance_power": 1.2} if stat in YARDS else {"objective": "poisson"}
    for s in TEST:
        tr = f.filter((pl.col("season") < s) & pl.col(f"y_{stat}").is_not_null())
        te = f.join(rp.filter(pl.col("season") == s).select("game_id", "player_id", "engine", "truth"),
                    on=["game_id", "player_id"], how="inner")
        if te.height == 0:
            continue
        y = tr[f"y_{stat}"].to_numpy().astype(float).clip(0, None)
        model = lgb.train({**PARAMS, **obj}, lgb.Dataset(tr.select(X_cols).to_numpy(), y), ROUNDS)
        p = model.predict(te.select(X_cols).to_numpy())
        e, t = te["engine"].to_numpy(), te["truth"].to_numpy()
        preds.append(pl.DataFrame({"positions": "+".join(pos), "stat": stat, "season": s, "game_id": te["game_id"],
                                   "player_id": te["player_id"], "engine": e, "lgbm": p, "truth": t}))
        rows.append({"population": popn, "positions": "+".join(pos), "stat": stat, "season": s, "n": te.height,
                     "mae_engine": np.abs(e - t).mean(), "mae_lgbm": np.abs(p - t).mean(),
                     "mae_blend": np.abs((e + p) / 2 - t).mean(),
                     "bias_engine": (e - t).mean(), "bias_lgbm": (p - t).mean(),
                     "corr_engine": np.corrcoef(e, t)[0, 1], "corr_lgbm": np.corrcoef(p, t)[0, 1]})
    print(f"{'+'.join(pos):6s} {stat:16s} done ({time.time() - t0:.0f}s)", flush=True)

R = pl.DataFrame(rows)
R.write_csv(OUT)
pl.concat(preds).write_parquet(PRED)
w = lambda c: (pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()
S = R.group_by("positions", "stat").agg(pl.col("n").sum(), w("mae_engine").alias("engine"), w("mae_lgbm").alias("lgbm"),
                                        w("mae_blend").alias("blend"), w("bias_engine").alias("bias_e"),
                                        w("bias_lgbm").alias("bias_l"),
                                        (pl.col("mae_lgbm") < pl.col("mae_engine")).sum().alias("lgbm_wins_seasons"))
S = S.with_columns(((pl.col("engine") - pl.col("lgbm")) / pl.col("engine") * 100).alias("lgbm_gain_%"),
                   ((pl.col("engine") - pl.col("blend")) / pl.col("engine") * 100).alias("blend_gain_%")).sort("positions", "stat")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
print(S.with_columns(pl.col(pl.Float64).round(3)))
