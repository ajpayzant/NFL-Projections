"""Does the player view (team sums of the out-of-sample player projections) add to the ridge game model?

Stage 1: ridge on the core team features, walk-forward (out-of-sample margin / total per game).
Stage 2: OLS of the result on [ridge, projected passing yards, passing TDs, rushing yards, rushing TDs,
receiving TDs] as home-minus-away (margin) and home-plus-away (total), fitted on the seasons before.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gamelines as G

allg = G.game_rows(list(range(2016, 2026)))
cols = G.feature_cols(allg)
CORE = re.compile(r"^(opp_)?(off|def)_(points|epa_per_play|epa_per_dropback|epa_per_rush|success_rate|plays|yards_per_attempt|explosive_pass_rate|sack_rate)_(est|ewma|season)$")
core = [c for c in cols if CORE.match(c)] + [c for c in cols if c in ("rest_days", "opp_rest_days", "qb_change", "qb_games_with_team_prior",
                                                                      "coach_change", "is_indoor", "div_game", "neutral_site", "off_bye", "short_week", "league_year", "temp", "wind")]
ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 13)))
oos = []
for s in range(2019, 2026):
    tr = allg.filter((pl.col("season") < s) & pl.col("margin").is_not_null())
    te = allg.filter((pl.col("season") == s) & pl.col("margin").is_not_null())
    p = {t: ridge().fit(G._X(tr, core), tr[t].to_numpy().astype(float)).predict(G._X(te, core)) for t in ("margin", "total")}
    oos.append(te.select("game_id", "season", pl.col("team").alias("home"), pl.col("opponent").alias("away"), "margin", "total",
                         "mkt_spread", "mkt_total").with_columns(pl.Series("r_margin", p["margin"]), pl.Series("r_total", p["total"])))
O = pl.concat(oos)

# team sums of the blended player projections (out of sample, 2021-2025)
P = pl.read_parquet(ROOT / "research" / "audit_engine_pred.parquet")
B = json.loads((ROOT / "data" / "models" / "blend.json").read_text())
P = P.with_columns(pl.struct("positions", "stat").map_elements(lambda r: B[f"{r['positions']}|{r['stat']}"]["w_engine"], return_dtype=pl.Float64).alias("w"))
P = P.with_columns((pl.col("w") * pl.col("engine") + (1 - pl.col("w")) * pl.col("lgbm")).alias("mu"))
import config as C
C.engine_on_path()
from src.data import storage
teams = storage.load("player_games", layer="processed", seasons=list(range(2021, 2026))).select("game_id", "player_id", "team").unique()
P = P.join(teams, on=["game_id", "player_id"], how="left")
want = {("QB", "passing_yards"): "pass_yds", ("QB", "passing_tds"): "pass_td", ("QB", "rushing_yards"): "rush_yds",
        ("RB", "rushing_yards"): "rush_yds", ("QB", "rushing_tds"): "rush_td", ("RB", "rushing_tds"): "rush_td"}
P = P.with_columns(pl.struct("positions", "stat").map_elements(lambda r: want.get((r["positions"], r["stat"])), return_dtype=pl.Utf8).alias("k")).filter(pl.col("k").is_not_null())
T = P.group_by("game_id", "team", "k").agg(pl.col("mu").sum()).pivot(on="k", index=["game_id", "team"], values="mu")
F = ["pass_yds", "pass_td", "rush_yds", "rush_td"]
O = O.join(T.rename({c: f"h_{c}" for c in F}).rename({"team": "home"}), on=["game_id", "home"], how="inner") \
     .join(T.rename({c: f"a_{c}" for c in F}).rename({"team": "away"}), on=["game_id", "away"], how="inner")
O = O.with_columns([pl.col(f"{p}_{c}").fill_null(0.0).fill_nan(0.0) for p in ("h", "a") for c in F])
print("games with player sums:", O.height, O.group_by("season").len().sort("season").to_dicts())


def stage2(df, target, ridge_col, sign):
    X = np.column_stack([np.ones(df.height), df[ridge_col].to_numpy()] + [df[f"h_{c}"].to_numpy() + sign * df[f"a_{c}"].to_numpy() for c in F])
    return X


res = {"ridge": [], "ridge+players": [], "market": []}
for s in (2022, 2023, 2024, 2025):
    tr, te = O.filter(pl.col("season") < s), O.filter(pl.col("season") == s)
    out = {}
    for target, rc, sign in (("margin", "r_margin", -1), ("total", "r_total", 1)):
        b = np.linalg.lstsq(stage2(tr, target, rc, sign), tr[target].to_numpy().astype(float), rcond=None)[0]
        out[target] = stage2(te, target, rc, sign) @ b
    ym, yt = te["margin"].to_numpy().astype(float), te["total"].to_numpy().astype(float)
    res["ridge"].append((te["r_margin"].to_numpy(), te["r_total"].to_numpy(), ym, yt))
    res["ridge+players"].append((out["margin"], out["total"], ym, yt))
    res["market"].append((te["mkt_spread"].to_numpy().astype(float), te["mkt_total"].to_numpy().astype(float), ym, yt))
for k, parts in res.items():
    m, t, ym, yt = (np.concatenate([p[i] for p in parts]) for i in range(4))
    keep = ym != 0
    sd = np.std(ym - m)
    p = stats.norm.cdf(m / sd)
    ll = float(-np.mean((ym[keep] > 0) * np.log(p[keep]) + (ym[keep] < 0) * np.log(1 - p[keep])))
    print(f"{k:15s} 2022-25 ({len(ym)} games): margin MAE {np.abs(m - ym).mean():.3f} | ML ll {ll:.4f} | total MAE {np.abs(t - yt).mean():.3f}")
