"""Market-free game lines: each game's projected margin and total, and the moneyline / spread / total
probabilities priced off them, for the board and for game-line edges.

The engine's team model reads the betting market (spread, total, implied points), so a game-line
"edge" from it would be the market compared with itself. This model sees none of it: one row per game
(home side) from the engine's pregame team feature table -- both offences' and defences' form, points,
EPA and success rates per play, pace, rest, weather, home field, coach and quarterback continuity --
with every market column dropped. A ridge regression on the core team-strength signals (points, EPA,
success and explosive rates per play, sack rate, pace, rest, weather, QB and coach continuity), which
beat LightGBM on all 552 features walk-forward 2019-2026 (research/gamelines_variants.py: margin MAE
10.19 against 10.42, ML log loss 0.635 against 0.645; the closing market 9.82 / 0.610). Adding team sums
of the player projections moved it by under 0.01 (research/gamelines_players.py), so it is not used.
Walk-forward by season;
errors are normal with a spread measured on the training seasons (whole-number scores, as in the NBA
tool, so integer lines can push). Benchmarked against the closing spread and total (nflverse), which
cost no odds credits.

    python gamelines.py       # walk-forward 2019-2025 vs the closing lines; fit on all; save
"""
from __future__ import annotations

import glob
import json
import pickle
import re

import numpy as np
import polars as pl
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

C.engine_on_path()
from src.data import storage  # noqa: E402

MARKET = re.compile(r"(spread|total_line|implied|moneyline|is_favorite|market)")
CORE = re.compile(r"^(opp_)?(off|def)_(points|epa_per_play|epa_per_dropback|epa_per_rush|success_rate|plays|"
                  r"yards_per_attempt|explosive_pass_rate|sack_rate)_(est|ewma|season)$")
CONTEXT = ("rest_days", "opp_rest_days", "qb_change", "qb_games_with_team_prior", "coach_change", "is_indoor",
           "div_game", "neutral_site", "off_bye", "short_week", "league_year", "temp", "wind")
FILE = C.MODELS / "gamelines.pkl"


def team_features(seasons: list[int]) -> pl.DataFrame:
    files = [f for s in seasons for f in glob.glob(str(C.ENGINE / f"data/features/team_week_features/season={s}/*.parquet"))]
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def game_rows(seasons: list[int]) -> pl.DataFrame:
    """One row per game from the home side, with the result where played."""
    tf = team_features(seasons).filter(pl.col("is_home").cast(pl.Boolean))
    g = storage.load("games", layer="processed", seasons=seasons).select(
        "game_id", "home_score", "away_score", "spread_line", pl.col("total_line").alias("mkt_total"), "season_type")
    return tf.join(g.rename({"spread_line": "mkt_spread"}), on="game_id", how="left").with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        (pl.col("home_score") + pl.col("away_score")).alias("total"))


def feature_cols(f: pl.DataFrame) -> list[str]:
    cols = [c for c, t in f.schema.items() if (t.is_numeric() or t == pl.Boolean) and not MARKET.search(c)]
    return [c for c in cols if CORE.match(c)] + [c for c in CONTEXT if c in cols]


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 13)))


def _X(f: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return f.select([pl.col(c).cast(pl.Float64) if c in f.columns else pl.lit(None, dtype=pl.Float64).alias(c)
                     for c in cols]).to_numpy()


def fit(train: pl.DataFrame) -> dict:
    tr = train.filter(pl.col("margin").is_not_null())
    cols = feature_cols(tr)
    X = _X(tr, cols)
    out = {"cols": cols}
    for target in ("margin", "total"):
        y = tr[target].to_numpy().astype(float)
        m = _ridge().fit(X, y)
        # the error spread out of sample is wider than in sample; measured on the last training season
        last = tr["season"].max()
        inner = tr.filter(pl.col("season") < last)
        held = tr.filter(pl.col("season") == last)
        if inner.height > 500 and held.height:
            m2 = _ridge().fit(_X(inner, cols), inner[target].to_numpy().astype(float))
            sd = float(np.std(held[target].to_numpy() - m2.predict(_X(held, cols))))
        else:
            sd = float(np.std(y - m.predict(X)))
        out[target] = {"model": m, "sd": sd}
    return out


def predict(model: dict, rows: pl.DataFrame) -> pl.DataFrame:
    X = _X(rows, model["cols"])
    m, t = model["margin"]["model"].predict(X), model["total"]["model"].predict(X)
    sdm = model["margin"]["sd"]
    return rows.select("game_id", "season", "week", pl.col("team").alias("home"), pl.col("opponent").alias("away")).with_columns(
        pl.Series("margin_hat", m), pl.Series("total_hat", t), pl.lit(sdm).alias("sd_margin"),
        pl.lit(model["total"]["sd"]).alias("sd_total"), pl.Series("p_home", stats.norm.cdf(m / sdm)))


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(tests=(2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)) -> pl.DataFrame:
    allg = game_rows(list(range(C.FIRST_SEASON, C.CURRENT_SEASON + 1)))
    rows, oos = [], []
    for s in tests:
        model = fit(allg.filter(pl.col("season") < s))
        te = allg.filter((pl.col("season") == s) & pl.col("margin").is_not_null() & pl.col("mkt_spread").is_not_null())
        p = predict(model, te)
        m, t = p["margin_hat"].to_numpy(), p["total_hat"].to_numpy()
        ym, yt = te["margin"].to_numpy().astype(float), te["total"].to_numpy().astype(float)
        ms, mt = te["mkt_spread"].to_numpy().astype(float), te["mkt_total"].to_numpy().astype(float)
        oos.append(pl.DataFrame({"season": s, "pred_margin": m, "margin": ym, "pred_total": t, "total": yt}))
        win = (ym > 0).astype(float)
        keep = ym != 0
        sd_mkt = model["margin"]["sd"]
        rows.append({"season": s, "games": te.height,
                     "margin_mae": np.abs(m - ym).mean(), "margin_mae_market": np.abs(ms - ym).mean(),
                     "total_mae": np.abs(t - yt).mean(), "total_mae_market": np.abs(mt - yt).mean(),
                     "winner": float(((m > 0) == (ym > 0))[keep].mean()), "winner_market": float(((ms > 0) == (ym > 0))[keep].mean()),
                     "ml_ll": _ll(stats.norm.cdf(m / model["margin"]["sd"])[keep], win[keep]),
                     "ml_ll_market": _ll(stats.norm.cdf(ms / sd_mkt)[keep], win[keep]),
                     "corr_margin_market": float(np.corrcoef(m, ms)[0, 1]), "corr_total_market": float(np.corrcoef(t, mt)[0, 1]),
                     "sd_margin": model["margin"]["sd"], "sd_total": model["total"]["sd"]})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pl.concat(oos).write_parquet(C.MODELS / "gamelines_oos.parquet")   # for dist.GameDist (key numbers)
    return pl.DataFrame(rows)


def main() -> None:
    R = evaluate()
    pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
    w = lambda c: (pl.col(c) * pl.col("games")).sum() / pl.col("games").sum()
    print(R.select(pl.col("games").sum(), *[w(c) for c in R.columns if c not in ("season", "games")]).with_columns(pl.col(pl.Float64).round(4)))
    allg = game_rows(list(range(C.FIRST_SEASON, C.CURRENT_SEASON + 1)))
    model = fit(allg)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    with open(FILE, "wb") as fh:
        pickle.dump(model, fh)
    (C.MODELS / "gamelines_eval.json").write_text(json.dumps(R.to_dicts(), indent=1, default=float))
    print(f"saved {FILE}")


if __name__ == "__main__":
    main()
