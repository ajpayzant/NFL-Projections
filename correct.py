"""The correction layer over the engine: a direct model per statistic, a blend, and the outcome distribution.

research/audit_engine.py scored the engine's walk-forward replay against a LightGBM fitted directly on
each statistic from the same pregame rows; neither wins everywhere (the engine is better on quarterback
volume, the direct model on backs' carries, receivers' targets and touchdowns), and a blend beats the
engine on 16 of 19 lines. research/blend_calibrate.py chose the blend weight per line walk-forward, and
research/distributions.py the outcome distribution: the negative binomial prices counts (receptions,
targets, carries, attempts, completions) to within 0.3 points at the lines books post, but under-called
yardage at the lower lines by 2-5 points; the empirical distribution of outcome / projection among
similar projections is calibrated there (mean gap 0.46 against 1.16 points), so yards and touchdowns use it.

    python correct.py        # fit the direct models on 2016-2025 and the distribution tables; save
"""
from __future__ import annotations

import json
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
import dist
from dist import EMPIRICAL, Empirical

C.engine_on_path()
from src.features import store  # noqa: E402

# group -> (engine frame kind, positions, engine population, statistics)
GROUPS = {"QB": ("qb", ("QB",), "quarterback",
                 ("attempts", "completions", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards",
                  "rushing_tds")),
          "RB": ("skill", ("RB",), "running_back",
                 ("carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds")),
          "WR+TE": ("skill", ("WR", "TE"), "receiver", ("targets", "receptions", "receiving_yards", "receiving_tds"))}
YARDS = {"passing_yards", "rushing_yards", "receiving_yards"}
PARAMS = {"learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.3,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "num_threads": 12}
ROUNDS = 500
FILE = C.MODELS / "correct.pkl"
DIST_FILE = C.MODELS / "dist.pkl"


def features(f: pl.DataFrame) -> list[str]:
    return [c for c, t in f.schema.items() if t.is_numeric() and not c.startswith("y_") and "_y_" not in c
            and c != "season"]


def p_over(model: dict, group: str, stat: str, mu, line: float) -> np.ndarray:
    """The probability a player goes over `line` given his blended projection mu (for a game he plays)."""
    return dist.p_over(model["dist"], group, stat, mu, line)


# ---------------------------------------------------------------- fit

def fit(last_season: int = C.CURRENT_SEASON - 1) -> dict:
    blend = json.loads((C.MODELS / "blend.json").read_text())
    audit = pl.read_parquet(C.ROOT / "research" / "audit_engine_pred.parquet")
    out = {"direct": {}, "features": {}, "w_engine": {}, "dist": {}}
    frames: dict[str, pl.DataFrame] = {}
    for group, (kind, pos, _, stats_) in GROUPS.items():
        if kind not in frames:
            frames[kind] = store.load_training_frame(kind, list(range(C.FIRST_SEASON, last_season + 1)), played_only=True)
        f = frames[kind].filter(pl.col("projected_position").is_in(list(pos)))
        cols = features(f)
        out["features"][group] = cols
        for stat in stats_:
            tr = f.filter(pl.col(f"y_{stat}").is_not_null())
            obj = {"objective": "tweedie", "tweedie_variance_power": 1.2} if stat in YARDS else {"objective": "poisson"}
            y = tr[f"y_{stat}"].to_numpy().astype(float).clip(0, None)
            out["direct"][f"{group}|{stat}"] = lgb.train({**PARAMS, **obj}, lgb.Dataset(tr.select(cols).to_numpy(), y), ROUNDS)
            w = blend[f"{group}|{stat}"]["w_engine"]
            out["w_engine"][f"{group}|{stat}"] = w
            a = audit.filter((pl.col("positions") == group) & (pl.col("stat") == stat))
            mu = (w * a["engine"] + (1 - w) * a["lgbm"]).to_numpy()
            out["dist"][f"{group}|{stat}"] = {"emp": Empirical(mu, a["truth"].to_numpy().astype(float)),
                                              "var": blend[f"{group}|{stat}"]["var"]}
            print(f"{group:6s} {stat:16s} fitted on {tr.height:,} rows", flush=True)
    out["dist"].update(composite_tables(audit, blend))
    return out


SCORING = {"fantasy_ppr": 1.0, "fantasy_half": 0.5, "fantasy_std": 0.0}


def fantasy(frame: pl.DataFrame, reception_points: float) -> pl.Expr:
    """Fantasy points from statistic columns (missing ones count zero); fumbles and two-point
    conversions are not projected, so they are left out of both projection and truth."""
    c = lambda name: pl.col(name).fill_null(0.0) if name in frame.columns else pl.lit(0.0)
    return (0.04 * c("passing_yards") + 4 * c("passing_tds") - 2 * c("interceptions") + 0.1 * c("rushing_yards")
            + 6 * c("rushing_tds") + 0.1 * c("receiving_yards") + 6 * c("receiving_tds") + reception_points * c("receptions"))


def composite_tables(audit: pl.DataFrame, blend: dict) -> dict:
    """Distributions for quantities built from several statistics: fantasy points, anytime touchdown,
    rushing + receiving yards, passing + rushing yards -- from the same out-of-sample predictions."""
    a = audit.with_columns(pl.struct("positions", "stat").map_elements(
        lambda r: blend[f"{r['positions']}|{r['stat']}"]["w_engine"], return_dtype=pl.Float64).alias("w"))
    a = a.with_columns((pl.col("w") * pl.col("engine") + (1 - pl.col("w")) * pl.col("lgbm")).alias("mu"))
    mu = a.pivot(on="stat", index=["positions", "game_id", "player_id"], values="mu")
    y = a.pivot(on="stat", index=["positions", "game_id", "player_id"], values="truth")
    out = {}
    for group in ("QB", "RB", "WR+TE"):
        m, t = mu.filter(pl.col("positions") == group), y.filter(pl.col("positions") == group)
        j = m.join(t, on=["game_id", "player_id"], suffix="_y")
        mm = j.select([pl.col(c) for c in m.columns if c not in ("positions", "game_id", "player_id")])
        tt = j.select([pl.col(f"{c}_y").alias(c) for c in m.columns if c not in ("positions", "game_id", "player_id")])
        for name, rp in SCORING.items():
            out[f"{group}|{name}"] = {"emp": Empirical(mm.select(fantasy(mm, rp)).to_series().to_numpy(),
                                                       tt.select(fantasy(tt, rp)).to_series().to_numpy())}
        tds = [c for c in ("rushing_tds", "receiving_tds") if c in mm.columns]
        if tds:
            out[f"{group}|anytime_td"] = {"emp": Empirical(mm.select(pl.sum_horizontal(tds)).to_series().to_numpy(),
                                                           tt.select(pl.sum_horizontal(tds)).to_series().to_numpy())}
        if "rushing_yards" in mm.columns and "receiving_yards" in mm.columns:
            out[f"{group}|rush_rec_yards"] = {"emp": Empirical(
                mm.select(pl.col("rushing_yards") + pl.col("receiving_yards")).to_series().to_numpy(),
                tt.select(pl.col("rushing_yards") + pl.col("receiving_yards")).to_series().to_numpy())}
        if "passing_yards" in mm.columns:
            out[f"{group}|pass_rush_yards"] = {"emp": Empirical(
                mm.select(pl.col("passing_yards") + pl.col("rushing_yards")).to_series().to_numpy(),
                tt.select(pl.col("passing_yards") + pl.col("rushing_yards")).to_series().to_numpy())}
    return out


def load() -> dict:
    with open(FILE, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------- apply

def apply(model: dict, engine_rows: dict[str, pl.DataFrame], season: int, week: int) -> pl.DataFrame:
    """Blend the engine's forward rows with the direct models; one row per player with `<stat>` means."""
    out = []
    frames: dict[str, pl.DataFrame] = {}
    for group, (kind, pos, popn, stats_) in GROUPS.items():
        eng = engine_rows.get(popn)
        if eng is None or eng.is_empty():
            continue
        if kind not in frames:
            frames[kind] = store.load_training_frame(kind, [season], played_only=False).filter(pl.col("week") == week)
        f = frames[kind].filter(pl.col("projected_position").is_in(list(pos)))
        f = f.join(eng.select("game_id", "player_id"), on=["game_id", "player_id"], how="semi")
        cols = model["features"][group]
        X = f.select([pl.col(c) if c in f.columns else pl.lit(None, dtype=pl.Float64).alias(c) for c in cols]).to_numpy()
        res = f.select("game_id", "team", "player_id", "player_name", "projected_position")
        for stat in stats_:
            key = f"{group}|{stat}"
            direct = model["direct"][key].predict(X)
            e = res.join(eng.select("game_id", "player_id", pl.col(_engine_col(eng, stat)).alias("e")),
                         on=["game_id", "player_id"], how="left")["e"].to_numpy()
            e = np.where(np.isfinite(e), e, direct)
            w = model["w_engine"][key]
            res = res.with_columns(pl.Series(stat, w * e + (1 - w) * direct), pl.Series(f"{stat}_engine", e),
                                   pl.Series(f"{stat}_direct", direct))
        out.append(res.with_columns(pl.lit(group).alias("group")))
    return pl.concat(out, how="diagonal") if out else pl.DataFrame()


def _engine_col(frame: pl.DataFrame, stat: str) -> str:
    for c in (f"p_{stat}_chain", f"p_{stat}"):
        if c in frame.columns:
            return c
    raise KeyError(stat)


if __name__ == "__main__":
    m = fit()
    C.MODELS.mkdir(parents=True, exist_ok=True)
    with open(FILE, "wb") as fh:
        pickle.dump(m, fh)
    with open(DIST_FILE, "wb") as fh:          # what the app reads: no models, no engine
        pickle.dump(m["dist"], fh)
    print(f"saved {FILE} and {DIST_FILE}")
