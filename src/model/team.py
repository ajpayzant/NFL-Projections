"""Team environment: what a season looks like, then what each of the 17 games looks like.

Two stages, and the split is the whole point. The workbook this replaces had only the first one -- a
single flat per-game number for a 17-game season -- so it could not tell a trip to Buffalo in December
from a home dome game against the worst defence in the league.

**Stage 1, season shape.** For each team metric, `estimate = league_mean + keep x (recent_form -
league_mean)` where `recent_form = w x last season + (1 - w) x the season before`. `w` and `keep` are
grid-searched per metric on 2019-2025, each target season predicted only from seasons before it, and
scored against the obvious baseline of copying last season forward. Pace and pass rate are sticky and
keep most of their signal; touchdown rates and explosive rates are not and get pulled hard toward the
mean. The fit says which is which instead of us asserting it.

**Stage 2, per-game factors.** A multiplicative chain on top of the season estimate:

    game value = season estimate  x  f_home x f_rest x f_roof x f_surface x f_wind x f_temp x f_div
                                  x  f_script x f_part_of_season
                                  x  (1 + g_market x (implied_ratio - 1))
                                  x  (1 + g_defence x (opponent_factor - 1))

Everything in that chain is fitted from 2016-2025 splits, and fitted the same way it is applied:

- The response is always `rel = game value / that team-season's mean`, which cancels team quality
  exactly. A bucket mean of `rel` is therefore a clean multiplicative effect -- no need to control for
  the fact that good teams play more home games against bad defences than the reverse.
- Terms are fitted by backfitting: each term is estimated on the residual left after dividing out the
  others, three passes. Without it every factor would double-count -- home field, a soft defence and a
  high total are three views of the same game, and the posted line already prices all three.
- Categorical buckets are shrunk toward 1 by `n / (n + context_k)` team-games, so a 40-game bucket
  cannot move a projection the way a 2,000-game one can, then renormalised to a weighted mean of 1 so
  the chain does not quietly inflate the league.
- Continuous terms (the posted line, the opponent's defensive rating) are fitted as a slope on the
  ratio rather than taken at face value: `1 + g x (x - 1)`. A defence that allowed 6% more targets than
  average does not hand the next offence 6% more targets, and `g` measures how much actually transfers.

The defensive ratings used in fitting are same-season, which measures the true effect size; the ones
used in projection come out of stage 1 and are already regressed to the mean, which handles the fact
that a defensive rating is a noisy thing to carry forward. Keeping those two jobs separate is
deliberate: fitting with a regressed rating would understate the effect, applying with a raw one would
overstate it.

Backfitting means the fitted factors are *partial* effects, and they read strangely on their own: the
`script` term says big favourites run 0.9% *less* than they average, because the defence term has
already explained why favourites run more -- they are favourites against defences that let you run.
The raw split says +3.7%, which is the honest headline. Both are stored, `value` for the chain to
multiply and `marginal` for a page to display, and confusing the two is how a chain ends up
triple-counting one game.

Nothing here is kept on faith. Every metric's `(w, keep)` is scored leave-one-season-out against a
flat league mean, and a metric that cannot beat it -- a defence's yards per carry allowed, red-zone
touchdown rate -- is given `keep = 0` and flagged, so it contributes a constant rather than a
confident guess. The per-game chain is scored the same way and dropped entirely for any metric where
it loses, leaving that metric's factors at 1.0.

Market lines are used where they exist and ignored where they do not: `market_weight` is fitted by
asking how much of a game's actual scoring the posted total explains that our own team estimates do
not. For 2026, 112 of 272 games have a line posted as of August; the rest fall back to the model's own
number rather than to a stale one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import polars as pl

from src.config import (
    FITTED,
    HISTORY_FROM,
    LAST_COMPLETE_SEASON,
    PROJ_SEASON,
    TEAM_FIXUP,
    Settings,
    ensure_dirs,
)
from src.data import history, lake

SHAPE_PATH = FITTED / "team_shape.json"
FACTOR_PATH = FITTED / "team_factors.parquet"
MARKET_PATH = FITTED / "team_market.json"

FIT_FIRST_TARGET = 2019          # needs two prior seasons of history and a league mean
W_GRID = tuple(round(x, 2) for x in np.arange(0.50, 1.001, 0.05))
KEEP_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.001, 0.05))
BACKFIT_ROUNDS = 3
SLOPE_CLIP = (-1.5, 3.0)         # a fitted transfer coefficient outside this is a small-sample artefact
FACTOR_FLOOR = 0.35              # a multiplicative chain must never go negative
MARKET_OFFSET_K = 4.0            # games of evidence at which a team's market/model gap counts half
CLIMATE_WINDOW = 5               # seasons of venue weather behind an expected temperature


# --------------------------------------------------------------------------- #
# what gets projected
# --------------------------------------------------------------------------- #
# Per-game context metric -> the season-shape metric it scales. Everything the player layer needs a
# team denominator for is in here; the extras exist because a team page should show them.
SHAPE_OF = {
    "plays": "plays_per_game",
    "volume_plays": "volume_plays_per_game",
    "drives": "drives_per_game",
    "seconds_per_play": "seconds_per_play",
    "plays_per_drive": "plays_per_drive",
    "dropbacks": "dropbacks_per_game",
    "pass_attempts": "pass_att_per_game",
    "carries": "rush_att_per_game",
    "targets": "targets_per_game",
    "air_yards": "air_yards_per_game",
    "late_down_targets": "late_down_tgt_per_game",
    "dropback_rate": "dropback_rate",
    "neutral_pass_rate": "neutral_pass_rate",
    "targets_per_attempt": "targets_per_attempt",
    "points": "points_per_game",
    "offensive_tds": "off_td_per_game",
    "pass_tds": "pass_td_per_game",
    "rush_tds": "rush_td_per_game",
    "red_zone_trips": "rz_trips_per_game",
    "red_zone_targets": "rz_target_per_game",
    "red_zone_carries": "rz_carry_per_game",
    "inside_5_carries": "inside_5_per_game",
    "short_yardage_carries": "short_yardage_per_game",
    "yards_per_attempt": "yards_per_attempt",
    "yards_per_carry": "yards_per_carry",
    "success_rate": "success_rate",
}
CONTEXT_METRICS = tuple(SHAPE_OF)

FAMILY = {
    **{m: "pace" for m in ("plays", "volume_plays", "drives", "seconds_per_play", "plays_per_drive")},
    **{m: "mix" for m in ("dropbacks", "pass_attempts", "carries", "targets", "air_yards",
                          "late_down_targets", "dropback_rate", "neutral_pass_rate",
                          "targets_per_attempt")},
    **{m: "scoring" for m in ("points", "offensive_tds", "pass_tds", "rush_tds", "red_zone_trips",
                              "red_zone_targets", "red_zone_carries", "inside_5_carries",
                              "short_yardage_carries")},
    **{m: "efficiency" for m in ("yards_per_attempt", "yards_per_carry", "success_rate")},
}

# Which defensive rating a metric answers to, as a tuple whose ratings multiply: a defence that
# faces 3% more plays than average and passes on 4% more of them is a 7% pass-volume matchup. Empty
# where no defensive column measures the same thing -- pace-of-drive and targets-per-attempt are the
# offence's own habits.
#
# Pass volume is deliberately built from `plays_allowed_pg x pass_rate_faced` rather than from
# `pass_att_allowed_pg`, which measures the same quantity directly. Both components carry forward
# (stage 1 keeps 0.35 of each); the aggregate does not carry at all and auto-reverts to the league
# mean, which would leave every defence a neutral pass matchup. Same reasoning sends yards per carry
# to `success_rate_allowed`: a defence's yards-per-carry allowed is famously noise year to year --
# stage 1 reverts it -- while its success rate allowed is one of the most stable defensive numbers
# there is, so it is what a rushing matchup can honestly be built on.
DEF_OF = {
    "plays": ("plays_allowed_pg",), "volume_plays": ("plays_allowed_pg",),
    "drives": ("plays_allowed_pg",),
    "seconds_per_play": (), "plays_per_drive": (),
    "dropbacks": ("plays_allowed_pg", "pass_rate_faced"),
    "pass_attempts": ("plays_allowed_pg", "pass_rate_faced"),
    "targets": ("plays_allowed_pg", "pass_rate_faced"),
    "air_yards": ("plays_allowed_pg", "pass_rate_faced"),
    "late_down_targets": ("plays_allowed_pg", "pass_rate_faced"),
    "carries": ("carries_allowed_pg",),
    "dropback_rate": ("pass_rate_faced",), "neutral_pass_rate": ("pass_rate_faced",),
    "targets_per_attempt": (),
    "points": ("points_allowed_pg",),
    "offensive_tds": ("points_allowed_pg",), "pass_tds": ("points_allowed_pg",),
    "rush_tds": ("points_allowed_pg",), "red_zone_trips": ("points_allowed_pg",),
    "red_zone_targets": ("points_allowed_pg",), "red_zone_carries": ("points_allowed_pg",),
    "inside_5_carries": ("points_allowed_pg",), "short_yardage_carries": ("points_allowed_pg",),
    "yards_per_attempt": ("yards_per_att_allowed",),
    "yards_per_carry": ("success_rate_allowed",),
    "success_rate": ("success_rate_allowed",),
}
DEF_RATINGS = tuple(sorted({c for cols in DEF_OF.values() for c in cols}))


def _def_x(lookup, metric: str) -> np.ndarray | None:
    """The opponent term for one metric: the product of its ratings, each already a ratio to the
    league mean. A missing rating counts as neutral rather than dropping the game."""
    cols = DEF_OF.get(metric) or ()
    if not cols:
        return None
    out = None
    for c in cols:
        x = np.nan_to_num(lookup(c), nan=1.0)
        out = x if out is None else out * x
    return out

# Season-shape metrics. The offensive list is everything the chain scales plus the descriptive shape
# a team page shows; the defensive list is what an opponent adjustment can be built from.
OFFENSE_SHAPE = tuple(dict.fromkeys(
    list(SHAPE_OF.values())
    + ["pass_rate", "proe", "epa_per_play", "epa_per_dropback", "epa_per_rush", "red_zone_rate",
       "explosive_pass_rate", "explosive_rush_rate", "td_per_rz_trip", "pass_td_share_of_off",
       "points_allowed_per_game"]
))
DEFENSE_SHAPE = (
    "plays_allowed_pg", "pass_att_allowed_pg", "carries_allowed_pg", "targets_allowed_pg",
    "points_allowed_pg", "off_tds_allowed_pg", "rz_trips_allowed_pg", "yards_per_att_allowed",
    "yards_per_carry_allowed", "pass_rate_faced", "epa_per_play_allowed", "success_rate_allowed",
)


# --------------------------------------------------------------------------- #
# stage 1: season shape
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def offense_seasons(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """`team_seasons` plus the per-game columns the chain needs and history.py does not derive."""
    ts = history.team_seasons(seasons)
    extra = {
        "air_yards_per_game": "air_yards",
        "late_down_tgt_per_game": "late_down_targets",
        "rz_target_per_game": "red_zone_targets",
        "rz_carry_per_game": "red_zone_carries",
    }
    return ts.with_columns(
        [(pl.col(src) / pl.col("games")).alias(dst) for dst, src in extra.items() if src in ts.columns]
    )


@lru_cache(maxsize=4)
def defense_ratings(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """`defense_seasons` keyed on `team`, with the two per-game columns it stops short of."""
    ds = history.defense_seasons(seasons)
    return ds.rename({"defense": "team"}).with_columns(
        (pl.col("targets_allowed") / pl.col("games")).alias("targets_allowed_pg"),
        (pl.col("rz_trips_allowed") / pl.col("games")).alias("rz_trips_allowed_pg"),
    )


def _long(df: pl.DataFrame, metrics: Sequence[str], side: str) -> pl.DataFrame:
    have = [m for m in metrics if m in df.columns]
    return (
        df.select("season", "team", *have)
        .unpivot(index=["season", "team"], variable_name="metric", value_name="value")
        .with_columns(pl.lit(side).alias("side"), pl.col("value").cast(pl.Float64))
    )


@lru_cache(maxsize=4)
def shape_panel(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One row per (season, team, side, metric) with the value and that season's league mean."""
    seasons = seasons or lake.history_seasons()
    panel = pl.concat([
        _long(offense_seasons(seasons), OFFENSE_SHAPE, "offense"),
        _long(defense_ratings(seasons), DEFENSE_SHAPE, "defense"),
    ])
    return panel.with_columns(
        pl.col("value").mean().over(["season", "side", "metric"]).alias("league_mean")
    )


def _fit_rows(
    panel: pl.DataFrame, targets: Sequence[int], require_actual: bool = True
) -> pl.DataFrame:
    """Each target season joined to its two predecessors -- values and league means both.

    `require_actual=False` is the projection case: the target season has not been played, so its
    `actual` is null and only the two lags have to be there.
    """
    keys = ["side", "metric", "team"]
    needed = ["last", "prev", "lmean1", "lmean2"]
    lag1 = panel.select(*keys, (pl.col("season") + 1).alias("season"),
                        pl.col("value").alias("last"), pl.col("league_mean").alias("lmean1"))
    lag2 = panel.select(*keys, (pl.col("season") + 2).alias("season"),
                        pl.col("value").alias("prev"), pl.col("league_mean").alias("lmean2"))
    return (
        panel.filter(pl.col("season").is_in(list(targets)))
        .select(*keys, "season", pl.col("value").alias("actual"))
        .join(lag1, on=[*keys, "season"], how="inner")
        .join(lag2, on=[*keys, "season"], how="inner")
        .drop_nulls(["actual", *needed] if require_actual else needed)
    )


def _estimate(last, prev, lmean1, lmean2, w: float, keep: float):
    """The stage-1 formula, on arrays. The league mean is blended with the same `w`, because a
    projection made in August 2026 does not know 2026's league mean either."""
    recent = w * last + (1.0 - w) * prev
    lmean = w * lmean1 + (1.0 - w) * lmean2
    return lmean + keep * (recent - lmean)


def fit_shape(
    seasons: tuple[int, ...] | None = None,
    targets: Sequence[int] | None = None,
) -> dict[str, dict]:
    """Grid-search `(w, keep)` per metric out of sample. Returns one record per metric."""
    panel = shape_panel(seasons)
    targets = list(targets or range(FIT_FIRST_TARGET, LAST_COMPLETE_SEASON + 1))
    rows = _fit_rows(panel, targets)
    combos = [(w, keep) for w in W_GRID for keep in KEEP_GRID]
    flat = [i for i, (_, keep) in enumerate(combos) if keep == 0.0]
    out: dict[str, dict] = {}

    def select(err: np.ndarray, among: Sequence[int] | None = None) -> int:
        cols = list(among) if among is not None else range(err.shape[1])
        means = err.mean(axis=0)
        return min(cols, key=lambda i: means[i])

    def loso(err: np.ndarray, season: np.ndarray, among: Sequence[int] | None = None) -> float:
        """MAE of the grid choice made without the season it is scored on.

        `w` and `keep` are two parameters fitted on a few hundred team-seasons, so in-sample they
        flatter themselves. Restricting `among` to the `keep == 0` columns scores the same procedure
        for the flat league-mean model, which is what makes the two comparable.
        """
        total, n = 0.0, 0
        for s in sorted(set(season.tolist())):
            train, test = season != s, season == s
            if not train.any() or not test.any():
                continue
            total += float(err[test, select(err[train], among)].sum())
            n += int(test.sum())
        return total / n if n else float("nan")

    for (side, metric), grp in rows.group_by(["side", "metric"], maintain_order=True):
        last = grp["last"].to_numpy()
        prev = grp["prev"].to_numpy()
        l1, l2 = grp["lmean1"].to_numpy(), grp["lmean2"].to_numpy()
        actual = grp["actual"].to_numpy()
        season = grp["season"].to_numpy()

        # absolute error of every (w, keep) on every team-season, once
        err = np.column_stack([
            np.abs(_estimate(last, prev, l1, l2, w, keep) - actual) for w, keep in combos
        ])
        mae_loso = loso(err, season)
        mean_loso = loso(err, season, flat)

        # Auto-revert: a metric whose own two seasons of form do not beat a flat league mean out of
        # sample is told to carry nothing forward. `yards_per_carry_allowed` is the honest example --
        # a defence's yards per carry one year says nothing about the next, so pretending otherwise
        # would put a spurious matchup adjustment in front of every running back we project.
        reverted = mean_loso <= mae_loso
        best = select(err, flat if reverted else None)
        w, keep = combos[best]

        copy_last = float(np.abs(last - actual).mean())
        quote = min(mae_loso, mean_loso)
        out[f"{side}.{metric}"] = {
            "side": side, "metric": metric, "w": w, "keep": keep, "n": int(len(actual)),
            "mae": float(err[:, best].mean()), "mae_loso": quote,
            "mae_copy_last": copy_last, "mae_league_mean": mean_loso,
            "reverted_to_league_mean": bool(reverted),
            "gain_vs_copy_last_pct": 100.0 * (copy_last - quote) / copy_last if copy_last else 0.0,
            "gain_vs_mean_pct": 100.0 * (mean_loso - quote) / mean_loso if mean_loso else 0.0,
            "league_mean": float(actual.mean()),
            "spread": float(actual.std()),
        }
    return out


@lru_cache(maxsize=1)
def load_shape_fit() -> dict[str, dict]:
    if not SHAPE_PATH.exists():
        return {}
    return json.loads(SHAPE_PATH.read_text(encoding="utf-8"))


def season_shape(season: int = PROJ_SEASON, settings: Settings | None = None) -> pl.DataFrame:
    """Projected team shape for `season`, one row per (team, side, metric).

    Carries `last`, `prev`, `recent_form`, `league_mean` and the fitted `w`/`keep` alongside the
    estimate, because a user who wants to move a number is owed the arithmetic that produced it.
    """
    st = settings or Settings()
    fit = load_shape_fit()
    panel = shape_panel(tuple(s for s in lake.history_seasons() if s < season))
    rows = _fit_rows(
        pl.concat([panel, _blank_target(panel, season)]), [season], require_actual=False
    )
    w_map = {k: v["w"] for k, v in fit.items()}
    keep_map = {k: v["keep"] for k, v in fit.items()}
    key = pl.col("side") + pl.lit(".") + pl.col("metric")
    return (
        rows.with_columns(
            key.replace_strict(w_map, default=st.team_weight_recent, return_dtype=pl.Float64).alias("w"),
            key.replace_strict(keep_map, default=st.team_keep_vs_mean, return_dtype=pl.Float64).alias("keep"),
        )
        .with_columns(
            (pl.col("w") * pl.col("last") + (1 - pl.col("w")) * pl.col("prev")).alias("recent_form"),
            (pl.col("w") * pl.col("lmean1") + (1 - pl.col("w")) * pl.col("lmean2")).alias("league_mean"),
        )
        .with_columns(
            (pl.col("league_mean") + pl.col("keep") * (pl.col("recent_form") - pl.col("league_mean")))
            .alias("estimate")
        )
        .with_columns(pl.lit(season, pl.Int32).alias("season"))
        .drop("actual", "lmean1", "lmean2")
        .sort(["side", "metric", "team"])
    )


def _blank_target(panel: pl.DataFrame, season: int) -> pl.DataFrame:
    """A placeholder row per (team, side, metric) for the season being projected.

    `_fit_rows` is written for backtesting, where the target season's actual exists. Projecting means
    asking it for a season that has not happened, so the target rows are stubbed with a null actual
    and the join to the two prior seasons does the rest.
    """
    teams = panel.select("team").unique()
    grid = panel.select("side", "metric").unique().join(teams, how="cross")
    return grid.with_columns(
        pl.lit(season, panel.schema["season"]).alias("season"),
        pl.lit(None, pl.Float64).alias("value"),
        pl.lit(None, pl.Float64).alias("league_mean"),
    ).select(panel.columns)


def shape_wide(season: int = PROJ_SEASON, settings: Settings | None = None) -> pl.DataFrame:
    """`season_shape` pivoted to one row per team, columns `off_*` / `def_*`."""
    sh = season_shape(season, settings)
    return (
        sh.with_columns(
            (pl.when(pl.col("side") == "offense").then(pl.lit("off_")).otherwise(pl.lit("def_"))
             + pl.col("metric")).alias("col")
        )
        .pivot(on="col", index="team", values="estimate")
        .sort("team")
    )


# --------------------------------------------------------------------------- #
# stage 2: per-game context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    """A categorical context term: a bucket label per team-game, and where it applies."""

    name: str
    bucket: pl.Expr
    applies: pl.Expr | None = None


def _rest_bucket() -> pl.Expr:
    r = pl.col("rest_days")
    return (
        pl.when(r <= 4).then(pl.lit("short"))
        .when(r <= 6).then(pl.lit("six"))
        .when(r == 7).then(pl.lit("normal"))
        .when(r <= 11).then(pl.lit("long"))
        .otherwise(pl.lit("off_bye"))
    )


def _turf() -> pl.Expr:
    s = pl.col("surface").str.strip_chars().str.to_lowercase()
    return (
        pl.when(s.is_null() | (s == "")).then(pl.lit(None, pl.String))
        .when(s.str.contains("grass")).then(pl.lit("grass"))
        .otherwise(pl.lit("turf"))
    )


def _script_bucket(col: str = "team_spread") -> pl.Expr:
    """Game script from the team's own side of the line. Negative is favoured."""
    s = pl.col(col)
    return (
        pl.when(s.is_null()).then(pl.lit(None, pl.String))
        .when(s >= 7).then(pl.lit("big_dog"))
        .when(s >= 3).then(pl.lit("dog"))
        .when(s > -3).then(pl.lit("even"))
        .when(s > -7).then(pl.lit("fav"))
        .otherwise(pl.lit("big_fav"))
    )


def _week_bucket() -> pl.Expr:
    """Where in the season the game sits. Week 1 is a rusty-offence week and the finale is a
    resting-starters week; neither is an average game, and a weekly projection has to say so.

    The finale is found by week number relative to the calendar's length, not by a fixed 18: the
    league played 17-week seasons through 2020, and lumping their week 17 in with a modern week 17
    would mix the resting week into an ordinary one.
    """
    w = pl.col("week")
    last = pl.when(pl.col("season") >= 2021).then(18).otherwise(17)
    return (
        pl.when(w >= last).then(pl.lit("finale"))
        .when(w <= 1).then(pl.lit("wk1"))
        .when(w <= 4).then(pl.lit("early"))
        .when(w <= 9).then(pl.lit("mid"))
        .when(w <= 13).then(pl.lit("late"))
        .otherwise(pl.lit("stretch"))
    )


SPLITS = (
    Split("part_of_season", _week_bucket()),
    Split("venue", pl.when(pl.col("neutral_site")).then(pl.lit("neutral"))
          .when(pl.col("is_home")).then(pl.lit("home")).otherwise(pl.lit("away"))),
    Split("rest", _rest_bucket()),
    Split("roof", pl.when(pl.col("is_indoor")).then(pl.lit("indoor")).otherwise(pl.lit("outdoor"))),
    Split("surface", _turf()),
    Split("wind", pl.when(pl.col("wind") < 8).then(pl.lit("calm"))
          .when(pl.col("wind") < 15).then(pl.lit("breezy")).otherwise(pl.lit("windy")),
          applies=~pl.col("is_indoor") & pl.col("wind").is_not_null()),
    Split("temp", pl.when(pl.col("temp") < 32).then(pl.lit("freezing"))
          .when(pl.col("temp") < 50).then(pl.lit("cold"))
          .when(pl.col("temp") < 75).then(pl.lit("mild")).otherwise(pl.lit("hot")),
          applies=~pl.col("is_indoor") & pl.col("temp").is_not_null()),
    Split("divisional", pl.when(pl.col("div_game")).then(pl.lit("div")).otherwise(pl.lit("non_div"))),
    Split("script", _script_bucket()),
)
SLOPES = ("market", "defense")


def _context_frame(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Team-games with the response, the buckets and the two continuous ratios, all mean-1."""
    tw = history.team_weeks(seasons)
    have = [m for m in CONTEXT_METRICS if m in tw.columns]
    df = tw.with_columns(
        pl.col("div_game").cast(pl.Boolean),
        pl.col("neutral_site").cast(pl.Boolean),
    ).with_columns(
        [
            (pl.col(m) / pl.col(m).mean().over(["season", "team"])).alias(f"rel_{m}")
            for m in have
        ]
        + [
            (pl.col("implied_points") / pl.col("implied_points").mean().over(["season", "team"]))
            .alias("x_market")
        ]
        + [s.bucket.alias(f"b_{s.name}") for s in SPLITS]
    )
    for s in SPLITS:
        if s.applies is not None:
            df = df.with_columns(
                pl.when(s.applies).then(pl.col(f"b_{s.name}")).otherwise(None).alias(f"b_{s.name}")
            )

    # The opponent's defensive rating, as a ratio to that season's league mean.
    dr = defense_ratings(seasons).select(
        "season", pl.col("team").alias("opponent"),
        *[pl.col(f"{f}__factor").alias(f"xdef_{f}") for f in DEF_RATINGS],
    )
    return df.join(dr, on=["season", "opponent"], how="left")


def _codes(values: pl.Series) -> tuple[np.ndarray, list[str]]:
    labels = sorted(v for v in values.unique().to_list() if v is not None)
    lookup = {v: i for i, v in enumerate(labels)}
    codes = np.array([lookup.get(v, -1) for v in values.to_list()], dtype=np.int64)
    return codes, labels


def _fit_one_metric(
    metric: str,
    rel: np.ndarray,
    split_codes: dict[str, tuple[np.ndarray, list[str]]],
    x_market: np.ndarray,
    x_def: np.ndarray | None,
    k: float,
    rounds: int = BACKFIT_ROUNDS,
    partial: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Backfit the chain for one metric. Returns bucket factors, slopes and bucket counts.

    `partial=False` estimates every term on `rel` itself instead of on the residual left by the
    others, which gives each term's *marginal* effect: "big favourites run 4.7% more than they
    average". That is the honest headline number and the one to show a user, but it is not what the
    chain can multiply -- the defence term already explains most of why favourites run more, since
    they are favourites against defences that let you run. The partial effects are what compose."""
    ok = np.isfinite(rel) & (rel >= 0)
    factors = {name: np.ones(len(labels)) for name, (_, labels) in split_codes.items()}
    counts = {name: np.zeros(len(labels)) for name, (_, labels) in split_codes.items()}
    slopes = {"market": 0.0, "defense": 0.0}
    xs = {"market": x_market, "defense": x_def}

    def term_factor(name: str) -> np.ndarray:
        if name in factors:
            codes, _ = split_codes[name]
            f = np.ones_like(rel)
            good = codes >= 0
            f[good] = factors[name][codes[good]]
            return f
        x = xs[name]
        if x is None:
            return np.ones_like(rel)
        f = 1.0 + slopes[name] * (np.nan_to_num(x, nan=1.0) - 1.0)
        return np.maximum(f, FACTOR_FLOOR)

    names = list(factors) + [s for s in SLOPES if xs[s] is not None]
    for _ in range(rounds):
        for name in names:
            other = np.ones_like(rel)
            for o in names if partial else ():
                if o != name:
                    other = other * term_factor(o)
            resid = np.where(ok & (other > 0), rel / np.where(other > 0, other, 1.0), np.nan)
            if name in factors:
                codes, labels = split_codes[name]
                for i in range(len(labels)):
                    sel = ok & (codes == i) & np.isfinite(resid)
                    n = int(sel.sum())
                    counts[name][i] = n
                    obs = float(resid[sel].mean()) if n else 1.0
                    factors[name][i] = (n * obs + k) / (n + k)      # shrink toward 1
                total = counts[name].sum()
                if total > 0:
                    level = float((counts[name] * factors[name]).sum() / total)
                    if level > 0:
                        factors[name] /= level
            else:
                x = xs[name]
                sel = ok & np.isfinite(resid) & np.isfinite(x)
                dx = x[sel] - 1.0
                denom = float((dx * dx).sum())
                g = float((dx * (resid[sel] - 1.0)).sum() / denom) if denom > 0 else 0.0
                slopes[name] = float(np.clip(g, *SLOPE_CLIP))
    return factors, slopes, counts


def _predict_rel(
    split_codes: dict[str, tuple[np.ndarray, list[str]]],
    factors: dict[str, np.ndarray],
    slopes: dict[str, float],
    x_market: np.ndarray,
    x_def: np.ndarray | None,
    n_rows: int,
) -> np.ndarray:
    pred = np.ones(n_rows)
    for name, f in factors.items():
        codes, _ = split_codes[name]
        good = codes >= 0
        step = np.ones(n_rows)
        step[good] = f[codes[good]]
        pred = pred * step
    for name, x in (("market", x_market), ("defense", x_def)):
        if x is None:
            continue
        pred = pred * np.maximum(1.0 + slopes[name] * (np.nan_to_num(x, nan=1.0) - 1.0), FACTOR_FLOOR)
    return pred


def fit_context(
    seasons: tuple[int, ...] | None = None,
    settings: Settings | None = None,
    holdout: Sequence[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit the per-game chain. Returns (factor table, per-metric accuracy).

    Accuracy is leave-one-season-out: for each season in `holdout` the chain is refitted without it
    and scored on it, against the null model of no context at all (every factor 1). That number is
    the only honest answer to "is this chain earning its complexity".
    """
    st = settings or Settings()
    df = _context_frame(seasons)
    holdout = list(holdout or range(2021, LAST_COMPLETE_SEASON + 1))
    season = df["season"].to_numpy()
    split_codes = {s.name: _codes(df[f"b_{s.name}"]) for s in SPLITS}
    x_market = df["x_market"].to_numpy()

    rows: list[dict] = []
    scores: list[dict] = []
    for metric in CONTEXT_METRICS:
        if f"rel_{metric}" not in df.columns:
            continue
        rel = df[f"rel_{metric}"].cast(pl.Float64).to_numpy()
        x_def = _def_x(lambda c: df[f"xdef_{c}"].cast(pl.Float64).to_numpy(), metric)
        factors, slopes, counts = _fit_one_metric(
            metric, rel, split_codes, x_market, x_def, st.context_k
        )
        marg_f, marg_s, _ = _fit_one_metric(
            metric, rel, split_codes, x_market, x_def, st.context_k, rounds=1, partial=False
        )
        for name, f in factors.items():
            _, labels = split_codes[name]
            for i, label in enumerate(labels):
                rows.append({"metric": metric, "family": FAMILY[metric], "kind": "split",
                             "term": name, "bucket": label, "value": float(f[i]),
                             "marginal": float(marg_f[name][i]), "n": int(counts[name][i])})
        for name in SLOPES:
            if name == "defense" and x_def is None:
                continue
            rows.append({"metric": metric, "family": FAMILY[metric], "kind": "slope",
                         "term": name,
                         "bucket": "*".join(DEF_OF[metric]) if name == "defense" else "implied_points",
                         "value": slopes[name], "marginal": marg_s[name],
                         "n": int(np.isfinite(rel).sum())})

        # leave-one-season-out score
        base_err, chain_err, n_eval = 0.0, 0.0, 0
        for s in holdout:
            train = season != s
            f_tr, g_tr, _ = _fit_one_metric(
                metric, np.where(train, rel, np.nan), split_codes, x_market, x_def, st.context_k
            )
            pred = _predict_rel(split_codes, f_tr, g_tr, x_market, x_def, len(rel))
            sel = (~train) & np.isfinite(rel)
            if not sel.any():
                continue
            base_err += float(np.abs(rel[sel] - 1.0).sum())
            chain_err += float(np.abs(rel[sel] - pred[sel]).sum())
            n_eval += int(sel.sum())
        if n_eval:
            scores.append({
                "metric": metric, "family": FAMILY[metric], "n": n_eval,
                "mae_no_context": base_err / n_eval, "mae_chain": chain_err / n_eval,
                "gain_pct": 100.0 * (base_err - chain_err) / base_err if base_err else 0.0,
                "market_slope": slopes["market"],
                "defense_slope": slopes["defense"] if x_def is not None else None,
            })
    # A node that loses out of sample does not get to play. Dropping a metric's rows leaves its chain
    # at a flat 1.0, so the projection falls back to the season estimate rather than to a factor that
    # made things worse -- and the scores table says which metrics those were.
    factors, scores = pl.DataFrame(rows), pl.DataFrame(scores)
    if not scores.is_empty():
        keep = set(scores.filter(pl.col("gain_pct") > 0)["metric"].to_list())
        scores = scores.with_columns(pl.col("metric").is_in(list(keep)).alias("context_used"))
        factors = factors.filter(pl.col("metric").is_in(list(keep)))
    return factors, scores.sort("gain_pct", descending=True)


@lru_cache(maxsize=1)
def load_factors() -> pl.DataFrame:
    if not FACTOR_PATH.exists():
        return pl.DataFrame(schema={"metric": pl.String, "family": pl.String, "kind": pl.String,
                                    "term": pl.String, "bucket": pl.String, "value": pl.Float64,
                                    "marginal": pl.Float64, "n": pl.Int64})
    return pl.read_parquet(FACTOR_PATH)


SCORE_PATH = FITTED / "team_context_scores.parquet"


@lru_cache(maxsize=1)
def load_context_scores() -> pl.DataFrame:
    """Per-metric out-of-sample gain from the chain, and whether it was kept."""
    return pl.read_parquet(SCORE_PATH) if SCORE_PATH.exists() else pl.DataFrame()


# --------------------------------------------------------------------------- #
# the market: how much of a game's scoring level comes from the posted line
# --------------------------------------------------------------------------- #
def home_points_factor(seasons: tuple[int, ...] | None = None) -> float:
    """Raw home/away points ratio, used only to build a model line before any factor is applied."""
    tw = history.team_weeks(seasons).filter(~pl.col("neutral_site").cast(pl.Boolean))
    home = tw.filter(pl.col("is_home"))["points"].mean()
    away = tw.filter(~pl.col("is_home"))["points"].mean()
    both = (home + away) / 2
    return float(home / both)


def model_lines(season: int, settings: Settings | None = None, rows: pl.DataFrame | None = None) -> pl.DataFrame:
    """Our own expected points per team-game, before the per-game chain.

    `own offence x opponent defence / league`, the standard multiplicative pairing, plus a flat home
    adjustment. This is what a game with no posted line falls back to, and what the posted line is
    blended with where there is one.
    """
    rows = game_rows(season) if rows is None else rows
    wide = shape_wide(season, settings)
    hf = home_points_factor()
    lg = float(wide["off_points_per_game"].mean())
    own = wide.select("team", pl.col("off_points_per_game").alias("own_ppg"),
                      pl.col("def_points_allowed_pg").alias("own_papg"))
    opp = wide.select(pl.col("team").alias("opponent"),
                      pl.col("off_points_per_game").alias("opp_ppg"),
                      pl.col("def_points_allowed_pg").alias("opp_papg"))
    out = rows.join(own, on="team", how="left").join(opp, on="opponent", how="left")
    side = pl.when(pl.col("neutral_site")).then(1.0).when(pl.col("is_home")).then(hf).otherwise(2.0 - hf)
    return out.with_columns(
        (pl.col("own_ppg") * pl.col("opp_papg") / lg * side).alias("model_points"),
        (pl.col("opp_ppg") * pl.col("own_papg") / lg * (2.0 - side)).alias("model_points_opp"),
    ).with_columns(
        (pl.col("model_points") + pl.col("model_points_opp")).alias("model_total"),
        (pl.col("model_points_opp") - pl.col("model_points")).alias("model_spread"),
    )


def fit_market(
    settings: Settings | None = None,
    targets: Sequence[int] | None = None,
) -> dict:
    """How much weight the posted line should carry against our own expected points.

    Both numbers are ex-ante in the sense that the model estimate comes only from seasons before the
    one being scored -- but a week-12 line is not ex-ante at all, because it knows weeks 1 to 11. A
    projection built in August has only preseason lines, so the weight that governs is the one fitted
    on **week 1** alone, where the market is as blind as we are. The all-weeks number is reported
    beside it precisely to show how much of the market's edge is in-season information.
    """
    targets = list(targets or range(FIT_FIRST_TARGET, LAST_COMPLETE_SEASON + 1))
    frames = []
    for s in targets:
        actual = history.team_weeks((s,)).select(
            "game_id", "team", "opponent", "week", "is_home", "neutral_site", "points",
            "implied_points", "spread_line", "total_line", "team_spread",
        )
        frames.append(model_lines(s, settings, rows=actual))
    df = pl.concat(frames).drop_nulls(["points", "implied_points", "model_points"])

    def _grid(d: pl.DataFrame) -> dict[float, float]:
        market, model, actual = (d["implied_points"].to_numpy(), d["model_points"].to_numpy(),
                                 d["points"].to_numpy())
        return {round(a, 2): float(np.abs(a * market + (1 - a) * model - actual).mean())
                for a in np.arange(0.0, 1.001, 0.05)}

    week1 = _grid(df.filter(pl.col("week") == 1))
    every = _grid(df)
    best1, best_all = min(week1, key=week1.get), min(every, key=every.get)
    return {
        "market_weight": best1, "mae": week1[best1],
        "mae_market_only": week1[1.0], "mae_model_only": week1[0.0],
        "n": int(df.filter(pl.col("week") == 1).height),
        "market_weight_all_weeks": best_all, "mae_all_weeks": every[best_all],
        "mae_market_only_all_weeks": every[1.0], "mae_model_only_all_weeks": every[0.0],
        "n_all_weeks": int(df.height),
        "seasons": [min(targets), max(targets)],
        "grid_week1": {str(k): v for k, v in week1.items()},
        "grid_all_weeks": {str(k): v for k, v in every.items()},
    }


@lru_cache(maxsize=1)
def load_market() -> dict:
    if not MARKET_PATH.exists():
        return {"market_weight": 0.5}
    return json.loads(MARKET_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the schedule, and what a game is expected to look like
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def climate(from_season: int = PROJ_SEASON - CLIMATE_WINDOW, to_season: int | None = None) -> pl.DataFrame:
    """Expected temperature and wind by home team and part of the season.

    2026's schedule has no weather in it and never will until kickoff, so an outdoor game gets the
    climate its venue has actually produced. Recent seasons only, so a team that moved indoors is not
    averaged with its own past. `to_season` is exclusive and is for the backtest: a held-out season's
    expected weather must not average in the weather that season actually got.

    Both ends move with the target rather than only the far one, which is what `game_rows` passes.
    Holding the near end at a fixed year left a 2021 fold with a window of nothing.
    """
    tw = history.team_weeks().filter(
        (pl.col("season") >= from_season) & pl.col("is_home") & ~pl.col("is_indoor")
    ).with_columns(_period())
    if to_season is not None:
        tw = tw.filter(pl.col("season") < to_season)
    by_period = tw.group_by("team", "period").agg(
        pl.col("temp").mean().alias("exp_temp"), pl.col("wind").mean().alias("exp_wind"),
        pl.len().alias("n"),
    )
    by_team = tw.group_by("team").agg(
        pl.col("temp").mean().alias("team_temp"), pl.col("wind").mean().alias("team_wind")
    )
    return by_period.join(by_team, on="team", how="left")


@lru_cache(maxsize=8)
def league_climate(from_season: int = PROJ_SEASON - CLIMATE_WINDOW, to_season: int | None = None) -> pl.DataFrame:
    """Outdoor temperature and wind by part of the season, league-wide.

    The last resort for a venue that has no outdoor history of its own: an outdoor game at a neutral
    site nominally hosted by a dome team. 2024 has one -- Minnesota against the Jets at Tottenham --
    and without this it goes into the chain with no weather at all, which silently means average.
    Saying average out loud is the same answer and a legible one.
    """
    tw = history.team_weeks().filter(
        (pl.col("season") >= from_season) & pl.col("is_home") & ~pl.col("is_indoor")
    ).with_columns(_period())
    if to_season is not None:
        tw = tw.filter(pl.col("season") < to_season)
    return tw.group_by("period").agg(
        pl.col("temp").mean().alias("lg_temp"), pl.col("wind").mean().alias("lg_wind")
    )


def _period() -> pl.Expr:
    w = pl.col("week")
    return (
        pl.when(w <= 4).then(pl.lit("early"))
        .when(w <= 9).then(pl.lit("mid"))
        .when(w <= 13).then(pl.lit("late"))
        .otherwise(pl.lit("winter"))
        .alias("period")
    )


@lru_cache(maxsize=4)
def _roof_by_team() -> dict[str, str]:
    """Most recent known roof at each team's home venue, for the retractables the 2026 file leaves
    null (ARI, ATL, DAL, HOU, IND -- all of which play closed by default)."""
    tw = history.team_weeks().filter(pl.col("is_home") & pl.col("roof").is_not_null())
    last = tw.sort("season", "week").group_by("team").agg(pl.col("roof").last().alias("roof"))
    return {r["team"]: r["roof"] for r in last.iter_rows(named=True)}


@lru_cache(maxsize=8)
def game_rows(season: int = PROJ_SEASON, expected_weather: bool = False) -> pl.DataFrame:
    """Two rows per game -- one per team -- with every context column the chain reads.

    `expected_weather` discards the temperature and wind the schedule file records and uses the
    venue's climate instead. For 2026 there is nothing to discard, so it changes nothing; for a
    held-out season it is the difference between projecting a game and remembering it.
    """
    sched = lake.read("schedules", layer="raw", seasons=(season,)).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    base = [
        "game_id", "season", "week", "gameday", "roof", "surface", "temp", "wind",
        "div_game", "spread_line", "total_line", "location", "stadium",
    ]
    home = sched.select(*base, pl.col("home_team").replace(TEAM_FIXUP).alias("team"),
                        pl.col("away_team").replace(TEAM_FIXUP).alias("opponent"),
                        pl.col("home_rest").alias("rest_days"), pl.lit(True).alias("is_home"))
    away = sched.select(*base, pl.col("away_team").replace(TEAM_FIXUP).alias("team"),
                        pl.col("home_team").replace(TEAM_FIXUP).alias("opponent"),
                        pl.col("away_rest").alias("rest_days"), pl.lit(False).alias("is_home"))
    rows = pl.concat([home, away]).with_columns(
        pl.col("div_game").cast(pl.Boolean),
        (pl.col("location").str.to_lowercase() != "home").fill_null(False).alias("neutral_site"),
        # the home team owns the venue, so its roof fills the away row too
        pl.when(pl.col("is_home")).then(pl.col("team")).otherwise(pl.col("opponent")).alias("host"),
    )
    rows = rows.with_columns(
        pl.col("roof")
        .fill_null(pl.col("host").replace_strict(
            _roof_by_team(), default="outdoors", return_dtype=pl.String))
        .alias("roof")
    ).with_columns(
        pl.col("roof").is_in(["dome", "closed"]).alias("is_indoor"),
        # the team's own side of the posted line; negative means favoured
        pl.when(pl.col("spread_line").is_null()).then(None)
        .when(pl.col("is_home")).then(-pl.col("spread_line"))
        .otherwise(pl.col("spread_line")).alias("market_spread"),
    ).with_columns(
        pl.when(pl.col("total_line").is_null()).then(None)
        .otherwise(pl.col("total_line") / 2 - pl.col("market_spread") / 2)
        .alias("market_points")
    )

    # expected weather where the game is outdoors and the file has none
    if expected_weather:
        rows = rows.with_columns(
            pl.lit(None, pl.Float64).alias("temp"), pl.lit(None, pl.Float64).alias("wind")
        )
    to = season if expected_weather else None
    first = max(HISTORY_FROM, season - CLIMATE_WINDOW)
    cl = climate(first, to).rename({"team": "host"})
    rows = (
        rows.with_columns(_period())
        .join(cl, on=["host", "period"], how="left")
        .join(league_climate(first, to), on="period", how="left")
    )
    return rows.with_columns(
        pl.when(pl.col("is_indoor")).then(None)
        .otherwise(pl.col("temp").cast(pl.Float64).fill_null(pl.col("exp_temp"))
                   .fill_null(pl.col("team_temp")).fill_null(pl.col("lg_temp"))).alias("temp"),
        pl.when(pl.col("is_indoor")).then(None)
        .otherwise(pl.col("wind").cast(pl.Float64).fill_null(pl.col("exp_wind"))
                   .fill_null(pl.col("team_wind")).fill_null(pl.col("lg_wind"))).alias("wind"),
    ).drop("exp_temp", "exp_wind", "team_temp", "team_wind", "lg_temp", "lg_wind", "n") \
        .sort("week", "game_id", "is_home")


ROW_SCHEMA = (
    "game_id", "season", "week", "team", "opponent", "is_home", "neutral_site", "rest_days",
    "roof", "surface", "is_indoor", "temp", "wind", "div_game", "market_spread", "market_points",
    "total_line",
)


def historical_rows(season: int) -> pl.DataFrame:
    """Played games in the same shape `game_rows` produces, so the chain can be scored on them.

    Only context columns come across. The actual results are carried with an `a_` prefix, which keeps
    them impossible to mistake for an input.
    """
    tw = history.team_weeks((season,))
    metrics = [m for m in CONTEXT_METRICS if m in tw.columns]
    return tw.select(
        "game_id", "season", "week", "team", "opponent", "is_home",
        pl.col("neutral_site").cast(pl.Boolean),
        "rest_days", "roof", "surface", "is_indoor",
        pl.col("temp").cast(pl.Float64), pl.col("wind").cast(pl.Float64),
        pl.col("div_game").cast(pl.Boolean),
        pl.col("team_spread").alias("market_spread"),
        pl.col("implied_points").alias("market_points"),
        "total_line",
        *[pl.col(m).alias(f"a_{m}") for m in metrics],
    )


def _factor_maps(factors: pl.DataFrame) -> tuple[dict, dict]:
    splits: dict[tuple[str, str], dict[str, float]] = {}
    slopes: dict[tuple[str, str], float] = {}
    for r in factors.iter_rows(named=True):
        if r["kind"] == "split":
            splits.setdefault((r["metric"], r["term"]), {})[r["bucket"]] = r["value"]
        else:
            slopes[(r["metric"], r["term"])] = r["value"]
    return splits, slopes


def game_environment(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    rows: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (game_id, team) for `season`: context, the game's line, and projected volume.

    The chain is applied to the stage-1 season estimate, so a column here is directly comparable to
    the season number it came from -- `plays` in week 12 against `off_plays_per_game`.

    `rows` defaults to the real schedule; passing `historical_rows(season)` projects a season that has
    already been played, which is how the chain gets scored.
    """
    st = settings or Settings()
    rows = model_lines(season, st, rows=rows)
    wide = shape_wide(season, st)
    factors = load_factors()
    splits, slopes = _factor_maps(factors)

    # scoring level: the posted line where there is one, our own estimate where there is not
    alpha = st.market_weight if st.market_weight is not None else float(load_market()["market_weight"])

    # Half of 2026's games have a line and half do not, so the two halves have to be put on the same
    # scale before they sit in one season. Where a team has lines, the average gap between the posted
    # number and ours is that team's level correction, and it carries into its unlined games -- shrunk
    # toward zero by n/(n+MARKET_OFFSET_K) because it is estimated from a handful of games.
    # `market_weight = 0` means the market is not an input, and that has to include the level
    # correction: otherwise a backtest that switched the line off would still be reading the closing
    # number through the offset.
    rows = rows.with_columns(
        (pl.col("market_points") - pl.col("model_points")).alias("_gap")
        if alpha > 0.0 else pl.lit(None, pl.Float64).alias("_gap")
    ).with_columns(
        (pl.col("_gap").mean().over("team").fill_null(0.0)
         * (pl.col("_gap").count().over("team")
            / (pl.col("_gap").count().over("team") + MARKET_OFFSET_K)))
        .alias("market_offset")
    ).with_columns(
        (pl.col("model_points") + pl.col("market_offset")).alias("model_points"),
        (pl.col("model_total") + pl.col("market_offset")
         + pl.col("market_offset").mean().over("opponent").fill_null(0.0)).alias("model_total"),
    ).drop("_gap")

    rows = rows.with_columns(
        pl.when(pl.col("market_points").is_not_null())
        .then(alpha * pl.col("market_points") + (1 - alpha) * pl.col("model_points"))
        .otherwise(pl.col("model_points")).alias("implied_points"),
        pl.when(pl.col("market_spread").is_not_null())
        .then(alpha * pl.col("market_spread") + (1 - alpha) * pl.col("model_spread"))
        .otherwise(pl.col("model_spread")).alias("spread"),
        pl.when(pl.col("total_line").is_not_null())
        .then(alpha * pl.col("total_line") + (1 - alpha) * pl.col("model_total"))
        .otherwise(pl.col("model_total")).alias("total"),
        pl.col("market_points").is_not_null().alias("has_market"),
    ).with_columns(
        _script_bucket("spread").alias("b_script"),
    ).with_columns(
        # The market term was fitted as a game's implied points over that team-season's own mean, so
        # it is applied the same way: mean 1 *within a team*. That is a deliberate division of labour.
        # Level -- how good a team is, and how easy its schedule is -- belongs to the season estimate
        # and to the defensive term, whose ratios are already league-normalised and so carry an
        # unbalanced schedule without bias. The market's job here is to say which of a team's 17 games
        # is the shootout. Letting it carry level too made things measurably worse: an estimate of a
        # team's own scoring level that sits a point off the market's shifts all 17 of its games in
        # the same direction, which is exactly the error a factor should not introduce.
        (pl.col("implied_points") / pl.col("implied_points").mean().over("team")).alias("x_market"),
    )
    # bucket labels, using the blended line for script
    for s in SPLITS:
        if s.name == "script":
            continue
        rows = rows.with_columns(s.bucket.alias(f"b_{s.name}"))
        if s.applies is not None:
            rows = rows.with_columns(
                pl.when(s.applies).then(pl.col(f"b_{s.name}")).otherwise(None).alias(f"b_{s.name}")
            )

    # opponent defensive ratings from stage 1, as ratios to the projected league mean
    lg = {c: float(wide[f"def_{c}"].mean()) for c in DEF_RATINGS}
    opp_def = wide.select(
        pl.col("team").alias("opponent"),
        *[(pl.col(f"def_{c}") / lg[c]).alias(f"xdef_{c}") for c in DEF_RATINGS],
    )
    rows = rows.join(opp_def, on="opponent", how="left")

    est = {m: wide.select("team", pl.col(f"off_{SHAPE_OF[m]}").alias(f"est_{m}"))
           for m in CONTEXT_METRICS if f"off_{SHAPE_OF[m]}" in wide.columns}
    for frame in est.values():
        rows = rows.join(frame, on="team", how="left")

    for metric in est:
        chain = pl.lit(1.0)
        if st.use_context_factors:
            for s in SPLITS:
                mapping = splits.get((metric, s.name))
                if not mapping:
                    continue
                chain = chain * pl.col(f"b_{s.name}").replace_strict(
                    mapping, default=1.0, return_dtype=pl.Float64
                ).fill_null(1.0)
            g_mkt = slopes.get((metric, "market"))
            if g_mkt is not None:
                chain = chain * pl.max_horizontal(
                    1.0 + g_mkt * (pl.col("x_market").fill_null(1.0) - 1.0), pl.lit(FACTOR_FLOOR)
                )
            dcols = DEF_OF.get(metric) or ()
            g_def = slopes.get((metric, "defense"))
            if dcols and g_def is not None:
                x_def = pl.lit(1.0)
                for c in dcols:
                    x_def = x_def * pl.col(f"xdef_{c}").fill_null(1.0)
                chain = chain * pl.max_horizontal(
                    1.0 + g_def * (x_def - 1.0), pl.lit(FACTOR_FLOOR)
                )
        rows = rows.with_columns(chain.alias(f"f_{metric}"))

    if st.schedule_renormalise:
        rows = rows.with_columns(
            [(pl.col(f"f_{m}") / pl.col(f"f_{m}").mean().over("team")).alias(f"f_{m}") for m in est]
        )

    rows = rows.with_columns(
        [(pl.col(f"est_{m}") * pl.col(f"f_{m}")).alias(m) for m in est]
    )
    # Points is the one metric where a better answer than the chain exists: the blended line. Measured
    # on week-1 games -- the only ones where the posted line is as blind as a preseason projection --
    # the blend is the best of the three (6.83 MAE against 7.04 for the line alone and 7.11 for the
    # chain). The chain's own number stays beside it under its own name, because the TD and red-zone
    # metrics *are* chain outputs and a user comparing them is entitled to a like-for-like column.
    return rows.rename({"points": "points_chain"}).with_columns(
        pl.col("implied_points").alias("points")
    ).sort("week", "game_id", pl.col("is_home"), descending=[False, False, True])


def score_environment(
    targets: Sequence[int] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Score the two stages on played games: does the per-game chain beat the flat season number?

    Three columns to compare per metric -- `mae_flat` is the season estimate repeated 17 times, which
    is all the workbook this replaces could do; `mae_chain` is that estimate times the per-game chain;
    `mae_market` exists for points only and is the posted line taken at face value.

    The season estimate is honest here (it only ever reads seasons before the one being scored). The
    factors are fitted on all seasons, so `mae_chain` flatters itself slightly; `fit_context`'s
    leave-one-season-out gain is the leak-free version of the same question.
    """
    st = settings or Settings()
    targets = list(targets or range(2021, LAST_COMPLETE_SEASON + 1))
    env = pl.concat(
        [game_environment(s, st, rows=historical_rows(s)) for s in targets], how="diagonal"
    )
    out = []
    for m in CONTEXT_METRICS:
        if f"a_{m}" not in env.columns or m not in env.columns:
            continue
        d = env.select(
            pl.col(f"a_{m}").cast(pl.Float64).alias("a"),
            pl.col(m).alias("chain"),
            pl.col(f"est_{m}").alias("flat"),
            pl.col("implied_points").alias("market"),
        ).drop_nulls(["a", "chain", "flat"])
        if d.is_empty():
            continue
        flat = float((d["a"] - d["flat"]).abs().mean())
        chain = float((d["a"] - d["chain"]).abs().mean())
        rec = {
            "metric": m, "family": FAMILY[m], "n": d.height,
            "mae_flat": flat, "mae_chain": chain,
            "gain_pct": 100.0 * (flat - chain) / flat if flat else 0.0,
        }
        rec["mae_market"] = float((d["a"] - d["market"]).abs().mean()) if m == "points" else None
        out.append(rec)
    return pl.DataFrame(out).sort("gain_pct", descending=True)


def season_environment(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """The 17 games summed back to a season, which is what the player layer takes a share of.

    Volume metrics sum; rates are re-derived from the summed counts, never averaged. `*_flat` is what
    the season estimate alone would have said, so the schedule's contribution is visible.
    """
    env = game_environment(season, settings)
    counts = [m for m in CONTEXT_METRICS if FAMILY[m] in ("pace", "scoring")
              and m in env.columns and m not in ("seconds_per_play", "plays_per_drive")]
    counts += [m for m in ("dropbacks", "pass_attempts", "carries", "targets", "air_yards",
                           "late_down_targets") if m in env.columns]
    out = env.group_by("team").agg(
        pl.len().alias("games"),
        *[pl.col(m).sum().alias(m) for m in counts],
        *[pl.col(f"est_{m}").sum().alias(f"{m}_flat") for m in counts],
        pl.col("implied_points").mean().alias("implied_points_per_game"),
        pl.col("f_points").mean().alias("schedule_factor"),
        pl.col("has_market").sum().alias("games_with_line"),
    )
    return out.with_columns(
        (pl.col("pass_attempts") / (pl.col("pass_attempts") + pl.col("carries"))).alias("pass_rate"),
        (pl.col("dropbacks") / pl.col("plays")).alias("dropback_rate"),
        (pl.col("targets") / pl.col("pass_attempts")).alias("targets_per_attempt"),
        (pl.col("points") / pl.col("points_flat") - 1).alias("schedule_points_pct"),
    ).sort("points", descending=True)


# --------------------------------------------------------------------------- #
# fit + report
# --------------------------------------------------------------------------- #
def clear_cache() -> None:
    for fn in (offense_seasons, defense_ratings, shape_panel, climate, game_rows,
               load_shape_fit, load_factors, load_context_scores, load_market, _roof_by_team):
        fn.cache_clear()


def fit_all(settings: Settings | None = None) -> dict:
    ensure_dirs()
    st = settings or Settings()
    shape = fit_shape()
    SHAPE_PATH.write_text(json.dumps(shape, indent=2, sort_keys=True), encoding="utf-8")
    load_shape_fit.cache_clear()

    factors, scores = fit_context(settings=st)
    factors.write_parquet(FACTOR_PATH)
    scores.write_parquet(SCORE_PATH)
    load_factors.cache_clear()
    load_context_scores.cache_clear()

    market = fit_market(st)
    MARKET_PATH.write_text(json.dumps(market, indent=2, sort_keys=True), encoding="utf-8")
    load_market.cache_clear()
    return {"shape": shape, "factors": factors, "scores": scores, "market": market}


def _report(art: dict | None = None) -> None:
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(60)
    shape = art["shape"] if art else load_shape_fit()
    if shape:
        sh = pl.DataFrame(list(shape.values())).select(
            "side", "metric", "w", "keep", "mae_loso", "mae_copy_last", "gain_vs_copy_last_pct",
            "gain_vs_mean_pct", "reverted_to_league_mean", "league_mean", "spread",
        ).sort("gain_vs_copy_last_pct", descending=True)
        print(f"\n== season shape, {FIT_FIRST_TARGET}-{LAST_COMPLETE_SEASON} out of sample "
              f"({sh.height} metrics)")
        print(sh)
        print(f"mean gain vs copying last season: {sh['gain_vs_copy_last_pct'].mean():.1f}%   "
              f"metrics carrying no form forward: "
              f"{sh.filter('reverted_to_league_mean').height}")

    scores = art["scores"] if art else load_context_scores()
    if not scores.is_empty():
        print("\n== per-game context, leave-one-season-out 2021-2025 (error in ratio-to-own-mean units)")
        print(scores)
        print(f"mean gain from context: {scores['gain_pct'].mean():.1f}%")

    factors = art["factors"] if art else load_factors()
    if not factors.is_empty():
        print("\n== context factors: `value` is the partial effect the chain multiplies, "
              "`marg` the raw split")
        show = factors.filter(pl.col("metric").is_in(["points", "carries", "pass_attempts"]))
        print(show.with_columns(pl.col("marginal").round(4), pl.col("value").round(4))
              .pivot(on="metric", index=["kind", "term", "bucket"], values=["value", "marginal"])
              .rename(lambda c: c.replace("marginal_", "marg_").replace("value_", ""))
              .sort("kind", "term", "bucket"))

    market = art["market"] if art else load_market()
    if "mae" in market:
        print(f"\n== market blend, week 1 only (the preseason line, which is what 2026 has): "
              f"weight {market['market_weight']:.2f}  MAE {market['mae']:.3f} pts/game  "
              f"(line only {market['mae_market_only']:.3f}, model only "
              f"{market['mae_model_only']:.3f}, n={market['n']})")
        print(f"   all weeks, where the line knows the season so far: "
              f"weight {market['market_weight_all_weeks']:.2f}  "
              f"MAE {market['mae_all_weeks']:.3f}  (line only "
              f"{market['mae_market_only_all_weeks']:.3f}, model only "
              f"{market['mae_model_only_all_weeks']:.3f}, n={market['n_all_weeks']})")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Fit and report the team environment.")
    ap.add_argument("--fit", action="store_true", help="refit and write data/fitted artifacts")
    ap.add_argument("--report", action="store_true", help="print the fitted numbers")
    ap.add_argument("--season", type=int, default=PROJ_SEASON)
    args = ap.parse_args(argv)

    art = fit_all() if args.fit else None
    if args.fit or args.report:
        _report(art)
    if not args.fit and not args.report:
        args.report = True
        _report(None)

    pl.Config.set_tbl_width_chars(240)
    env = game_environment(args.season)
    se = season_environment(args.season)
    print(f"\n== {args.season} environment: {env.height} team-games, "
          f"{env['team'].n_unique()} teams, {env.filter(pl.col('has_market')).height} with a line")
    print(env.head(6).select("week", "game_id", "team", "opponent", "is_home", "rest_days", "roof",
                             "temp", "wind", "spread", "total", "implied_points", "plays",
                             "pass_attempts", "carries", "targets", "points", "offensive_tds"))
    print(f"\n== {args.season} season roll-up, top 8 by projected points")
    print(se.head(8).select("team", "games", "plays", "pass_attempts", "carries", "targets",
                            "points", "offensive_tds", "pass_rate", "schedule_factor",
                            "schedule_points_pct", "games_with_line"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
