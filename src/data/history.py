"""Observed history, rolled up to the grains the engine projects from.

Three grains, each with a weekly and a season form:

- **team** -- what an offence did per game: pace, dropback rate, PROE, points, situational volume.
- **skill** -- RB/WR/TE usage from `player_usage`, which already carries the team denominator on
  every row. That is what makes a share *opportunity-weighted*: summing numerator and denominator
  over the games a player actually played means a missed game never reads as a usage drop, which is
  the one convention worth carrying over from the workbook this replaces.
- **quarterback** -- `player_games` for the box score, `passer_games` for the parts of a QB's
  rushing that a single "rush share" cannot express: designed runs and scrambles are different
  football and project differently.

Plus **defence faced**, derived by grouping the same team rows on `opponent`. No separate defensive
dataset is needed: every team-game row is also a row about the defence that was on the other side.

Weekly frames exist because dispersion has to be measured, not assumed -- the simulation's shock
sizes come from real week-to-week spread in these tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import polars as pl

from src.config import LAST_COMPLETE_SEASON, Scoring
from src.data import lake

# A share is only meaningful where the denominator exists; these are the pools we divide into.
TEAM_POOLS = {
    "target_share": ("targets", "team_targets"),
    "carry_share": ("carries", "team_carries"),
    "air_yards_share": ("receiving_air_yards", "team_air_yards"),
    "rz_target_share": ("rz_targets", "team_rz_targets"),
    "rz_carry_share": ("rz_carries", "team_rz_carries"),
    "inside_5_carry_share": ("inside_5_carries", "team_inside_5_carries"),
    "short_yardage_carry_share": ("short_yardage_carries", "team_short_yardage_carries"),
    "late_down_target_share": ("late_down_targets", "team_late_down_targets"),
    "rec_td_share": ("receiving_tds", "team_receiving_tds"),
    "rush_td_share": ("rushing_tds", "team_rushing_tds"),
}

# Descriptive shares, not projection inputs: yards are a product of volume and efficiency, so
# projecting a yard share directly would double-count. These exist to describe a season.
YARD_SHARES = {
    "rec_yards_share": ("receiving_yards", "team_receiving_yards"),
    "rush_yards_share": ("rushing_yards", "team_rushing_yards"),
}

# Participation is not a pool -- several players are on the field at once, so these sum past 1.
PARTICIPATION = {
    "snap_share": ("offense_snaps", "team_offense_snaps"),
    "route_participation": ("routes", "team_dropbacks"),
    "rush_participation": ("rush_plays", "team_designed_rushes"),
    "late_down_route_participation": ("late_down_routes", "team_late_down_dropbacks"),
    "two_minute_route_participation": ("two_minute_routes", "team_two_minute_dropbacks"),
}

# Per-opportunity efficiency: (numerator, denominator).
SKILL_RATES = {
    "catch_rate": ("receptions", "targets"),
    "yards_per_reception": ("receiving_yards", "receptions"),
    "yards_per_target": ("receiving_yards", "targets"),
    "yards_per_carry": ("rushing_yards", "carries"),
    "adot": ("receiving_air_yards", "targets"),
    "tprr": ("targets", "routes"),
    "yards_per_route": ("receiving_yards", "routes"),
    "rush_success_rate": ("rush_successes", "carries"),
    "targets_per_snap": ("targets", "offense_snaps"),
}

QB_RATES = {
    "completion_pct": ("completions", "attempts"),
    "yards_per_attempt": ("passing_yards", "attempts"),
    "pass_td_rate": ("passing_tds", "attempts"),
    "int_rate": ("interceptions", "attempts"),
    "sack_rate": ("sacks_suffered", "dropbacks"),
    "scramble_rate": ("scrambles", "dropbacks"),
    "attempt_rate": ("attempts", "dropbacks"),
    "qb_yards_per_carry": ("rushing_yards", "carries"),
    "designed_rush_share": ("designed_qb_rushes", "team_designed_rushes"),
    "qb_carry_share": ("carries", "team_carries"),
    "dropback_share": ("dropbacks", "team_dropbacks"),
    "air_yards_per_attempt": ("passing_air_yards", "attempts"),
}


def _rate(num: str, den: str, alias: str) -> pl.Expr:
    """num/den, null where the denominator is absent. A rate over zero opportunities is unknown,
    not zero -- filling it with zero is what drags a backup's efficiency into a starter's prior."""
    return (
        pl.when(pl.col(den) > 0)
        .then(pl.col(num) / pl.col(den))
        .otherwise(None)
        .alias(alias)
    )


def _with_rates(df: pl.DataFrame, specs: dict[str, tuple[str, str]]) -> pl.DataFrame:
    have = set(df.columns)
    exprs = [_rate(n, d, alias) for alias, (n, d) in specs.items() if {n, d} <= have]
    return df.with_columns(exprs) if exprs else df


def fantasy_points(columns: Iterable[str], scoring: Scoring | None = None) -> pl.Expr:
    """Fantasy points from components, so scoring stays a setting rather than a stored column.

    Columns absent from the frame contribute nothing -- a receiver's table has no `interceptions`
    and asking for one would fail rather than score zero.
    """
    s = scoring or Scoring()
    have = set(columns)
    weights = {
        "passing_yards": s.pass_yard, "passing_tds": s.pass_td, "interceptions": s.interception,
        "completions": s.completion, "rushing_yards": s.rush_yard, "rushing_tds": s.rush_td,
        "receiving_yards": s.rec_yard, "receiving_tds": s.rec_td, "receptions": s.reception,
        "rushing_fumbles_lost": s.fumble_lost, "receiving_fumbles_lost": s.fumble_lost,
    }
    # A projection carries one `fumbles_lost` column where history carries two; counting both would
    # double the penalty, so the combined column only scores when the components are absent.
    if "fumbles_lost" in have and not ({"rushing_fumbles_lost", "receiving_fumbles_lost"} & have):
        weights["fumbles_lost"] = s.fumble_lost
    terms = [
        pl.col(c).fill_null(0.0) * w for c, w in weights.items() if c in have and w
    ]
    expr = terms[0] if terms else pl.lit(0.0)
    for term in terms[1:]:
        expr = expr + term
    return expr.alias("fantasy_points")


# --------------------------------------------------------------------------- #
# team
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def team_weeks(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One row per team-game: volume, pace, efficiency and the game's own context."""
    seasons = seasons or lake.history_seasons()
    tg = lake.regular_season(lake.read("team_games", seasons=seasons))
    return tg.with_columns(
        _rate("pass_attempts", "plays", "attempt_rate_of_plays"),
        _rate("targets", "pass_attempts", "targets_per_attempt"),
        _rate("pass_tds", "offensive_tds", "pass_td_share_of_off"),
        _rate("inside_5_carries", "carries", "inside_5_rate"),
        _rate("red_zone_trips", "drives", "rz_trips_per_drive"),
        (pl.col("pass_attempts") + pl.col("carries")).alias("volume_plays"),
    )


@lru_cache(maxsize=8)
def team_seasons(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Team-season totals plus the per-game shape the projection starts from."""
    tw = team_weeks(seasons)
    sums = [
        "plays", "volume_plays", "dropbacks", "designed_rushes", "scrambles", "pass_attempts",
        "completions", "carries", "targets", "receptions", "pass_yards", "rush_yards", "yards",
        "pass_tds", "rush_tds", "offensive_tds", "interceptions", "sacks", "air_yards", "points",
        "points_allowed", "drives", "red_zone_trips", "red_zone_targets", "red_zone_carries",
        "inside_5_carries", "inside_5_targets", "short_yardage_carries", "late_down_targets",
    ]
    means = [
        "epa_per_play", "epa_per_dropback", "epa_per_rush", "success_rate", "proe",
        "neutral_pass_rate", "seconds_per_play", "seconds_per_play_neutral", "plays_per_drive",
        "explosive_pass_rate", "explosive_rush_rate", "red_zone_rate",
    ]
    have_s = [c for c in sums if c in tw.columns]
    have_m = [c for c in means if c in tw.columns]
    agg = tw.group_by(["season", "team"]).agg(
        pl.len().alias("games"),
        *[pl.col(c).sum().alias(c) for c in have_s],
        *[pl.col(c).mean().alias(c) for c in have_m],
    )
    per_game = {
        "plays_per_game": "plays", "volume_plays_per_game": "volume_plays",
        "dropbacks_per_game": "dropbacks", "pass_att_per_game": "pass_attempts",
        "rush_att_per_game": "carries", "targets_per_game": "targets",
        "points_per_game": "points", "off_td_per_game": "offensive_tds",
        "pass_td_per_game": "pass_tds", "rush_td_per_game": "rush_tds",
        "drives_per_game": "drives", "rz_trips_per_game": "red_zone_trips",
        "inside_5_per_game": "inside_5_carries",
        "short_yardage_per_game": "short_yardage_carries",
        "points_allowed_per_game": "points_allowed",
    }
    return agg.with_columns(
        [(pl.col(src) / pl.col("games")).alias(dst) for dst, src in per_game.items() if src in agg.columns]
    ).with_columns(
        _rate("pass_attempts", "volume_plays", "pass_rate"),
        _rate("dropbacks", "plays", "dropback_rate"),
        _rate("pass_tds", "offensive_tds", "pass_td_share_of_off"),
        _rate("targets", "pass_attempts", "targets_per_attempt"),
        _rate("pass_yards", "pass_attempts", "yards_per_attempt"),
        _rate("rush_yards", "carries", "yards_per_carry"),
        _rate("offensive_tds", "red_zone_trips", "td_per_rz_trip"),
    )


# --------------------------------------------------------------------------- #
# skill players
# --------------------------------------------------------------------------- #
def _position(col: str = "position") -> pl.Expr:
    """FB is RB. The depth charts separate them; the carries do not."""
    return pl.col(col).replace({"FB": "RB"})


@lru_cache(maxsize=8)
def skill_weeks(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One row per RB/WR/TE game, usage and team denominators together."""
    seasons = seasons or lake.history_seasons()
    us = lake.regular_season(lake.read("player_usage", seasons=seasons))
    us = us.filter(_position().is_in(["RB", "WR", "TE"])).with_columns(_position().alias("position"))

    # `plays`, `dropbacks` and `designed_rushes` in this table are the *team's* counts, repeated on
    # every player row -- they are the denominators behind route and rush participation. Prefixing
    # them keeps the one rule that makes the rest of this module readable: team-level columns say so.
    us = us.drop("dropbacks").rename({"plays": "team_plays", "designed_rushes": "team_designed_rushes"})

    pg = lake.regular_season(lake.read("player_games", seasons=seasons)).select(
        "game_id", "player_id", "rushing_fumbles_lost", "receiving_fumbles_lost",
        "receiving_first_downs", "rushing_first_downs", "receiving_epa", "rushing_epa",
        "receiving_yac",
    )
    us = us.join(pg, on=["game_id", "player_id"], how="left")

    # A row with no snap, no route and no touch is a player who did not play; keeping it would
    # dilute every per-game average with games he was not part of.
    us = us.filter(
        (pl.col("targets").fill_null(0) > 0)
        | (pl.col("carries").fill_null(0) > 0)
        | (pl.col("offense_snaps").fill_null(0) > 0)
        | (pl.col("routes").fill_null(0) > 0)
    )
    return _with_rates(us, SKILL_RATES).with_columns(fantasy_points(us.columns))


@lru_cache(maxsize=8)
def skill_seasons(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Player-season sums, then shares and rates computed from the sums.

    Order matters: a share of a season is the ratio of the two totals, never the average of weekly
    ratios. The latter weights a three-target game the same as a twelve-target one.
    """
    sw = skill_weeks(seasons)
    counts = [
        "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
        "carries", "rushing_yards", "rushing_tds", "routes", "offense_snaps", "rush_plays",
        "rz_targets", "rz_carries", "inside_5_carries", "inside_5_targets",
        "short_yardage_carries", "late_down_targets", "late_down_routes", "two_minute_routes",
        "rush_successes", "receiving_first_downs", "rushing_first_downs",
        "rushing_fumbles_lost", "receiving_fumbles_lost", "fantasy_points",
        "team_targets", "team_carries", "team_air_yards", "team_receiving_tds",
        "team_rushing_tds", "team_offense_snaps", "team_dropbacks", "team_designed_rushes",
        "team_rz_targets", "team_rz_carries", "team_inside_5_carries", "team_inside_5_targets",
        "team_short_yardage_carries", "team_late_down_targets", "team_late_down_dropbacks",
        "team_two_minute_dropbacks", "team_plays", "team_receiving_yards", "team_rushing_yards",
    ]
    have = [c for c in counts if c in sw.columns]
    agg = sw.group_by(["player_id", "season"]).agg(
        pl.col("player_name").drop_nulls().mode().first().alias("player"),
        pl.col("position").drop_nulls().mode().first().alias("position"),
        pl.col("team").drop_nulls().mode().first().alias("team"),
        pl.col("game_id").n_unique().alias("games"),
        *[pl.col(c).sum().alias(c) for c in have],
    )
    return _with_rates(agg, {**TEAM_POOLS, **YARD_SHARES, **PARTICIPATION, **SKILL_RATES}).with_columns(
        (pl.col("fantasy_points") / pl.col("games")).alias("fantasy_points_per_game"),
        (pl.col("carries") + pl.col("receptions")).alias("touches"),
        # both kinds together, because the scoring rule does not care how he lost it
        (pl.col("rushing_fumbles_lost") + pl.col("receiving_fumbles_lost")).alias("fumbles_lost"),
        (1.5 * pl.col("target_share").fill_null(0) + 0.7 * pl.col("air_yards_share").fill_null(0))
        .alias("wopr"),
    )


# --------------------------------------------------------------------------- #
# quarterbacks
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def qb_weeks(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One row per QB game: box score, dropback detail, and the team totals to take a share of."""
    seasons = seasons or lake.history_seasons()
    pg = lake.regular_season(lake.read("player_games", seasons=seasons)).filter(
        (pl.col("position") == "QB")
        & ((pl.col("attempts").fill_null(0) > 0) | (pl.col("carries").fill_null(0) > 0))
    )
    passer = lake.regular_season(lake.read("passer_games", seasons=seasons)).select(
        "game_id", "player_id", "dropbacks", "scrambles", "sacks_taken", "scramble_yards",
        "designed_qb_rushes", "designed_qb_rush_yards", "neutral_dropbacks", "neutral_scrambles",
        "qb_rushes", "qb_rush_yards", "kneels", "spikes",
    )
    tw = team_weeks(seasons).select(
        "game_id", "team", "opponent",
        pl.col("pass_attempts").alias("team_pass_attempts"),
        pl.col("dropbacks").alias("team_dropbacks"),
        pl.col("carries").alias("team_carries"),
        pl.col("designed_rushes").alias("team_designed_rushes"),
        pl.col("pass_tds").alias("team_pass_tds"),
        pl.col("rush_tds").alias("team_rushing_tds"),
        pl.col("offensive_tds").alias("team_offensive_tds"),
        pl.col("plays").alias("team_plays"),
    )
    out = pg.join(passer, on=["game_id", "player_id"], how="left").join(
        tw, on=["game_id", "team"], how="left"
    )
    # dropbacks is the denominator for sack and scramble rate; where passer_games has no row for a
    # QB (a single wildcat snap, say) attempts+sacks is the honest reconstruction.
    #
    # `carries` from the box score counts kneels as rushes, and a kneel is not football a projection
    # should extrapolate: Bo Nix 2025 was 83 carries for 356 yards at face value, 66 for 373 once the
    # 17 kneeldowns come out -- 4.3 yd/carry against 5.7. So the projectable rush is designed runs
    # plus scrambles, kept separate because a designed-run QB and a scrambler are different players,
    # and kneels are carried alongside so composition can add their (negative) yards back rather
    # than pretend they never happened.
    return _with_rates(
        out.with_columns(
            pl.col("dropbacks")
            .fill_null(pl.col("attempts").fill_null(0) + pl.col("sacks_suffered").fill_null(0))
            .alias("dropbacks"),
            (pl.col("designed_qb_rushes").fill_null(0) + pl.col("scrambles").fill_null(0))
            .alias("rush_attempts_clean"),
            (pl.col("designed_qb_rush_yards").fill_null(0) + pl.col("scramble_yards").fill_null(0))
            .alias("rush_yards_clean"),
        ),
        QB_RATES,
    ).with_columns(fantasy_points(out.columns))


@lru_cache(maxsize=8)
def qb_seasons(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    qw = qb_weeks(seasons)
    counts = [
        "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
        "sacks_suffered", "passing_air_yards", "passing_first_downs", "carries", "rushing_yards",
        "rushing_tds", "dropbacks", "scrambles", "scramble_yards", "designed_qb_rushes",
        "designed_qb_rush_yards", "neutral_dropbacks", "neutral_scrambles", "qb_rushes",
        "qb_rush_yards", "kneels", "rush_attempts_clean", "rush_yards_clean",
        "rushing_fumbles_lost", "receiving_fumbles_lost", "fantasy_points",
        "team_pass_attempts", "team_dropbacks", "team_carries", "team_designed_rushes",
        "team_pass_tds", "team_rushing_tds", "team_offensive_tds", "team_plays",
    ]
    have = [c for c in counts if c in qw.columns]
    agg = qw.group_by(["player_id", "season"]).agg(
        pl.col("player_name").drop_nulls().mode().first().alias("player"),
        pl.col("team").drop_nulls().mode().first().alias("team"),
        pl.col("game_id").n_unique().alias("games"),
        *[pl.col(c).sum().alias(c) for c in have],
    )
    return _with_rates(
        agg,
        {
            **QB_RATES,
            "pass_td_share": ("passing_tds", "team_pass_tds"),
            "qb_rush_td_share": ("rushing_tds", "team_rushing_tds"),
            "scramble_yards_per_scramble": ("scramble_yards", "scrambles"),
            "designed_yards_per_rush": ("designed_qb_rush_yards", "designed_qb_rushes"),
            "yards_per_clean_rush": ("rush_yards_clean", "rush_attempts_clean"),
            "clean_rush_share": ("rush_attempts_clean", "team_carries"),
        },
    ).with_columns(
        pl.lit("QB").alias("position"),
        (pl.col("fantasy_points") / pl.col("games")).alias("fantasy_points_per_game"),
        (pl.col("rushing_fumbles_lost") + pl.col("receiving_fumbles_lost")).alias("fumbles_lost"),
        (pl.col("kneels") / pl.col("games")).alias("kneels_per_game"),
        (pl.col("rush_attempts_clean") / pl.col("games")).alias("rush_attempts_per_game"),
    )


# --------------------------------------------------------------------------- #
# the box score, every position on one grain
# --------------------------------------------------------------------------- #
# What a season of a player is, in the names the projection also uses. One definition, because a
# mapping table between projected and actual stat names is a place for a silent mismatch to live.
SEASON_STATS = ("fantasy_points", "targets", "carries", "receptions", "receiving_yards",
                "receiving_tds", "rushing_yards", "rushing_tds", "attempts", "completions",
                "passing_yards", "passing_tds", "interceptions", "offense_snaps")


@lru_cache(maxsize=8)
def player_seasons(seasons: tuple[int, ...], scoring: Scoring | None = None) -> pl.DataFrame:
    """Season totals per player from `player_games`, scored with the scoring passed in.

    `player_games` rather than the usage tables because it is the one place a quarterback's passing
    and his rushing sit on the same row, and fantasy points need both. This is what the backtest
    scores against and what the app shows beside a projection, so it is one function.
    """
    pg = lake.regular_season(lake.read("player_games", seasons=seasons))
    pg = pg.with_columns(fantasy_points(pg.columns, scoring))
    have = [c for c in SEASON_STATS if c in pg.columns]
    return pg.group_by(["season", "player_id"]).agg(
        pl.col("player_name").drop_nulls().first().alias("player"),
        pl.col("position").drop_nulls().first().alias("position"),
        pl.col("team").drop_nulls().mode().first().alias("team"),
        pl.len().cast(pl.Float64).alias("games"),
        *[pl.col(c).cast(pl.Float64).fill_null(0.0).sum().alias(c) for c in have],
    )


# --------------------------------------------------------------------------- #
# defence faced
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def defense_seasons(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """What each defence allowed per game, and how that compares to the league that season.

    Built by grouping team-games on `opponent`: the offence's row is the defence's row seen from the
    other side. Factors are expressed as ratios to the league mean of the same season, so they are
    directly multiplicative on a projection and dimensionless across eras.
    """
    tw = team_weeks(seasons)
    allowed = tw.group_by(["season", pl.col("opponent").alias("defense")]).agg(
        pl.len().alias("games"),
        pl.col("plays").sum().alias("plays_allowed"),
        pl.col("volume_plays").sum().alias("volume_plays_allowed"),
        pl.col("pass_attempts").sum().alias("pass_att_allowed"),
        pl.col("carries").sum().alias("carries_allowed"),
        pl.col("targets").sum().alias("targets_allowed"),
        pl.col("pass_yards").sum().alias("pass_yards_allowed"),
        pl.col("rush_yards").sum().alias("rush_yards_allowed"),
        pl.col("pass_tds").sum().alias("pass_tds_allowed"),
        pl.col("rush_tds").sum().alias("rush_tds_allowed"),
        pl.col("offensive_tds").sum().alias("off_tds_allowed"),
        pl.col("points").sum().alias("points_allowed"),
        pl.col("epa_per_play").mean().alias("epa_per_play_allowed"),
        pl.col("success_rate").mean().alias("success_rate_allowed"),
        pl.col("explosive_pass_rate").mean().alias("explosive_pass_rate_allowed"),
        pl.col("explosive_rush_rate").mean().alias("explosive_rush_rate_allowed"),
        pl.col("seconds_per_play").mean().alias("seconds_per_play_faced"),
        pl.col("red_zone_trips").sum().alias("rz_trips_allowed"),
    )
    allowed = allowed.with_columns(
        (pl.col("plays_allowed") / pl.col("games")).alias("plays_allowed_pg"),
        (pl.col("pass_att_allowed") / pl.col("games")).alias("pass_att_allowed_pg"),
        (pl.col("carries_allowed") / pl.col("games")).alias("carries_allowed_pg"),
        (pl.col("points_allowed") / pl.col("games")).alias("points_allowed_pg"),
        (pl.col("off_tds_allowed") / pl.col("games")).alias("off_tds_allowed_pg"),
        _rate("pass_yards_allowed", "pass_att_allowed", "yards_per_att_allowed"),
        _rate("rush_yards_allowed", "carries_allowed", "yards_per_carry_allowed"),
        _rate("pass_att_allowed", "volume_plays_allowed", "pass_rate_faced"),
    )
    factor_cols = [
        "plays_allowed_pg", "pass_att_allowed_pg", "carries_allowed_pg", "points_allowed_pg",
        "off_tds_allowed_pg", "yards_per_att_allowed", "yards_per_carry_allowed",
        "pass_rate_faced", "epa_per_play_allowed", "success_rate_allowed",
    ]
    return allowed.with_columns(
        [
            (pl.col(c) / pl.col(c).mean().over("season")).alias(f"{c}__factor")
            for c in factor_cols
        ]
    )


@lru_cache(maxsize=8)
def defense_by_position(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """What each defence allowed to each position group, as a ratio to the league mean.

    This is the part a team-level rating cannot say: two defences with identical points allowed can
    differ completely in whether the damage came from backs or from receivers.
    """
    sw = skill_weeks(seasons)
    agg = sw.group_by(["season", pl.col("opponent").alias("defense"), "position"]).agg(
        pl.col("game_id").n_unique().alias("games"),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("receiving_yards").sum().alias("receiving_yards"),
        pl.col("receiving_tds").sum().alias("receiving_tds"),
        pl.col("carries").sum().alias("carries"),
        pl.col("rushing_yards").sum().alias("rushing_yards"),
        pl.col("rushing_tds").sum().alias("rushing_tds"),
        pl.col("fantasy_points").sum().alias("fantasy_points"),
    )
    agg = agg.with_columns(
        (pl.col("targets") / pl.col("games")).alias("targets_pg"),
        (pl.col("carries") / pl.col("games")).alias("carries_pg"),
        (pl.col("fantasy_points") / pl.col("games")).alias("fantasy_points_pg"),
        _rate("receiving_yards", "targets", "yards_per_target_allowed"),
        _rate("rushing_yards", "carries", "yards_per_carry_allowed"),
        _rate("receptions", "targets", "catch_rate_allowed"),
        ((pl.col("receiving_tds") + pl.col("rushing_tds")) / pl.col("games")).alias("tds_allowed_pg"),
    )
    factor_cols = [
        "targets_pg", "carries_pg", "fantasy_points_pg", "yards_per_target_allowed",
        "yards_per_carry_allowed", "catch_rate_allowed", "tds_allowed_pg",
    ]
    return agg.with_columns(
        [
            (pl.col(c) / pl.col(c).mean().over(["season", "position"])).alias(f"{c}__factor")
            for c in factor_cols
        ]
    )


# --------------------------------------------------------------------------- #
# the injury report
# --------------------------------------------------------------------------- #
# Availability is the one term in the engine that is *stated* rather than fitted: a status of PUP or
# NON is turned into a multiplier by a table of assumptions. This is the record against which those
# assumptions are read -- not to fit them, but so that "back in week 6" and "he misses about three"
# are typed beside what this player has actually missed rather than beside nothing.
#
# It is a weekly report, so the grain is a player-week and the counts below are counts of *distinct
# weeks*: a team files three practice reports a week, and counting rows would say a hamstring cost
# somebody fifty-one games.
PRACTICE = {"Did Not": "DNP", "Limited": "limited", "Full": "full"}


@lru_cache(maxsize=8)
def injury_weeks(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One row per player-week on the league's injury report, regular season only.

    `game_type` rather than `season_type` decides what is regular season: the latter is null on most
    of the table, and a null read as REG would fold January into a seventeen-game season.
    """
    seasons = seasons or lake.history_seasons()
    inj = lake.read("injuries", layer="raw", seasons=seasons)
    keep = inj.filter(pl.col("game_type") == "REG")
    practice = pl.col("practice_status").fill_null("")
    return keep.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("full_name").alias("player"),
        "season",
        pl.col("week").cast(pl.Int32).alias("week"),
        "team",
        "position",
        pl.col("report_status").alias("status"),
        pl.coalesce("report_primary_injury", "practice_primary_injury").alias("injury"),
        pl.when(practice.str.starts_with("Did Not")).then(pl.lit("DNP"))
        .when(practice.str.starts_with("Limited")).then(pl.lit("limited"))
        .when(practice.str.starts_with("Full")).then(pl.lit("full"))
        .otherwise(None).alias("practice"),
    )


@lru_cache(maxsize=8)
def injury_report(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One row per player-season: how much of it he spent on the report, and with what.

    `weeks_out` is the number that matters -- a Friday designation of Out is the league telling you he
    will not play, which is as close to observed missed time as a report gets. `weeks_questionable` is
    much softer and is here to be read as noise beside it rather than added to it.
    """
    iw = injury_weeks(seasons)
    if iw.is_empty():
        return iw
    weeks_where = lambda cond: pl.col("week").filter(cond).n_unique()  # noqa: E731
    return iw.group_by(["player_id", "season"]).agg(
        pl.col("player").drop_nulls().mode().first().alias("player"),
        pl.col("team").drop_nulls().mode().first().alias("team"),
        pl.col("position").drop_nulls().mode().first().alias("position"),
        pl.col("week").n_unique().alias("weeks_listed"),
        weeks_where(pl.col("status") == "Out").alias("weeks_out"),
        weeks_where(pl.col("status") == "Doubtful").alias("weeks_doubtful"),
        weeks_where(pl.col("status") == "Questionable").alias("weeks_questionable"),
        weeks_where(pl.col("practice") == "DNP").alias("weeks_dnp"),
        weeks_where(pl.col("practice") == "limited").alias("weeks_limited"),
        pl.col("injury").drop_nulls().mode().first().alias("main_injury"),
        pl.col("injury").n_unique().alias("distinct_injuries"),
    ).sort(["player_id", "season"])


def league_means(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """League average of each team-season metric -- the mean a team estimate regresses toward."""
    ts = team_seasons(seasons)
    metrics = [
        c for c, dt in zip(ts.columns, ts.dtypes, strict=True)
        if dt.is_numeric() and c not in ("season", "games")
    ]
    return ts.group_by("season").agg([pl.col(c).mean().alias(c) for c in metrics]).sort("season")


def clear_cache() -> None:
    for fn in (team_weeks, team_seasons, skill_weeks, skill_seasons, qb_weeks, qb_seasons,
               defense_seasons, defense_by_position, injury_weeks, injury_report):
        fn.cache_clear()


if __name__ == "__main__":
    import sys

    seasons = lake.history_seasons()
    pl.Config.set_tbl_width_chars(220)
    ts, sk, qb = team_seasons(seasons), skill_seasons(seasons), qb_seasons(seasons)
    print(f"team_seasons {ts.shape}  skill_seasons {sk.shape}  qb_seasons {qb.shape}")
    last = LAST_COMPLETE_SEASON
    print(f"\n-- {last} pace and shape, fastest 5")
    print(ts.filter(pl.col("season") == last).sort("seconds_per_play").head(5).select(
        "team", "plays_per_game", "seconds_per_play", "pass_rate", "proe", "points_per_game"
    ))
    print(f"\n-- {last} top 5 WR by fantasy points")
    print(sk.filter((pl.col("season") == last) & (pl.col("position") == "WR"))
          .sort("fantasy_points", descending=True).head(5)
          .select("player", "team", "games", "target_share", "tprr", "adot", "catch_rate",
                  "yards_per_reception", "fantasy_points"))
    print(f"\n-- {last} top 5 QB, dropback detail")
    print(qb.filter(pl.col("season") == last).sort("fantasy_points", descending=True).head(5)
          .select("player", "team", "games", "dropback_share", "attempt_rate", "scramble_rate",
                  "sack_rate", "designed_rush_share", "yards_per_attempt", "fantasy_points"))
    print(f"\n-- {last} most time on the injury report, skill positions")
    print(injury_report(seasons)
          .filter((pl.col("season") == last) & pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
          .sort("weeks_out", descending=True).head(5)
          .select("player", "position", "team", "weeks_listed", "weeks_out", "weeks_dnp",
                  "main_injury"))
    print(f"\n-- {last} five stingiest defences vs WR (fantasy points allowed factor)")
    print(defense_by_position(seasons)
          .filter((pl.col("season") == last) & (pl.col("position") == "WR"))
          .sort("fantasy_points_pg__factor").head(5)
          .select("defense", "targets_pg", "fantasy_points_pg", "fantasy_points_pg__factor",
                  "yards_per_target_allowed__factor"))
    sys.stdout.flush()
