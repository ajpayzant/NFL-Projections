"""Opportunity times efficiency: the stat line, the fantasy points, the season.

This is the arithmetic the Excel workbook did in cells, done per game instead of per season:

    stat(week) = count(week) x rate(week)
    season     = sum over the games on the real 2026 schedule

Nothing is fitted here. Every count comes from `opportunity` (a share of a team pool, weighted by
availability, optionally normalized) and every rate from `efficiency` (the player's shrunk history
times his team's per-game factor). Composition's whole job is to be a correct identity, which is why
it is worth its own module and its own tests: if a receiver's yards do not equal his targets times his
yards per target, the number on the board is not the number the engine believes.

Three places the identity needs care:

**A scramble is two events at once**, and that is the hard part. It is a dropback that broke down and
also a rush attempt, so it appears in the dropback identity and in the carries pool, and the two do not
have to agree -- projected independently they disagreed by up to 5%. The carries pool wins, because it
is what keeps a team's ground game balanced, and the dropback identity is then closed around it:

    scrambles  = clean rushes x (scramble rate / (scramble rate + designed-run share))
    designed   = clean rushes - scrambles
    attempts + sacks = dropbacks x 1.003 - scrambles,  split by the fitted attempt:sack ratio

Both identities now hold exactly rather than approximately. The 1.003 is measured, not assumed -- a
handful of dropbacks over 2016-2025 end in neither an attempt, a sack, nor a scramble, and forcing the
sum to 1.000 would delete them. Pass attempts absorb the residual because attempt rate is definitionally
what is left of a dropback; the fitted rate is kept for the attempt-versus-sack split, where it is
carrying information the identity does not already contain.

Yardage then uses the two very different rush rates (3.05 yd a designed run, 7.25 a scramble) rather
than one blended figure, which is the whole reason for splitting them.

**Touchdowns come from pools, not from rates.** A red-zone share tells you who is on the field near
the goal line; the touchdown itself is the team's projected total divided among claimants, so team TDs
and player TDs cannot disagree.

    python -m src.model.compose                  # season board, team reconciliation, one team's week
    python -m src.model.compose --team PHI
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import polars as pl

from src.config import PROJ_SEASON, Settings
from src.data import history, lake
from src.model import efficiency, opportunity

# What a player's week looks like when composition is done. Order is the display order.
COUNT_COLUMNS = ("offense_snaps", "routes", "rush_plays", "targets", "carries", "dropbacks")

PASS_COLUMNS = ("attempts", "completions", "passing_yards", "passing_tds", "interceptions",
                "sacks", "passing_air_yards")

RUSH_COLUMNS = ("rushing_yards", "rushing_tds", "designed_rushes", "scrambles",
                "designed_rush_yards", "scramble_yards")

REC_COLUMNS = ("receptions", "receiving_yards", "receiving_tds", "receiving_air_yards")

STAT_COLUMNS = COUNT_COLUMNS + PASS_COLUMNS + RUSH_COLUMNS + REC_COLUMNS + ("fumbles_lost",)

# columns that are counts of events and therefore sum over a season; everything else is derived
SEASON_SUM = STAT_COLUMNS + ("fantasy_points",)

EPS = 1e-9


@lru_cache(maxsize=8)
def measure_dropback_split(seasons: tuple[int, ...] | None = None) -> float:
    """(attempts + sacks + scrambles) / dropbacks over the whole history.

    Measured rather than assumed for the same reason the pool targets are: it comes back at 1.003, and
    forcing it to 1.000 would quietly delete the dropbacks that end in neither of the three.
    """
    seasons = seasons or lake.history_seasons()
    qb = history.qb_seasons(seasons)
    tot = qb.select(
        (pl.col("attempts").sum() + pl.col("sacks_suffered").sum() + pl.col("scrambles").sum())
        / pl.col("dropbacks").sum()
    )
    return float(tot.to_series()[0])


def _is_qb() -> pl.Expr:
    return pl.col("position") == "QB"


def _dropback_fates(target: float) -> list[pl.Expr]:
    """Close the dropback identity around the scramble count the carries pool already fixed.

    `r_scrambles` is the QB's slice of the team's clean rushes, split off by the ratio of his scramble
    rate to his designed-run share -- both fitted, and only their ratio is used here, so the absolute
    level of either cannot move his rushing volume. `r_attempts` and `r_sacks` then divide whatever is
    left of his dropbacks in the fitted attempt:sack proportion.
    """
    n_carries = pl.col("carries").fill_null(0.0)
    n_drops = pl.col("dropbacks").fill_null(0.0)

    pre_scr = n_drops * pl.col("used_scramble_rate").fill_null(0.0)
    pre_des = pl.col("designed_qb_rushes").fill_null(0.0)
    split_den = pre_scr + pre_des
    scr_frac = pl.when(split_den > EPS).then(pre_scr / split_den).otherwise(pl.lit(0.0))
    scrambles = pl.when(_is_qb()).then(n_carries * scr_frac).otherwise(pl.lit(0.0))

    att = pl.col("used_attempt_rate").fill_null(0.0)
    sack = pl.col("used_sack_rate").fill_null(0.0)
    # a scramble is already spent, so the rest of the dropbacks are thrown or lost behind the line
    rest = (n_drops * pl.lit(target) - scrambles).clip(0.0)
    att_frac = pl.when(att + sack > EPS).then(att / (att + sack)).otherwise(pl.lit(1.0))
    attempts = rest * att_frac
    return [
        scrambles.alias("r_scrambles"),
        attempts.alias("r_attempts"),
        (rest - attempts).alias("r_sacks"),
    ]


def _stat_line() -> list[pl.Expr]:
    """Every stat as an expression over counts and `used_` rates. One place, one arithmetic."""
    n_targets = pl.col("targets").fill_null(0.0)
    n_carries = pl.col("carries").fill_null(0.0)
    n_drops = pl.col("dropbacks").fill_null(0.0)

    attempts = pl.col("r_attempts")
    scrambles = pl.col("r_scrambles")
    sacks = pl.col("r_sacks")
    qb_scrambles = scrambles
    qb_designed = pl.when(_is_qb()).then(n_carries - scrambles).otherwise(pl.lit(0.0))

    receptions = n_targets * pl.col("used_catch_rate").fill_null(0.0)
    touches = n_carries + receptions

    return [
        # --- passing -------------------------------------------------------- #
        attempts.alias("attempts"),
        (attempts * pl.col("used_completion_pct").fill_null(0.0)).alias("completions"),
        (attempts * pl.col("used_yards_per_attempt").fill_null(0.0)).alias("passing_yards"),
        pl.col("passing_tds").fill_null(0.0).alias("passing_tds"),
        (attempts * pl.col("used_int_rate").fill_null(0.0)).alias("interceptions"),
        sacks.alias("sacks"),
        (attempts * pl.col("used_air_yards_per_attempt").fill_null(0.0)).alias("passing_air_yards"),
        # --- rushing -------------------------------------------------------- #
        qb_designed.alias("designed_rushes"),
        qb_scrambles.alias("scrambles"),
        pl.when(_is_qb())
        .then(qb_designed * pl.col("used_designed_rush_ypc").fill_null(0.0))
        .otherwise(pl.lit(0.0))
        .alias("designed_rush_yards"),
        pl.when(_is_qb())
        .then(qb_scrambles * pl.col("used_scramble_ypc").fill_null(0.0))
        .otherwise(pl.lit(0.0))
        .alias("scramble_yards"),
        pl.when(_is_qb())
        .then(qb_designed * pl.col("used_designed_rush_ypc").fill_null(0.0)
              + qb_scrambles * pl.col("used_scramble_ypc").fill_null(0.0))
        .otherwise(n_carries * pl.col("used_yards_per_carry").fill_null(0.0))
        .alias("rushing_yards"),
        pl.col("rushing_tds").fill_null(0.0).alias("rushing_tds"),
        # --- receiving ------------------------------------------------------ #
        receptions.alias("receptions"),
        (n_targets * pl.col("used_yards_per_target").fill_null(0.0)).alias("receiving_yards"),
        pl.col("receiving_tds").fill_null(0.0).alias("receiving_tds"),
        pl.col("air_yards").fill_null(0.0).alias("receiving_air_yards"),
        # --- the ball on the floor ------------------------------------------ #
        # a QB's fumble rate is per dropback (most of them are sacks and exchanges), a skill player's
        # per touch. Two denominators, so the position picks which one applies.
        pl.when(_is_qb())
        .then(n_drops * pl.col("used_qb_fumble_rate").fill_null(0.0))
        .otherwise(touches * pl.col("used_fumble_rate").fill_null(0.0))
        .alias("fumbles_lost"),
    ]


KEEP = ("season", "week", "game_id", "team", "opponent", "is_home", "player_id", "player",
        "position", "depth_slot", "slot_bucket", "is_rookie", "draft_pick", "status",
        "expected_games", "p_play", "spread", "total", "implied_points", "has_market")


def weekly(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    opp: pl.DataFrame | None = None,
    player_rates: pl.DataFrame | None = None,
    split_target: float | None = None,
    adj: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per player per game: counts, stats, fantasy points.

    `p_play` is already inside every count, so a row is an *expectation* for that week rather than a
    line conditional on playing. That is what makes the season a plain sum.

    `adj` is the counts already joined to their rates -- what `efficiency.adjusted` returns. A caller
    passes it when it has edited a rate for one game, since the edit has to land between the join and
    the stat line and there is nowhere else to put it. Everything from there on is the same arithmetic.
    """
    settings = settings or Settings()
    opp = opportunity.opportunity(season, settings) if opp is None else opp
    if player_rates is None:
        player_rates = efficiency.rates(season, settings)
    if adj is None:
        adj = efficiency.adjusted(opp, player_rates, settings)

    target = measure_dropback_split() if split_target is None else split_target
    out = adj.with_columns(_dropback_fates(target)).with_columns(_stat_line())
    keep = [c for c in KEEP if c in out.columns]
    counts = [c for c in COUNT_COLUMNS if c in out.columns]
    stats = [c for c in STAT_COLUMNS if c not in counts]
    out = out.select(*keep, *[pl.col(c).fill_null(0.0) for c in counts], *stats)
    return out.with_columns(
        history.fantasy_points(out.columns, settings.scoring)
    ).sort(["team", "week", "position", "depth_slot"])


def seasonal(
    week_frame: pl.DataFrame | None = None,
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Season totals and per-game means, one row per player.

    `games` is the sum of weekly availability, not a count of rows: a player expected to miss three
    weeks has 14 games and his per-game line is scored over those 14, which is the number a lineup
    decision needs.
    """
    settings = settings or Settings()
    wk = weekly(season, settings) if week_frame is None else week_frame
    sums = [c for c in SEASON_SUM if c in wk.columns]
    ident = ("season", "team", "player_id", "player", "position", "depth_slot", "slot_bucket",
             "is_rookie", "draft_pick", "status")
    out = (
        wk.group_by([c for c in ident if c in wk.columns])
        .agg(
            pl.len().alias("weeks"),
            pl.col("p_play").sum().alias("games"),
            *[pl.col(c).sum().alias(c) for c in sums],
        )
        .with_columns(
            pl.when(pl.col("games") > EPS)
            .then(pl.col("fantasy_points") / pl.col("games"))
            .otherwise(pl.lit(0.0))
            .alias("points_per_game"),
        )
    )
    return out.sort("fantasy_points", descending=True).with_columns(
        pl.col("fantasy_points").rank("min", descending=True).cast(pl.Int32).alias("overall_rank"),
        pl.col("fantasy_points").rank("min", descending=True).over("position")
        .cast(pl.Int32).alias("position_rank"),
    )


def board(
    season_frame: pl.DataFrame,
    settings: Settings | None = None,
    prev: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The ranking board: season totals plus the comparisons a draft board is read for.

    Four derived columns, and each answers a question a raw projection does not:

    - `tier` groups a position into blocks of `settings.tier_size`, which is how a board is used --
      inside a tier the order is noise, between tiers it is not.
    - `drop_next` is the points between him and the next man at his position. A five-point drop says
      wait; a forty-point drop says this is the pick.
    - `vs_starter` is his points against the mean of the startable players at his position, so a
      quarterback and a receiver are comparable without inventing a positional scarcity model.
    - `last_points` / `delta_points` is the projection against what he actually did last year, which
      is the first thing anybody checks and the fastest way to spot a projection that has lost its
      mind. `prev` is a season aggregate from `history.player_seasons`; passing it is optional
      because the board must still render when the lake has no prior season.

    Arithmetic only. Formatting, colour and column choice belong to the app.
    """
    settings = settings or Settings()
    # sorted here rather than trusting the caller: `drop_next` is a window over the row below, so the
    # order of the frame is part of the arithmetic
    out = season_frame.sort("fantasy_points", descending=True).with_columns(
        ((pl.col("position_rank") - 1) // max(settings.tier_size, 1) + 1)
        .cast(pl.Int32).alias("tier"),
        (pl.col("fantasy_points") - pl.col("fantasy_points").shift(-1).over("position"))
        .alias("drop_next"),
    )
    starters = pl.col("position").replace_strict(
        settings.starters, default=0, return_dtype=pl.Int32
    )
    avg_starter = (
        pl.when(pl.col("position_rank") <= starters).then(pl.col("fantasy_points")).otherwise(None)
        .mean().over("position")
    )
    out = out.with_columns(
        (pl.col("fantasy_points") - avg_starter).alias("vs_starter"),
        (pl.col("position_rank") <= starters).alias("startable"),
    )
    if prev is not None and not prev.is_empty():
        last = prev.select(
            "player_id",
            pl.col("team").alias("last_team"),
            pl.col("games").alias("last_games"),
            pl.col("fantasy_points").alias("last_points"),
        )
        out = out.join(last, on="player_id", how="left").with_columns(
            (pl.col("fantasy_points") - pl.col("last_points")).alias("delta_points"),
            # null last_team means he did not play last season at all, which is not a move
            (pl.col("last_team").is_not_null() & (pl.col("last_team") != pl.col("team")))
            .alias("changed_team"),
        )
    return out.sort("fantasy_points", descending=True)


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #
# player stat -> the team-environment column it has to add up to. The left side is a sum over the
# roster, the right side is what the team layer projected before anybody was named.
RECONCILE = {
    "targets": "team_targets",
    "carries": "team_carries",
    "dropbacks": "team_dropbacks",
    "attempts": "team_pass_attempts",
    "receiving_tds": "team_pass_tds",
    "passing_tds": "team_pass_tds",
    "rushing_tds": "team_rush_tds",
    "receiving_yards": "team_pass_yards",
    "rushing_yards": "team_rush_yards",
    "receiving_air_yards": "team_air_yards",
}

# the team layer projects yards as a rate times a count, so its yardage totals are implied rather
# than stored. Deriving them here keeps one definition instead of two.
DERIVED_TEAM = {
    "team_pass_yards": ("team_pass_attempts", "team_yards_per_attempt"),
    "team_rush_yards": ("team_carries", "team_yards_per_carry"),
}


def team_check(
    week_frame: pl.DataFrame | None = None,
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    env: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Do the players add up to the team? The invariant, as a table rather than a promise.

    Counts drawn straight from a normalized pool must match to rounding. Yards and completions are
    the product of a count and a rate and are *not* forced to match: a team whose receivers are all
    more efficient than its recent offence really should project above the team's own yardage, and
    silently scaling that away would hide the disagreement instead of showing it.
    """
    settings = settings or Settings()
    wk = weekly(season, settings) if week_frame is None else week_frame
    # the same environment the players were divided from, edits included, or a fresh one if the
    # caller has none: reconciling against an unedited team would report the edit as a discrepancy
    env = opportunity._game_frame(season, settings, env=env)
    env = env.with_columns([
        (pl.col(a) * pl.col(b)).alias(name)
        for name, (a, b) in DERIVED_TEAM.items() if a in env.columns and b in env.columns
    ])
    have = {k: v for k, v in RECONCILE.items() if k in wk.columns and v in env.columns}
    per_team = wk.group_by(["game_id", "team"]).agg(
        [pl.col(k).sum().alias(k) for k in have]
    ).join(
        env.select("game_id", "team", *dict.fromkeys(have.values())),
        on=["game_id", "team"], how="inner",
    )
    rows = []
    for stat, team_col in have.items():
        d = per_team.select(
            (pl.col(stat) - pl.col(team_col)).alias("d"), pl.col(stat), pl.col(team_col)
        )
        rows.append({
            "stat": stat,
            "team_source": team_col,
            "players_per_game": float(d[stat].mean()),
            "team_per_game": float(d[team_col].mean()),
            "gap_pct": 100.0 * float(d["d"].mean()) / max(float(d[team_col].mean()), EPS),
            "worst_abs_gap": float(d["d"].abs().max()),
            "from_pool": stat in {p.name for p in opportunity.POOLS if p.exclusive},
        })
    return pl.DataFrame(rows)


def _report(season: int, settings: Settings, team_code: str) -> None:
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_fmt_float("mixed")

    wk = weekly(season, settings)
    yr = seasonal(wk, season, settings)
    print(f"\nWEEKLY  {wk.height:,} player-games   SEASON  {yr.height:,} players"
          f"   dropback split target {measure_dropback_split():.4f}")

    print("\nTEAM RECONCILIATION  players summed against the team environment they were divided from")
    print(team_check(wk, season, settings).with_columns(
        pl.col("players_per_game").round(2), pl.col("team_per_game").round(2),
        pl.col("gap_pct").round(2), pl.col("worst_abs_gap").round(3),
    ).sort("from_pool", descending=True))

    for pos in ("QB", "RB", "WR", "TE"):
        cols = {
            "QB": ("attempts", "passing_yards", "passing_tds", "interceptions", "designed_rushes",
                   "scrambles", "rushing_yards", "rushing_tds"),
            "RB": ("carries", "rushing_yards", "rushing_tds", "targets", "receptions",
                   "receiving_yards", "receiving_tds"),
            "WR": ("targets", "receptions", "receiving_yards", "receiving_tds", "carries",
                   "rushing_yards"),
            "TE": ("targets", "receptions", "receiving_yards", "receiving_tds", "routes"),
        }[pos]
        top = yr.filter(pl.col("position") == pos).head(12)
        print(f"\nTOP {pos}  {season} season totals")
        print(top.select(
            "position_rank", "player", "team", pl.col("games").round(1),
            *[pl.col(c).round(1) for c in cols if c in top.columns],
            pl.col("fantasy_points").round(1), pl.col("points_per_game").round(2),
        ))

    ex = wk.filter((pl.col("team") == team_code) & (pl.col("week") == 1))
    print(f"\n{team_code} WEEK 1  the full offensive line, expectation per player")
    print(ex.select(
        "position", "depth_slot", "player", pl.col("p_play").round(2),
        *[pl.col(c).round(2) for c in ("targets", "receptions", "receiving_yards", "carries",
                                       "rushing_yards", "attempts", "passing_yards",
                                       "fantasy_points") if c in ex.columns],
    ).head(24))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--team", default="PHI", help="team code for the week-1 detail table")
    p.add_argument("--no-normalize", action="store_true", help="compose from unscaled pools")
    a = p.parse_args(argv)
    _report(a.season, Settings(normalize_pools=not a.no_normalize), a.team)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
