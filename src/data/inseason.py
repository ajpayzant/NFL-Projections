"""What has already happened this season, in the columns the projection speaks.

A season projection made in November that still says what it said in August is not a projection, it is a
souvenir. Two things have to be true for it to keep up, and this module is the first of them: the results
have to be *readable*. The heavy played-game tables in `processed/` are built by the pbp pipeline in the
other repo and arrive on its schedule, which is fine for five years of history and useless for last
Sunday -- so the weekly stat tables are fetched directly here, the same way `refresh.py` already fetches
the rosters, and turned into one frame per player per week with the names `compose.weekly` uses.

The second thing is `compose.actualise`, which spends it: a completed week stops being a projection and
becomes what happened, and the season total becomes results-to-date plus the projected rest. Nothing in
this module decides that. It reports facts and says how far they go.

Three sources, because no one of them is enough:

    player_stats   the stat line -- every scoring column, which is what the season total is made of
    snap_counts    who was on the field, which is what says an absence was an absence rather than a
                   quiet game, and the only measure of playing time there is
    team_stats     the team's own totals, so a team's results can be checked without summing players

    python -m src.data.inseason              # what is complete, and a sample of it
    python -m src.data.inseason --week 2
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import polars as pl

from src.config import PROJ_SEASON
from src.data import lake

# The live tables this module reads. `refresh.py` owns all three; `lake.read` prefers its copies.
TABLES = ("player_stats", "snap_counts", "team_stats")

# upstream name -> the name the weekly frame uses. Everything the scoring dict can pay for is in here,
# which is the property that matters: a substituted week has to be able to price itself.
PLAYER_STATS = {
    "attempts": "attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_interceptions": "interceptions",
    "passing_air_yards": "passing_air_yards",
    "sacks_suffered": "sacks",
    "carries": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "receiving_air_yards": "receiving_air_yards",
}

# Fumbles lost are three columns upstream -- on a carry, on a catch, on a sack -- and one column here,
# because that is how they are scored. A man who fumbles twice in a game loses two.
FUMBLE_PARTS = ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost")

MEASURED = tuple(PLAYER_STATS.values()) + ("fumbles_lost", "offense_snaps")


def _reg(df: pl.DataFrame) -> pl.DataFrame:
    """Regular season only, whichever column this table spells it in."""
    for col in ("season_type", "game_type"):
        if col in df.columns:
            return df.filter(pl.col(col) == "REG")
    return df


@lru_cache(maxsize=8)
def team_weeks(season: int = PROJ_SEASON) -> pl.DataFrame:
    """Which (team, week) pairs are in the books. The unit everything else is keyed on.

    Per team rather than per league, and that is not pedantry: byes, a Thursday game and a Monday game
    mean two teams in the same week are not equally far into their season, and a projection that
    substituted a whole league week at a time would either throw away a Thursday result for four days or
    zero out a game that had not been played yet.

    A team is credited with a week when its own row appears in `team_stats`, which is written from the
    finished game. An in-progress game has no row, so a Sunday afternoon reads as not yet played, which
    is the answer that keeps the projection honest while the ball is in the air.
    """
    try:
        ts = _reg(lake.read("team_stats", "raw", (season,)))
    except Exception:                       # noqa: BLE001 -- no table is a season that has not started
        return pl.DataFrame(schema={"team": pl.String, "week": pl.Int32})
    if ts.is_empty() or not {"team", "week"} <= set(ts.columns):
        return pl.DataFrame(schema={"team": pl.String, "week": pl.Int32})
    return (ts.select(pl.col("team").cast(pl.String), pl.col("week").cast(pl.Int32))
            .unique().sort(["team", "week"]))


def weeks_complete(season: int = PROJ_SEASON) -> int:
    """How many weeks every team has finished. The one number a header can honestly print.

    The minimum across teams, not the maximum: "through week 2" has to mean every team has played two,
    or a reader compares a team that has played three games with one that has played one and calls the
    projection broken. Zero before the season, and zero when the tables are missing.
    """
    tw = team_weeks(season)
    if tw.is_empty():
        return 0
    per_team = tw.group_by("team").len()
    # a team with no rows at all is not in this frame, so the count of teams matters as much as the min
    return 0 if per_team.height < 32 else int(per_team["len"].min())


def weeks_played(season: int = PROJ_SEASON) -> int:
    """The furthest any team has got, which is how much data exists rather than how much is comparable."""
    tw = team_weeks(season)
    return 0 if tw.is_empty() else int(tw["week"].max())


@lru_cache(maxsize=8)
def player_weeks(season: int = PROJ_SEASON) -> pl.DataFrame:
    """One row per player per completed game: what he actually did, in the projection's own column names.

    Every column in `MEASURED`, and `played` -- the 0/1 that replaces `p_play` for a week that is over.
    An expectation is the right thing to carry for a game that has not happened and the wrong thing to
    carry for one that has: 0.94 of a game he played all of, or of one he was inactive for, is a number
    about neither.

    `played` comes from the snap count rather than from the stat line, because a receiver who was on the
    field for thirty snaps and was not thrown at did play, and a stat line cannot say so. Falling back to
    the stat line when the snap table has not published yet is the lesser error -- it under-counts the men
    who did nothing, and gets every man who did something right.
    """
    empty = pl.DataFrame(schema={"player_id": pl.String, "team": pl.String, "week": pl.Int32,
                                "played": pl.Float64, **{c: pl.Float64 for c in MEASURED}})
    try:
        ps = _reg(lake.read("player_stats", "raw", (season,)))
    except Exception:                       # noqa: BLE001
        return empty
    if ps.is_empty() or "player_id" not in ps.columns:
        return empty

    have = {up: name for up, name in PLAYER_STATS.items() if up in ps.columns}
    fumbles = [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in FUMBLE_PARTS if c in ps.columns]
    out = ps.select(
        pl.col("player_id").cast(pl.String),
        pl.col("team").cast(pl.String),
        pl.col("week").cast(pl.Int32),
        *[pl.col(up).cast(pl.Float64).fill_null(0.0).alias(name) for up, name in have.items()],
        (pl.sum_horizontal(fumbles) if fumbles else pl.lit(0.0)).alias("fumbles_lost"),
    )
    # one row per man per week: upstream splits a mid-season trade into two team rows for the same game
    out = out.group_by(["player_id", "week"]).agg(
        pl.col("team").last(),
        *[pl.col(c).sum() for c in [*have.values(), "fumbles_lost"]],
    )
    out = out.join(snaps(season), on=["player_id", "week"], how="full", coalesce=True)
    for name in MEASURED:
        if name not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(name))
    return out.with_columns(
        pl.col("team").fill_null(pl.col("team_snaps")) if "team_snaps" in out.columns
        else pl.col("team"),
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in MEASURED],
    ).with_columns(
        # he was there if he took a snap; if the snap table is silent, if he recorded anything
        pl.when(pl.col("offense_snaps") > 0).then(1.0)
        .when(pl.sum_horizontal([pl.col(c) for c in PLAYER_STATS.values() if c in out.columns]) > 0)
        .then(1.0).otherwise(0.0).alias("played")
    ).drop([c for c in ("team_snaps",) if c in out.columns]).sort(["week", "team", "player_id"])


@lru_cache(maxsize=8)
def snaps(season: int = PROJ_SEASON) -> pl.DataFrame:
    """Offensive snaps per player per week, on gsis ids.

    The one table in the lake keyed on a pfr id, which is why this is its own function: the crosswalk
    lives in `players`, and a player missing from it is dropped rather than guessed at. That costs the
    snap count for a man the crosswalk has not caught up with -- an undrafted rookie signed on Tuesday --
    and `player_weeks` falls back to his stat line for whether he played, so the loss is bounded.
    """
    empty = pl.DataFrame(schema={"player_id": pl.String, "week": pl.Int32,
                                "offense_snaps": pl.Float64, "team_snaps": pl.String})
    try:
        sc = _reg(lake.read("snap_counts", "raw", (season,)))
    except Exception:                       # noqa: BLE001
        return empty
    if sc.is_empty() or "pfr_player_id" not in sc.columns:
        return empty
    cross = crosswalk()
    if cross.is_empty():
        return empty
    return (
        sc.select(
            pl.col("pfr_player_id").cast(pl.String),
            pl.col("week").cast(pl.Int32),
            pl.col("team").cast(pl.String).alias("team_snaps"),
            pl.col("offense_snaps").cast(pl.Float64).fill_null(0.0),
        )
        .join(cross, on="pfr_player_id", how="inner")
        .group_by(["player_id", "week"])
        .agg(pl.col("offense_snaps").sum(), pl.col("team_snaps").last())
    )


@lru_cache(maxsize=1)
def crosswalk() -> pl.DataFrame:
    """`pfr_player_id` -> `player_id`, from the players table. Fetched, not cached to disk.

    Small, changes only when somebody enters the league, and needed by exactly one function -- so it is
    fetched on demand rather than made a fourth owned table. An upstream failure returns empty, which
    costs the snap counts and no more.
    """
    try:
        import nflreadpy as nfl

        players = nfl.load_players()
    except Exception:                       # noqa: BLE001
        return pl.DataFrame(schema={"pfr_player_id": pl.String, "player_id": pl.String})
    cols = set(players.columns)
    gsis = "gsis_id" if "gsis_id" in cols else "player_id"
    if "pfr_id" in cols:
        pfr = "pfr_id"
    elif "pfr_player_id" in cols:
        pfr = "pfr_player_id"
    else:
        return pl.DataFrame(schema={"pfr_player_id": pl.String, "player_id": pl.String})
    return (
        players.select(
            pl.col(pfr).cast(pl.String).alias("pfr_player_id"),
            pl.col(gsis).cast(pl.String).alias("player_id"),
        )
        .filter(pl.col("pfr_player_id").is_not_null() & pl.col("player_id").is_not_null())
        .unique(subset="pfr_player_id")
    )


@lru_cache(maxsize=8)
def team_pools(season: int = PROJ_SEASON) -> pl.DataFrame:
    """The team's own totals per completed game, for the pools the engine divides.

    Not summed off the players: a team's attempts are a team fact, and reading them from the team table
    means a player missing from the crosswalk cannot quietly shrink his offence. Used to check a
    substituted week rather than to build one.
    """
    empty = pl.DataFrame(schema={"team": pl.String, "week": pl.Int32})
    try:
        ts = _reg(lake.read("team_stats", "raw", (season,)))
    except Exception:                       # noqa: BLE001
        return empty
    if ts.is_empty():
        return empty
    want = {"attempts": "attempts", "carries": "carries", "targets": "targets",
            "passing_yards": "passing_yards", "rushing_yards": "rushing_yards",
            "passing_tds": "passing_tds", "rushing_tds": "rushing_tds",
            "receptions": "receptions", "sacks_suffered": "sacks",
            # not used to check a substituted week -- it is the denominator of `air_yards_share`, and
            # the only place a team's air yards can be read without summing receivers
            "passing_air_yards": "passing_air_yards"}
    return ts.select(
        pl.col("team").cast(pl.String),
        pl.col("week").cast(pl.Int32),
        *[pl.col(up).cast(pl.Float64).fill_null(0.0).alias(name)
          for up, name in want.items() if up in ts.columns],
    ).group_by(["team", "week"]).agg(pl.all().sum()).sort(["team", "week"])


# --------------------------------------------------------------------------- #
# the season so far as one more season of history
# --------------------------------------------------------------------------- #
# What the weekly tables can and cannot measure, in the column names `history.skill_seasons` and
# `history.qb_seasons` use -- because those are the names `priors.Metric` reads, and a metric whose
# numerator and denominator are both here gets to learn from this season while one that is missing either
# keeps its preseason estimate untouched.
#
# The omissions are the point of the design and are deliberate rather than pending. Routes run, red-zone
# targets, short-yardage carries, late-down targets and rush success are all play-by-play facts: nobody
# publishes them within hours of a game, and they arrive when the pbp pipeline in the other repo rebuilds.
# Nor are dropbacks here, though it is tempting -- `attempts + sacks` is a dropback count missing every
# scramble, and feeding that in would quietly inflate every quarterback's attempt rate and sack rate and
# deflate his share of his team's dropbacks. A metric denominated in dropbacks is better served by three
# seasons of correct history than by two games of wrong arithmetic.
#
# A metric is fed from here only when *both* its numerator and its denominator are columns of this
# frame, and `estimate.own_rate` enforces that rather than a list kept in step by hand. The check is not
# a nicety. `blend_counts` fills a missing count with zero, so half a pair is worse than none of it: a
# quarterback's sacks with no dropbacks to divide by would add sacks to the numerator and nothing to the
# denominator, and report his sack rate as having doubled. Both sides or neither.
#
# history column -> the column of `player_weeks` it is summed from. Left side is what `priors.Metric`
# asks for, right side is what the weekly tables publish; the two disagree on exactly one spelling.
SKILL_TO_DATE = {
    "targets": "targets", "receptions": "receptions", "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds", "receiving_air_yards": "receiving_air_yards",
    "carries": "carries", "rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds",
    "offense_snaps": "offense_snaps", "fumbles_lost": "fumbles_lost",
}
QB_TO_DATE = {
    "attempts": "attempts", "completions": "completions", "passing_yards": "passing_yards",
    "passing_tds": "passing_tds", "interceptions": "interceptions",
    "passing_air_yards": "passing_air_yards", "sacks_suffered": "sacks",
    "carries": "carries", "rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds",
    "offense_snaps": "offense_snaps", "fumbles_lost": "fumbles_lost",
}

# history column -> the column of `team_pools` it is summed from. Read off the team table rather than
# summed from the players, so a man the crosswalk has not caught up with cannot shrink his own offence.
# A team's receiving touchdowns are its passing touchdowns, which is why one column serves both.
TEAM_TO_DATE = {
    "skill": {"team_targets": "targets", "team_carries": "carries",
              "team_air_yards": "passing_air_yards", "team_receiving_tds": "passing_tds",
              "team_rushing_tds": "rushing_tds"},
    "qb": {"team_pass_attempts": "attempts", "team_carries": "carries",
           "team_pass_tds": "passing_tds", "team_rushing_tds": "rushing_tds"},
}


@lru_cache(maxsize=8)
def team_snaps(season: int = PROJ_SEASON) -> pl.DataFrame:
    """Offensive snaps per team per completed game, recovered from the percentage beside each player.

    The snap table gives a player's snaps and the share of his team's they were, so the team's total is
    the one divided by the other. Taken as a median over the team's players rather than a single row: the
    percentage is rounded to two places upstream, so any one player implies the team total to within a
    snap or two and the middle of twenty of them is exact.
    """
    empty = pl.DataFrame(schema={"team": pl.String, "week": pl.Int32, "team_offense_snaps": pl.Float64})
    try:
        sc = _reg(lake.read("snap_counts", "raw", (season,)))
    except Exception:                       # noqa: BLE001
        return empty
    if sc.is_empty() or not {"offense_snaps", "offense_pct", "team", "week"} <= set(sc.columns):
        return empty
    return (
        sc.filter((pl.col("offense_pct").cast(pl.Float64) > 0.05)
                  & (pl.col("offense_snaps").cast(pl.Float64) > 0))
        .with_columns((pl.col("offense_snaps").cast(pl.Float64)
                       / pl.col("offense_pct").cast(pl.Float64)).alias("_implied"))
        .group_by(["team", "week"])
        .agg(pl.col("_implied").median().round(0).alias("team_offense_snaps"))
        .select(pl.col("team").cast(pl.String), pl.col("week").cast(pl.Int32), "team_offense_snaps")
    )


@lru_cache(maxsize=16)
def to_date(table: str, season: int = PROJ_SEASON) -> pl.DataFrame:
    """The season so far, shaped like one row of `history.skill_seasons` or `history.qb_seasons`.

    Which is the whole trick: the estimator is driven by `(numerator, denominator)` column pairs on a
    per-player-season frame, so a season-to-date row in the same shape is a season of history like any
    other and needs no special case anywhere downstream. `blend_counts` weights it by recency and shrinks
    by the opportunity in it, so two games of a receiver's targets carry two games of weight and grow into
    real evidence by November without anybody choosing a schedule for it.

    Only the columns in `SKILL_TO_DATE`/`QB_TO_DATE` and their team denominators, plus `touches` and
    `games`. A metric that needs a column not here keeps its preseason estimate -- see the note above
    those dicts, which is where the honest limits of the live tables are written down.
    """
    cols = SKILL_TO_DATE if table == "skill" else QB_TO_DATE
    schema: dict[str, pl.DataType] = {
        "player_id": pl.String, "season": pl.Int32, "team": pl.String, "games": pl.Float64,
        **{c: pl.Float64 for c in cols},
    }
    pw = player_weeks(season)
    if pw.is_empty():
        return pl.DataFrame(schema=schema)

    have = {name: mine for name, mine in cols.items() if mine in pw.columns}
    per_player = (
        pw.filter(pl.col("played") > 0)
        .group_by("player_id")
        .agg(
            pl.col("team").last(),
            pl.col("week").n_unique().cast(pl.Float64).alias("games"),
            *[pl.col(mine).sum().alias(name) for name, mine in have.items()],
        )
    )
    if per_player.is_empty():
        return pl.DataFrame(schema=schema)
    # what `fumble_rate` divides by: a carry or a catch is a chance to put it on the floor
    if {"carries", "receptions"} <= set(per_player.columns):
        per_player = per_player.with_columns(
            (pl.col("carries") + pl.col("receptions")).alias("touches")
        )

    # every team pool these metrics divide, summed over the games that team has played -- so a share is
    # measured against the same number of games it was earned in, and a team on a bye is not diluted
    pools = TEAM_TO_DATE["skill" if table == "skill" else "qb"]
    tp, ts = team_pools(season), team_snaps(season)
    team = pl.DataFrame(schema={"team": pl.String})
    agg = [pl.col(src).sum().alias(name) for name, src in pools.items() if src in tp.columns]
    if agg and not tp.is_empty():
        team = tp.group_by("team").agg(*agg)
    if not ts.is_empty():
        snap_pool = ts.group_by("team").agg(pl.col("team_offense_snaps").sum())
        team = snap_pool if team.is_empty() else team.join(snap_pool, on="team", how="left")

    out = per_player.join(team, on="team", how="left") if team.height else per_player
    return out.with_columns(pl.lit(season, pl.Int32).alias("season"))


def clear_cache() -> None:
    for fn in (team_weeks, player_weeks, snaps, crosswalk, team_pools, team_snaps, to_date):
        fn.cache_clear()


def status(season: int = PROJ_SEASON) -> dict:
    """How far the results go, for a header and for a test. Cheap enough to call on every page load."""
    tw = team_weeks(season)
    pw = player_weeks(season)
    return {
        "season": season,
        "weeks_complete": weeks_complete(season),
        "weeks_played": weeks_played(season),
        "team_games": tw.height,
        "teams": 0 if tw.is_empty() else tw["team"].n_unique(),
        "player_games": pw.height,
        "players": 0 if pw.is_empty() else pw["player_id"].n_unique(),
        "with_snaps": 0 if pw.is_empty() else int((pw["offense_snaps"] > 0).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--week", type=int, default=None, help="show one week rather than the summary")
    args = p.parse_args(argv)

    got = status(args.season)
    for key, val in got.items():
        print(f"  {key:<15} {val}")
    pw = player_weeks(args.season)
    if pw.is_empty():
        print("\nno results yet")
        return 0
    if args.week:
        pw = pw.filter(pl.col("week") == args.week)
    print()
    print(pw.sort("targets", descending=True).select(
        "player_id", "team", "week", "played", "offense_snaps", "targets", "receiving_yards",
        "carries", "rushing_yards", "attempts", "passing_yards",
    ).head(15).to_pandas().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
