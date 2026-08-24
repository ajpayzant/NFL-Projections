"""Depth charts, one shape, across two incompatible eras.

The NFL changed how it publishes charts in 2025 and the nflverse table carries both layouts:

- **2016-2024**: weekly rows keyed on `club_code`, `position`, `depth_position` (the alignment: LT,
  RG, WR) and `depth_team` (rank *within that alignment*). A team's three starting receivers are all
  `WR / depth_team=1` -- the data cannot tell WR1 from WR3.
- **2025 on**: dated snapshots keyed on `team` and `pos_abb`, with `pos_rank` ranking the whole
  position group (WR 1..12) and `pos_slot` giving the alignment. Ranks run slot-major, so ranks 1-3
  are the three starting receivers and 4-6 are their backups.

Both reduce to a **tier** -- 1 for starters, 2 for the next man up -- which is the finest grain the
legacy era supports. Tier alone is too coarse to project with, because a WR1 and a WR3 share tier 1
and almost nothing else, so `depth_slot` breaks ties inside a tier by prior-season opportunity per
game and then by draft pick. Both tie-breakers are known before the season starts, which is what
keeps a slot usable as a projection input rather than a peek at the answer.

Snapshot choice is deliberate: `when="preseason"` takes the last chart published *before* week 1, so
a backtest sees what a user would have seen in August. `when="latest"` takes the newest chart there
is, which for the projection season is the point.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

from src.config import PROJ_SEASON, TEAM_FIXUP
from src.data import lake

LEGACY_LAST_SEASON = 2024

# Offensive positions we track a depth order for. FB shares the RB room.
TRACKED = ("QB", "RB", "FB", "WR", "TE")

# How many players a team starts at each position. This is what turns a position-wide rank into a
# tier: WR rank 4 is a tier-2 receiver, RB rank 2 is already tier 2.
STARTERS = {"QB": 1, "RB": 1, "FB": 1, "WR": 3, "TE": 1}


def _team(col: str) -> pl.Expr:
    return pl.col(col).replace(TEAM_FIXUP).alias("team")


@lru_cache(maxsize=4)
def _season_starts() -> dict[int, str]:
    """First kickoff date of each regular season, for picking a pre-week-1 snapshot."""
    sched = lake.read("schedules", layer="raw")
    col = "gameday" if "gameday" in sched.columns else "game_date"
    starts = (
        sched.filter(pl.col("game_type") == "REG")
        .group_by("season")
        .agg(pl.col(col).min().alias("start"))
    )
    return {int(r["season"]): str(r["start"]) for r in starts.iter_rows(named=True)}


def _legacy(df: pl.DataFrame, season: int, when: str) -> pl.DataFrame:
    d = df.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    week = 1 if when == "preseason" else int(d["week"].max() or 1)
    d = d.filter(pl.col("week") == week)
    return d.select(
        pl.lit(season, pl.Int32).alias("season"),
        _team("club_code"),
        pl.col("gsis_id").alias("player_id"),
        pl.col("position").alias("chart_position"),
        pl.col("depth_position").alias("alignment"),
        pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth_tier"),
        pl.lit(f"week {week}").alias("snapshot"),
    )


def _modern(df: pl.DataFrame, season: int, when: str) -> pl.DataFrame:
    d = df.filter(pl.col("season") == season)
    start = _season_starts().get(season)
    if when == "preseason" and start is not None:
        before = d.filter(pl.col("dt") < start)
        d = before if not before.is_empty() else d
    snapshot = d["dt"].max()
    d = d.filter(pl.col("dt") == snapshot)

    # pos_rank runs slot-major within a position, so integer-dividing by the number of starters at
    # that position recovers the same tier the legacy layout states outright.
    starters = pl.col("pos_abb").replace_strict(STARTERS, default=1, return_dtype=pl.Int32)
    return d.select(
        pl.lit(season, pl.Int32).alias("season"),
        _team("team"),
        pl.col("gsis_id").alias("player_id"),
        pl.col("pos_abb").alias("chart_position"),
        pl.col("pos_name").alias("alignment"),
        ((pl.col("pos_rank").cast(pl.Int32) - 1) // starters + 1).alias("depth_tier"),
        pl.lit(str(snapshot)[:10]).alias("snapshot"),
    )


@lru_cache(maxsize=32)
def depth_chart(season: int, when: str = "latest") -> pl.DataFrame:
    """One row per player on a team's offensive depth chart, with `depth_tier` and `depth_slot`.

    `depth_slot` is the projection input: 1 = the team's clear starter at that position, and for
    receivers 1/2/3 separate the three starters rather than lumping them.
    """
    if when not in ("preseason", "latest"):
        raise ValueError(f"when must be 'preseason' or 'latest', got {when!r}")
    df = lake.read("depth_charts", layer="raw", seasons=(season,))
    chart = _legacy(df, season, when) if season <= LEGACY_LAST_SEASON else _modern(df, season, when)
    chart = (
        chart.filter(pl.col("chart_position").is_in(TRACKED) & pl.col("player_id").is_not_null())
        .with_columns(pl.col("chart_position").replace({"FB": "RB"}).alias("position"))
        # a player listed at two alignments (a guard who is also the backup centre; a receiver on two
        # slot lines) keeps his best listing. `alignment` is in the sort key and `maintain_order` is
        # on because polars' `unique` only honours `keep="first"` against a defined order, and an
        # undefined one here moves a player's tier between runs of the same fit.
        .sort(["season", "team", "position", "depth_tier", "alignment"], maintain_order=True)
        .unique(subset=["season", "team", "position", "player_id"], keep="first",
                maintain_order=True)
    )
    return _resolve_slots(chart, season)


def _tiebreakers(season: int) -> pl.DataFrame:
    """Prior-season opportunity per game and draft pick -- both known before week 1."""
    from src.data import history

    seasons = tuple(range(max(2016, season - 3), season))
    frames = []
    if seasons:
        sk = history.skill_seasons(seasons)
        frames.append(
            sk.group_by("player_id").agg(
                ((pl.col("targets").sum() + pl.col("carries").sum()) / pl.col("games").sum())
                .alias("prior_opp_per_game")
            )
        )
        qb = history.qb_seasons(seasons)
        frames.append(
            qb.group_by("player_id").agg(
                (pl.col("dropbacks").sum() / pl.col("games").sum()).alias("prior_opp_per_game")
            )
        )
    prior = (
        pl.concat(frames).group_by("player_id").agg(pl.col("prior_opp_per_game").max())
        if frames
        else pl.DataFrame(schema={"player_id": pl.String, "prior_opp_per_game": pl.Float64})
    )

    draft = lake.read("draft_picks", layer="raw").filter(pl.col("gsis_id").is_not_null())
    draft = (
        draft.group_by(pl.col("gsis_id").alias("player_id"))
        .agg(pl.col("pick").min().alias("draft_pick"), pl.col("season").min().alias("draft_season"))
    )
    return prior.join(draft, on="player_id", how="full", coalesce=True)


def _resolve_slots(chart: pl.DataFrame, season: int) -> pl.DataFrame:
    tb = _tiebreakers(season)
    return (
        chart.join(tb, on="player_id", how="left")
        .with_columns(
            # an undrafted player sorts behind every drafted one rather than ahead of pick 1
            pl.col("draft_pick").fill_null(400).alias("_pick"),
            pl.col("prior_opp_per_game").fill_null(-1.0).alias("_opp"),
        )
        # `player_id` is the last key and not a football judgement: two undrafted rookies in the same
        # tier tie on both real tiebreakers, and without a final one the slot order flips between runs
        # of the identical fit, moving every prior built on it. Reproducibility beats elegance here.
        .sort(
            ["season", "team", "position", "depth_tier", "_opp", "_pick", "player_id"],
            descending=[False, False, False, False, True, False, False],
            maintain_order=True,
        )
        .with_columns(
            (pl.int_range(pl.len()) + 1)
            .over(["season", "team", "position"])
            .cast(pl.Int32)
            .alias("depth_slot")
        )
        .drop("_pick", "_opp")
    )


def latest_snapshot_date(season: int = PROJ_SEASON) -> str:
    return depth_chart(season)["snapshot"][0]


def clear_cache() -> None:
    depth_chart.cache_clear()
    _tiebreakers.cache_clear() if hasattr(_tiebreakers, "cache_clear") else None
    _season_starts.cache_clear()


if __name__ == "__main__":
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_rows(30)
    for season, when in ((2022, "preseason"), (2026, "latest")):
        ch = depth_chart(season, when)
        print(f"\n== {season} ({when}) {ch.shape}  snapshot {ch['snapshot'][0]}")
        print(ch.group_by("position").agg(pl.len().alias("players"), pl.col("team").n_unique().alias("teams")).sort("position"))
        print(ch.filter((pl.col("team") == "PHI") & (pl.col("position").is_in(["WR", "RB", "TE", "QB"])))
              .select("position", "depth_tier", "depth_slot", "player_id", "prior_opp_per_game", "draft_pick")
              .sort(["position", "depth_slot"]).head(14))
