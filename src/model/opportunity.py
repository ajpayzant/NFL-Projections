"""Who gets the ball, game by game -- and the arithmetic that makes the team's books balance.

A team's targets in a game are a fixed quantity: whatever the team environment projects, exactly that
many are thrown, and every one of them goes to exactly one player. So a projection is only coherent if

    Sigma over players of  P(he plays) x his share  =  1

That sum is what this module computes, reports, and -- optionally -- enforces. The workbook it replaces
warned when shares exceeded 100% and then left them there, so its team target totals were wrong by
whatever the warning said.

**Availability belongs inside the sum.** The share a player holds is conditional on his being active;
the pool is divided among whoever is actually out there. Multiplying by `active_weeks` before summing
is what lets a 90-man roster hold 28 receivers without projecting 40 receptions a game.

**The target is measured, not assumed.** Most pools are exclusive and sum to 1, but not all of them do
in the data: `team_receiving_tds` counts a two-point conversion catch that the touchdown pool does not,
and a lateral puts a carry on a receiver's line without leaving the carries pool. So each pool's target
is the sum actually observed over 2016-2025, and normalization scales toward that. Nothing is tuned to
make it come out at 1; the report shows how close each pool gets on its own.

    python -m src.model.opportunity            # per-pool audit for 2026, before and after scaling
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache

import polars as pl

from src.config import PROJ_SEASON, Settings
from src.data import history, lake
from src.model import estimate, priors, roster, team


@dataclass(frozen=True)
class Pool:
    """A per-game team quantity and the player-level shares that divide it up.

    `exclusive` says whether the shares listed here account for every event in the pool. Snaps and
    routes are shared -- five players are on the field for the same snap -- and designed runs are only
    partly claimed, since the QB's slice is fitted but the handoff's owner is folded into his overall
    carry share. Normalizing either kind would scale a number toward a total it was never measuring.
    """

    name: str                     # the player-level count this produces
    team_col: str                 # the per-game column in `team.game_environment`
    shares: tuple[str, ...]       # share metrics that divide it, across positions
    exclusive: bool = True


POOLS = (
    Pool("targets", "targets", ("target_share",)),
    Pool("carries", "carries", ("carry_share", "clean_rush_share")),
    Pool("air_yards", "air_yards", ("air_yards_share",)),
    Pool("rz_targets", "red_zone_targets", ("rz_target_share",)),
    Pool("rz_carries", "red_zone_carries", ("rz_carry_share",)),
    Pool("inside_5_carries", "inside_5_carries", ("inside_5_carry_share",)),
    Pool("short_yardage_carries", "short_yardage_carries", ("short_yardage_carry_share",)),
    Pool("late_down_targets", "late_down_targets", ("late_down_target_share",)),
    Pool("receiving_tds", "pass_tds", ("rec_td_share",)),
    Pool("passing_tds", "pass_tds", ("pass_td_share",)),
    Pool("rushing_tds", "rush_tds", ("rush_td_share", "qb_rush_td_share")),
    Pool("dropbacks", "dropbacks", ("dropback_share",)),
    # participation: several players share one snap, so these are not divided and not scaled
    Pool("offense_snaps", "plays", ("snap_share",), exclusive=False),
    Pool("routes", "dropbacks", ("route_participation",), exclusive=False),
    Pool("rush_plays", "designed_rushes", ("rush_participation",), exclusive=False),
    # not a claim on the pool but a pre-split quantity: only its ratio to the QB's projected scrambles
    # is used, to divide his clean rushes into designed runs and scrambles in `compose`.
    Pool("designed_qb_rushes", "designed_rushes", ("designed_rush_share",), exclusive=False),
)

BY_POOL = {p.name: p for p in POOLS}

# Every share metric any pool needs, plus the rates the composition layer asks for by name.
SHARE_METRICS = tuple(dict.fromkeys(s for p in POOLS for s in p.shares))


@lru_cache(maxsize=8)
def measure_targets(seasons: tuple[int, ...] | None = None) -> dict[str, float]:
    """What each pool's shares really summed to, per team-season, over the whole population.

    The honest denominator for a share sum: numerators from every player who took the field, the pool
    from the team's own totals. A pool that comes back at 0.98 is telling you 2% of its events went to
    somebody the player tables do not carry, and scaling to 1.0 would invent that 2%.
    """
    seasons = seasons or lake.history_seasons()
    pools = priors.team_pools(seasons)
    hist = {"skill": history.skill_seasons(seasons), "qb": history.qb_seasons(seasons)}
    out: dict[str, float] = {}
    for pool in POOLS:
        num = pl.lit(0.0)
        den = None
        for name in pool.shares:
            metric = priors.BY_NAME[name]
            h = hist[metric.table]
            got = (
                h.group_by(["season", "team"])
                .agg(pl.col(metric.num).sum().alias("n"))
                .rename({"n": f"n_{name}"})
            )
            pools = pools.join(got, on=["season", "team"], how="left")
            num = num + pl.col(f"n_{name}").fill_null(0.0)
            den = metric.den
        agg = pools.filter(pl.col(den) > 0).select((num / pl.col(den)).alias("r"))
        out[pool.name] = float(agg["r"].mean())
    return out


# Everything the player layer takes from the team layer, all of it prefixed `team_`. The prefix is not
# decoration: a pool and the count it produces share a name -- `targets` divided among players is also
# `targets` -- and without it a join silently leaves the team's 31 targets on a receiver's row.
TEAM_INPUTS = ("yards_per_attempt", "yards_per_carry", "success_rate", "est_yards_per_attempt",
               "est_yards_per_carry", "est_success_rate", "pass_attempts", "designed_rushes")

GAME_CONTEXT = ("game_id", "season", "week", "team", "opponent", "is_home", "rest_days", "div_game",
                "roof", "surface", "spread", "total", "implied_points", "has_market", "b_script")


def _game_frame(
    season: int,
    settings: Settings,
    rows: pl.DataFrame | None = None,
    env: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The team's per-game numbers, renamed `team_*` for the join onto players.

    `env` is passed in when a caller has already adjusted the environment -- the override layer edits
    a team's projected targets or pace and hands the edited frame down, rather than re-projecting it.
    `designed_rushes` is derived here rather than upstream, so an edit to `plays` or `dropbacks`
    carries into it instead of leaving the two contradicting each other.
    """
    env = team.game_environment(season, settings, rows=rows) if env is None else env
    # every snap is either a dropback or a designed run, so this is an identity rather than a second
    # projection: over 2016-2025 plays - dropbacks equals designed_rushes exactly, 127,798 either way.
    if "designed_rushes" not in env.columns and {"plays", "dropbacks"} <= set(env.columns):
        env = env.with_columns(
            (pl.col("plays") - pl.col("dropbacks")).clip(0.0).alias("designed_rushes")
        )
    cols = [c for c in GAME_CONTEXT if c in env.columns]
    wanted = dict.fromkeys([p.team_col for p in POOLS] + list(TEAM_INPUTS))
    return env.select(
        *cols, *[pl.col(c).alias(f"team_{c}") for c in wanted if c in env.columns]
    )


def player_shares(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    ros: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """One row per player, one column per share, from the single estimator in `estimate`."""
    settings = settings or Settings()
    ros = roster.roster(season) if ros is None else ros
    detail = estimate.estimate(SHARE_METRICS, ros, season, settings, fitted)
    return estimate.wide(detail, SHARE_METRICS)


def opportunity(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    shares: pl.DataFrame | None = None,
    part: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
    rows: pl.DataFrame | None = None,
    pool_targets: dict[str, float] | None = None,
    env: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per player per game: how many of each thing he is projected to get.

    `p_play` is his availability, from `roster.participation`, and it multiplies the share *before*
    the pool is divided. A pool's scale factor is therefore the honest answer to "do the people I
    expect on the field this week add up to a whole offence?".
    """
    settings = settings or Settings()
    part = roster.participation(season, settings, fitted=fitted) if part is None else part
    ros = part.select("season", "team", "player_id", "player", "position", "depth_slot",
                      "slot_bucket", "is_rookie", "draft_pick", "status", "expected_games",
                      pl.col("active_weeks").alias("p_play"))
    shares = player_shares(season, settings, fitted=fitted) if shares is None else shares
    games = _game_frame(season, settings, rows, env)

    have = [s for s in SHARE_METRICS if s in shares.columns]
    grid = ros.join(shares.select("player_id", *have), on="player_id", how="left").join(
        games, on=["season", "team"], how="inner"
    )
    targets = measure_targets() if pool_targets is None else pool_targets

    scaled = []
    for pool in POOLS:
        parts = [s for s in pool.shares if s in grid.columns]
        team_col = f"team_{pool.team_col}"
        if not parts or team_col not in grid.columns:
            continue
        raw = pl.sum_horizontal([pl.col(s).fill_null(0.0) for s in parts]) * pl.col("p_play")
        g = grid.with_columns(raw.alias(f"raw_{pool.name}"))
        g = g.with_columns(
            pl.col(f"raw_{pool.name}").sum().over(["game_id", "team"]).alias(f"sum_{pool.name}")
        )
        want = pl.lit(targets.get(pool.name, 1.0))
        # a pool nobody claims is left alone rather than divided by zero
        factor = (
            pl.when(pl.col(f"sum_{pool.name}") > 1e-9)
            .then(want / pl.col(f"sum_{pool.name}"))
            .otherwise(pl.lit(1.0))
        )
        use = factor if (settings.normalize_pools and pool.exclusive) else pl.lit(1.0)
        scaled.append(
            g.select(
                "game_id", "player_id",
                (pl.col(f"raw_{pool.name}") * use).alias(f"share_{pool.name}"),
                (pl.col(f"raw_{pool.name}") * use * pl.col(team_col)).alias(pool.name),
            )
        )
    out = grid.drop([c for c in SHARE_METRICS if c in grid.columns])
    for frame in scaled:
        out = out.join(frame, on=["game_id", "player_id"], how="left")
    return out.sort(["team", "week", "position", "depth_slot"])


def pool_audit(
    season: int = PROJ_SEASON, settings: Settings | None = None, opp: pl.DataFrame | None = None
) -> pl.DataFrame:
    """What every pool summed to before scaling, and what scaling therefore had to do.

    Normalization is never allowed to be silent: this is the table the app shows next to the toggle.
    """
    settings = settings or Settings()
    opp = opportunity(season, settings) if opp is None else opp
    targets = measure_targets()
    rows = []
    for pool in POOLS:
        col, share_col = pool.name, f"share_{pool.name}"
        if col not in opp.columns:
            continue
        per_team = opp.group_by(["game_id", "team"]).agg(
            pl.col(share_col).sum().alias("after"),
            pl.col(col).sum().alias("count"),
        )
        want = targets.get(col, 1.0)
        rows.append({
            "pool": col, "team_col": pool.team_col, "exclusive": pool.exclusive,
            "measured_target": want,
            "after_mean": float(per_team["after"].mean()),
            "after_min": float(per_team["after"].min()),
            "after_max": float(per_team["after"].max()),
            "count_per_game": float(per_team["count"].mean()),
            "normalized": bool(settings.normalize_pools and pool.exclusive),
        })
    return pl.DataFrame(rows)


def raw_pool_sums(
    season: int = PROJ_SEASON, settings: Settings | None = None
) -> pl.DataFrame:
    """The same audit with normalization off, which is the only way to see what it was hiding."""
    settings = settings or Settings()
    off = replace(settings, normalize_pools=False)
    return pool_audit(season, off).rename({"after_mean": "raw_mean", "after_min": "raw_min",
                                           "after_max": "raw_max"})


def team_pool_sums(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    shares: pl.DataFrame | None = None,
    part: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """What one team's shares sum to before any scaling. One row per team and pool.

    The league-wide `pool_report` says whether the estimator is calibrated; this says which *rosters*
    it is struggling with, which is a different and more actionable question. A team summing short has
    a job its depth chart does not name -- a vacated target share nobody has inherited -- and that is
    a real finding about the roster rather than a defect in the scaling.

    Computed from shares and availability directly rather than by re-running `opportunity` with
    normalization off. It is the same quantity: a share and an availability are both season-level, so
    a pool's raw sum is identical in all 17 games, which is why `opportunity` can take the factor from
    any one of them.
    """
    settings = settings or Settings()
    part = roster.participation(season, settings, fitted=fitted) if part is None else part
    shares = player_shares(season, settings, fitted=fitted) if shares is None else shares
    have = [s for s in SHARE_METRICS if s in shares.columns]
    grid = part.select("team", "player_id", pl.col("active_weeks").alias("p_play")).join(
        shares.select("player_id", *have), on="player_id", how="left"
    )
    targets = measure_targets()

    frames = []
    for pool in POOLS:
        parts = [s for s in pool.shares if s in grid.columns]
        if not parts:
            continue
        want = targets.get(pool.name, 1.0)
        raw = pl.sum_horizontal([pl.col(s).fill_null(0.0) for s in parts]) * pl.col("p_play")
        frames.append(
            grid.group_by("team").agg(raw.sum().alias("raw_sum"), pl.len().alias("players"))
            .with_columns(
                pl.lit(pool.name).alias("pool"),
                pl.lit(pool.exclusive).alias("exclusive"),
                pl.lit(want).alias("measured_target"),
                pl.lit(settings.normalize_pools and pool.exclusive).alias("normalized"),
            )
        )
    out = pl.concat(frames, how="vertical")
    return out.with_columns(
        pl.when(pl.col("raw_sum") > 1e-9)
        .then(pl.col("measured_target") / pl.col("raw_sum")).otherwise(pl.lit(1.0)).alias("factor"),
        (100.0 * (pl.col("raw_sum") - pl.col("measured_target"))
         / pl.col("measured_target")).alias("gap_pct"),
    ).select("team", "pool", "exclusive", "measured_target", "raw_sum", "factor", "gap_pct",
             "normalized").sort(["team", "pool"])


def pool_report(
    season: int = PROJ_SEASON, settings: Settings | None = None, opp: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Before and after in one table: what the shares summed to, and what scaling did about it.

    `gap_pct` is the honest headline -- how far the estimated shares were from the pool they divide
    before anything was rescaled. This is the table the app puts next to the normalization toggle, so
    it lives here rather than being assembled in the page.
    """
    settings = settings or Settings()
    raw = raw_pool_sums(season, settings)
    after = pool_audit(season, settings, opp)
    return (
        raw.select("pool", "team_col", "exclusive", "measured_target",
                   "raw_mean", "raw_min", "raw_max")
        .join(after.select("pool", "after_mean", "count_per_game", "normalized"), on="pool")
        .with_columns(
            (100.0 * (pl.col("raw_mean") - pl.col("measured_target"))
             / pl.col("measured_target")).alias("gap_pct")
        )
        .sort("pool")
    )


def _report(season: int, settings: Settings) -> None:
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_fmt_float("mixed")
    opp = opportunity(season, settings)
    print(f"\nPOOL SUMS  {season}: availability-weighted shares per team-game, before and after scaling")
    print(pool_report(season, settings, opp).drop("team_col").with_columns(
        pl.col("^raw_.*$").round(3), pl.col("measured_target").round(3),
        pl.col("after_mean").round(3), pl.col("count_per_game").round(2),
        pl.col("gap_pct").round(1),
    ))

    print(f"\nOPPORTUNITY  {opp.height:,} player-games, {opp['player_id'].n_unique()} players")
    ex = opp.filter((pl.col("team") == "PHI") & (pl.col("week") == 1))
    cols = [c for c in ("targets", "carries", "rz_carries", "inside_5_carries", "receiving_tds",
                        "rushing_tds", "dropbacks", "routes") if c in ex.columns]
    print("\nPHI, week 1")
    print(ex.select("position", "depth_slot", "player", pl.col("p_play").round(2),
                    *[pl.col(c).round(2) for c in cols])
          .sort(["position", "depth_slot"]).head(20))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--no-normalize", action="store_true", help="show the pools unscaled")
    a = p.parse_args(argv)
    _report(a.season, Settings(normalize_pools=not a.no_normalize))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
