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

**A pool only one man can take is a queue, not a committee.** Where several players genuinely share a
pool, a room that claims 106% is best read as everyone being slightly high, so the correction is
proportional. A dropback is not like that -- one quarterback takes it, and the depth chart is a strict
order -- so an over-claiming room there is an over-claiming *backup*, and charging the starter a share
of it made the model contradict its own availability estimate. Those pools are filled in depth order
instead; see `Pool.queue`.

**Even in a committee the members are not equally wrong.** "Everyone is a little high" was the second
approximation, and it is measurably too crude: held out against 2021-2025, a receiver projected 2-4
targets a game is over-projected by 17-21% while one projected 6-8 is over by 2-4%. A flat percentage
therefore under-corrects the bench and over-corrects the starter, and does it in the one direction that
matters, since the starter is the number anybody reads. So the residual is charged in proportion to
`share ** pool_tilt` rather than to the share itself -- see `_tilt_alloc` for the arithmetic and
`Settings.pool_tilt` for the exponent, which is fitted and whose value 1.0 is the flat rescale exactly.

**A share somebody typed is held at what they typed.** With `Settings.lock_edited_shares` an explicit
override is settled against the pool *first* and the residual is taken from the rest of the room. The
alternative is what this replaced: a user setting a receiver to a 0.300 target share, the rescale
quietly delivering 0.2545, and the provenance log reporting the edit applied at 0.300. The pool is
still exactly whole -- somebody else pays -- which is the point of overriding one player and letting
the room settle around him.

    python -m src.model.opportunity            # per-pool audit for 2026, before and after scaling
    python -m src.model.backtest --fit-tilt    # refit the tilt exponent on held-out seasons
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache

import polars as pl

from src.config import FITTED, PROJ_SEASON, Settings
from src.data import history, lake
from src.model import estimate, priors, roster, team


@dataclass(frozen=True)
class Pool:
    """A per-game team quantity and the player-level shares that divide it up.

    `exclusive` says whether the shares listed here account for every event in the pool. Snaps and
    routes are shared -- five players are on the field for the same snap -- and designed runs are only
    partly claimed, since the QB's slice is fitted but the handoff's owner is folded into his overall
    carry share. Normalizing either kind would scale a number toward a total it was never measuring.

    `queue` says *how* an exclusive pool is divided when the claims do not add up. The default spreads
    the disagreement across the room, which is right for a pool several men genuinely share: if a
    team's receivers claim 106% of its targets, each of them is somewhat high. It is wrong for a pool
    exactly one man can take. A dropback goes to one quarterback, and the depth chart is a strict
    order, so a room that over-claims is a *backup* who is over-claiming -- and spreading it charges
    the starter for it. Under a queue the first-stringer is filled to his claim, the next man gets only
    what is left, and the error lands where it belongs. See `_queue_alloc`.

    "Across the room" is not "equally": a committee's share estimates are more over-stated the smaller
    they are, so the residual is weighted by `share ** Settings.pool_tilt`. See `_tilt_alloc`.

    A queued pool must be claimed by one position only; `tests/test_opportunity.py` asserts it, since
    the queue reads `depth_slot`, which is ranked within a position and not across the offence.
    """

    name: str                     # the player-level count this produces
    team_col: str                 # the per-game column in `team.game_environment`
    shares: tuple[str, ...]       # share metrics that divide it, across positions
    exclusive: bool = True
    queue: bool = False


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
    Pool("passing_tds", "pass_tds", ("pass_td_share",), queue=True),
    Pool("rushing_tds", "rush_tds", ("rush_td_share", "qb_rush_td_share")),
    Pool("dropbacks", "dropbacks", ("dropback_share",), queue=True),
    # participation: several players share one snap, so these are not divided and not scaled
    Pool("offense_snaps", "offense_snaps", ("snap_share", "qb_snap_share"), exclusive=False),
    Pool("routes", "dropbacks", ("route_participation",), exclusive=False),
    # There was a third here, `rush_plays` over `rush_participation`, and it was not participation at
    # all. The play-by-play credits a designed run to the man who carried it, so the column sums to one
    # per run rather than to the five or six players on the field for it -- `carries` under another name,
    # correlating 0.98 with `carry_share` among backs and predicting next season's carry share no better
    # (0.640 against 0.642). It was also populated for RB and FB only, so every receiver and tight end
    # in the league carried a measured-looking 0.000 and a fitted prior of exactly zero stated over
    # 55,000 designed runs. It reached no projected stat, it duplicated one that does, and the app
    # labelled it "runs he is on the field for", which it never was.
    # not a claim on the pool but a pre-split quantity: only its ratio to the QB's projected scrambles
    # is used, to divide his clean rushes into designed runs and scrambles in `compose`.
    Pool("designed_qb_rushes", "designed_rushes", ("designed_rush_share",), exclusive=False),
)

BY_POOL = {p.name: p for p in POOLS}

# Every share metric any pool needs, plus the rates the composition layer asks for by name.
SHARE_METRICS = tuple(dict.fromkeys(s for p in POOLS for s in p.shares))

# The unit every pool is divided within: one team, in one game. Named because four functions have to
# group by exactly the same thing and a pool that was summed over the wrong window is silently wrong.
ROOM = ["game_id", "team"]

# A boolean column named `lock_<share>` says this man's share was typed rather than estimated, and the
# pool division holds it at what was typed. Written by `overrides.apply`; the prefix is shared with it
# through this constant so neither side can drift.
LOCK_PREFIX = "lock_"

TILT_PATH = FITTED / "pool_tilt.json"


@lru_cache(maxsize=1)
def load_tilt() -> dict:
    """The fitted tilt exponent. 1.0 when nothing has been fitted, which is the flat rescale."""
    if not TILT_PATH.exists():
        return {"pool_tilt": 1.0}
    return json.loads(TILT_PATH.read_text(encoding="utf-8"))


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

# A snap is not a play: an offence takes the field for snaps wiped out by an accepted penalty, which
# never become plays. Measured 2021-2025 the gap is flat -- 65.5 snaps per 61.7 plays, 1.060 to 1.067
# by season and sd 0.005 across the 32 teams -- so it is a unit conversion, not a second projection.
# Without it a snap share measured against team snaps was being paid out of team plays and every snap
# count in the app read ~6% light.
SNAPS_PER_PLAY = 1.062

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
    if "offense_snaps" not in env.columns and "plays" in env.columns:
        env = env.with_columns((pl.col("plays") * SNAPS_PER_PLAY).alias("offense_snaps"))
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


def _queue_alloc(claim: pl.Expr, want: pl.Expr) -> pl.Expr:
    """Fill the depth chart in order and give each man only what the men ahead of him left.

        alloc_i = min(claim_i, max(0, want - Sigma_(j<i) claim_j))

    The alternative -- scaling everyone by the same factor -- is the defect this exists to fix. A
    quarterback room is a queue, not a committee: the starter takes every dropback he is available for
    and the backup takes the rest, so an over-claiming backup is *his own* error. Proportional scaling
    spread it over the room and left the model contradicting itself, projecting a starter 15.6 games
    and then 73% of his team's dropbacks.

    A zero claim consumes nothing and receives nothing, so the players who are not in this pool at all
    sit in the ordering harmlessly. If the room *under*-claims the caller's proportional scale-up still
    runs afterwards and lifts it to the target -- somebody has to throw the passes -- so the pool is
    conserved either way.
    """
    ahead = claim.cum_sum().over(ROOM, order_by=["depth_slot", "player_id"]) - claim
    return pl.min_horizontal(claim, (want - ahead).clip(lower_bound=0.0))


def _tilt_weight(claim: pl.Expr, p_play: pl.Expr, tilt: float) -> pl.Expr:
    """How much of a room's disagreement each man is charged, before it is normalised.

    The evidence is about a *share*, not about a claim: held-out error is measured per game played, so
    what is over-stated by 20% is a bench receiver's 3 targets a game and not his 3-times-nine-games.
    A man's claim is `share x p_play`, so undoing the availability and re-applying it linearly gives

        weight = share ** tilt  x  p_play  =  claim ** tilt  x  p_play ** (1 - tilt)

    which is the form used here because it makes the identity obvious: at `tilt = 1` the weight is the
    claim itself and the whole allocation collapses to the flat rescale. Availability stays linear on
    purpose -- a man who plays half a season can only be half as wrong about it, which is a counting
    fact and not something the exponent should be allowed to bend.
    """
    if tilt == 1.0:
        return claim
    return claim.clip(lower_bound=0.0) ** tilt * p_play.clip(0.0, 1.0) ** (1.0 - tilt)


def _tilt_alloc(claim: pl.Expr, weight: pl.Expr, want: pl.Expr) -> pl.Expr:
    """Settle a room against the pool it divides, charging the disagreement by `weight`.

        take_i  = min(claim_i, max(0, (Sigma claim - want) x weight_i / Sigma weight))
        alloc_i = (claim_i - take_i) x want / Sigma (claim - take)

    Three properties, all of them load-bearing and all of them pinned by `tests/test_opportunity.py`:

    - **`weight = claim` is the flat rescale, exactly.** Then `take_i = claim_i x over / Sigma claim`
      (the `min` never bites, because the overclaim is smaller than the claim), so the first line alone
      leaves the room summing to `want` and the second is a factor of 1. That is why `pool_tilt = 1.0`
      is the fallback rather than a value: it changes no number anywhere.
    - **The pool comes out exact.** A tilted `take` can exceed a small claim -- charging a man 0.003 of
      a pool he only claims 0.001 of -- and clipping it at his claim leaves the room short of paying.
      The second line is what closes that gap, spread over whoever still has something, and it cannot
      resurrect a man clipped to zero. One pass, no iteration, exact by construction.
    - **An under-claiming room is scaled up, not tilted.** `over` is then negative, the clip takes
      every `take` to zero, and the second line lifts the room proportionally: somebody has to catch
      the passes, and there is no evidence about who deserves a share of a shortfall.
    """
    over = claim.sum().over(ROOM) - want
    wsum = weight.sum().over(ROOM)
    take = pl.min_horizontal(
        claim,
        pl.when(wsum > 1e-12).then(over * weight / wsum).otherwise(pl.lit(0.0)),
    ).clip(lower_bound=0.0)
    rest = claim - take
    left = rest.sum().over(ROOM)
    # a pool nobody has any claim left on is left alone rather than divided by zero
    return pl.when(left > 1e-12).then(rest * want / left).otherwise(rest)


def _settle(claim: pl.Expr, weight: pl.Expr, want: pl.Expr, locked: pl.Expr | None,
            queue: bool) -> pl.Expr:
    """One room's final claims: the locked ones at what was typed, the rest settled around them.

    A lock is honoured out of the pool first, so the residual the others carry is `want` minus what the
    locks took. That is the whole behaviour a user asked for by overriding one player: his number is
    his number, and his teammates absorb it.

    Locks are still not allowed to break the pool. A room whose *typed* shares alone claim more than
    the pool holds cannot have all of them, so they are scaled against each other and the un-edited men
    get nothing -- which is a projection of what was asked for, reported by the pool audit as a room
    that sums to its target with nobody left in it, rather than a team throwing more passes than it was
    projected to throw.
    """
    held = pl.lit(0.0) if locked is None else pl.when(locked).then(claim).otherwise(pl.lit(0.0))
    held_sum = held.sum().over(ROOM)
    free = claim if locked is None else pl.when(locked).then(pl.lit(0.0)).otherwise(claim)
    free_w = weight if locked is None else pl.when(locked).then(pl.lit(0.0)).otherwise(weight)
    left = (want - held_sum).clip(lower_bound=0.0)

    if queue:
        # the queue is itself the answer to who pays, so what follows it is a plain rescale: the room is
        # filled in depth order against whatever the locks left, and the tilt has no business in a pool
        # exactly one man can take
        filled = _queue_alloc(free, left)
        settled = _tilt_alloc(filled, filled, left)
    else:
        settled = _tilt_alloc(free, free_w, left)
    if locked is None:
        return settled
    kept = pl.when(held_sum > want).then(want / held_sum).otherwise(pl.lit(1.0))
    return pl.when(locked).then(claim * kept).otherwise(settled)


PLAYING_TIME = "playing_time"

# The pool whose share *is* the playing time. Scaling a claim on it by how far that claim was moved
# would square the edit: a snap share set to twice the estimate would pay out four times the snaps.
PLAYING_TIME_POOL = "offense_snaps"


def _playing_time(grid: pl.DataFrame, pool: Pool, share: str, settings: Settings) -> pl.Expr:
    """How much a snap-share edit scales this claim on this pool. `1.0` unless somebody typed one.

    Written by `overrides.playing_time`, which is the only thing that knows the estimate the edit moved
    away from. The share he typed himself is exempt for the reason `lock_edited_shares` exists: a number
    a user stated is the number the projection uses, and scaling a typed target share by a typed snap
    share would be the app disagreeing with both of them.
    """
    if PLAYING_TIME not in grid.columns or pool.name == PLAYING_TIME_POOL:
        return pl.lit(1.0)
    scale = pl.col(PLAYING_TIME).fill_null(1.0)
    held = f"{LOCK_PREFIX}{share}"
    if held in grid.columns and settings.lock_edited_shares:
        return pl.when(pl.col(held).fill_null(False)).then(pl.lit(1.0)).otherwise(scale)
    return scale


def tilt_of(settings: Settings) -> float:
    """The exponent in force: the scenario's if it set one, otherwise the fitted one, otherwise flat."""
    if settings.pool_tilt is not None:
        return float(settings.pool_tilt)
    return float(load_tilt().get("pool_tilt", 1.0))


def opportunity(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    shares: pl.DataFrame | None = None,
    part: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
    rows: pl.DataFrame | None = None,
    pool_targets: dict[str, float] | None = None,
    env: pl.DataFrame | None = None,
    on_grid: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """One row per player per game: how many of each thing he is projected to get.

    `p_play` is his availability, from `roster.participation`, and it multiplies the share *before*
    the pool is divided. A pool's scale factor is therefore the honest answer to "do the people I
    expect on the field this week add up to a whole offence?".

    `on_grid` is handed the player-game frame after the shares and the team's numbers have been joined
    onto it and *before* any pool is divided. That is the seam a per-game edit belongs at: the override
    layer uses it to say "he is out in week 5" or "he runs the routes this week", and because the
    division has not happened yet, the pool still comes out whole -- the room absorbs what he is not
    taking, the quarterback queue promotes the backup for that week only, and the counts are re-derived
    rather than typed over. Applied after the division, the same edit would leave a team's targets not
    adding up to the targets it is projected to throw.
    """
    settings = settings or Settings()
    part = roster.participation(season, settings, fitted=fitted) if part is None else part
    ros = part.select("season", "team", "player_id", "player", "position", "depth_slot",
                      "slot_bucket", "is_rookie", "draft_pick", "status", "expected_games",
                      pl.col("active_weeks").alias("p_play"))
    shares = player_shares(season, settings, fitted=fitted) if shares is None else shares
    games = _game_frame(season, settings, rows, env)

    have = [s for s in SHARE_METRICS if s in shares.columns]
    # the lock markers ride along with the shares they belong to, so a share held at a typed value and
    # the marker saying it was typed can never end up on different rows
    marks = [f"{LOCK_PREFIX}{s}" for s in have if f"{LOCK_PREFIX}{s}" in shares.columns]
    pt = [PLAYING_TIME] if PLAYING_TIME in shares.columns else []
    grid = ros.join(shares.select("player_id", *have, *marks, *pt), on="player_id", how="left").join(
        games, on=["season", "team"], how="inner"
    )
    if on_grid is not None:
        grid = on_grid(grid)
    targets = measure_targets() if pool_targets is None else pool_targets
    tilt = tilt_of(settings)

    scaled = []
    for pool in POOLS:
        parts = [s for s in pool.shares if s in grid.columns]
        team_col = f"team_{pool.team_col}"
        if not parts or team_col not in grid.columns:
            continue
        raw_col = f"raw_{pool.name}"
        raw = pl.sum_horizontal(
            [pl.col(s).fill_null(0.0) * _playing_time(grid, pool, s, settings) for s in parts]
        ) * pl.col("p_play")
        g = grid.with_columns(raw.alias(raw_col))
        want = pl.lit(targets.get(pool.name, 1.0))
        normalize = settings.normalize_pools and pool.exclusive
        if normalize:
            held = [f"{LOCK_PREFIX}{s}" for s in parts if f"{LOCK_PREFIX}{s}" in g.columns]
            locked = (
                pl.any_horizontal([pl.col(c).fill_null(False) for c in held])
                if held and settings.lock_edited_shares else None
            )
            alloc = _settle(
                pl.col(raw_col),
                _tilt_weight(pl.col(raw_col), pl.col("p_play"), tilt),
                want, locked, pool.queue,
            )
        else:
            alloc = pl.col(raw_col)
        g = g.with_columns(alloc.alias(raw_col))
        scaled.append(
            g.select(
                "game_id", "player_id",
                pl.col(raw_col).alias(f"share_{pool.name}"),
                (pl.col(raw_col) * pl.col(team_col)).alias(pool.name),
            )
        )
    # the shares are kept rather than dropped: they are the *input* to every count on the row, and a
    # per-game edit lands on them, so a reader -- or an editing grid -- can see the number that was
    # divided beside what it produced. They are named nothing like the counts (`target_share` against
    # `targets` and `share_targets`), so nothing downstream can confuse the three.
    out = grid
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
            "queued": bool(pool.queue),
            # 1.0 says the residual was spread flat; a queued pool ignores it either way
            "tilt": 1.0 if pool.queue or not pool.exclusive else tilt_of(settings),
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

    `factor` is the *flat* correction the raw sum implies, which is the size of the disagreement rather
    than what is done about it. A `queued` pool is filled in depth order instead, so the whole
    correction lands on the last men in the queue; a committee charges the residual by
    `share ** pool_tilt`, so a starter pays less than this factor and a bench player more. Read it as
    "how far out is this room", not as anybody's multiplier.
    """
    settings = settings or Settings()
    part = roster.participation(season, settings, fitted=fitted) if part is None else part
    shares = player_shares(season, settings, fitted=fitted) if shares is None else shares
    have = [s for s in SHARE_METRICS if s in shares.columns]
    pt = [PLAYING_TIME] if PLAYING_TIME in shares.columns else []
    grid = part.select("team", "player_id", pl.col("active_weeks").alias("p_play")).join(
        shares.select("player_id", *have, *pt), on="player_id", how="left"
    )
    targets = measure_targets()

    frames = []
    for pool in POOLS:
        parts = [s for s in pool.shares if s in grid.columns]
        if not parts:
            continue
        want = targets.get(pool.name, 1.0)
        # the same scaling the projection applies, or the audit would report a room the projection
        # never divided
        raw = pl.sum_horizontal(
            [pl.col(s).fill_null(0.0) * _playing_time(grid, pool, s, settings) for s in parts]
        ) * pl.col("p_play")
        frames.append(
            grid.group_by("team").agg(raw.sum().alias("raw_sum"), pl.len().alias("players"))
            .with_columns(
                pl.lit(pool.name).alias("pool"),
                pl.lit(pool.exclusive).alias("exclusive"),
                pl.lit(want).alias("measured_target"),
                pl.lit(settings.normalize_pools and pool.exclusive).alias("normalized"),
                pl.lit(pool.queue).alias("queued"),
            )
        )
    out = pl.concat(frames, how="vertical")
    return out.with_columns(
        pl.when(pl.col("raw_sum") > 1e-9)
        .then(pl.col("measured_target") / pl.col("raw_sum")).otherwise(pl.lit(1.0)).alias("factor"),
        (100.0 * (pl.col("raw_sum") - pl.col("measured_target"))
         / pl.col("measured_target")).alias("gap_pct"),
    ).select("team", "pool", "exclusive", "measured_target", "raw_sum", "factor", "gap_pct",
             "normalized", "queued").sort(["team", "pool"])


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
        .join(after.select("pool", "after_mean", "count_per_game", "normalized", "queued", "tilt"),
              on="pool")
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

    print("\nQUARTERBACK ROOMS  season shares of the dropback pool, by depth slot")
    qb = (opp.filter(pl.col("position") == "QB")
          .group_by(["team", "player", "depth_slot"])
          .agg(pl.col("p_play").first(), pl.col("share_dropbacks").mean().alias("share"),
               pl.col("dropbacks").sum().alias("dropbacks")))
    print(qb.group_by("depth_slot").agg(
        pl.len().alias("n"), pl.col("share").mean().round(3).alias("share_mean"),
        pl.col("share").min().round(3).alias("share_min"),
        pl.col("share").max().round(3).alias("share_max"),
        pl.col("dropbacks").mean().round(0).alias("dropbacks"),
    ).sort("depth_slot"))
    print("\nlowest-projected starters")
    print(qb.filter(pl.col("depth_slot") == 1).sort("share").head(8).select(
        "team", "player", pl.col("p_play").round(3), pl.col("share").round(3),
        pl.col("dropbacks").round(0)))

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
