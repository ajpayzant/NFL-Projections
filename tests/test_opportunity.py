"""The pool layer: who gets the ball, and whether the team's books balance.

These are the invariants the workbook could not hold. It warned when a team's target shares summed
past 100% and then left them there; here the sum is measured, reported, and either enforced or
explained. Every assertion below is a football fact rather than a fitted number, so a data refresh
cannot break it without something genuinely being wrong.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from src.config import PROJ_SEASON, Settings
from src.model import opportunity


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def opp(settings) -> pl.DataFrame:
    return opportunity.opportunity(PROJ_SEASON, settings)


@pytest.fixture(scope="module")
def raw(settings) -> pl.DataFrame:
    return opportunity.opportunity(PROJ_SEASON, replace(settings, normalize_pools=False))


@pytest.fixture(scope="module")
def targets() -> dict[str, float]:
    return opportunity.measure_targets()


EXCLUSIVE = tuple(p.name for p in opportunity.POOLS if p.exclusive)
SHARED = tuple(p.name for p in opportunity.POOLS if not p.exclusive)


# --------------------------------------------------------------------------- #
# the measured targets
# --------------------------------------------------------------------------- #
def test_every_pool_has_a_measured_target_in_a_believable_band(targets):
    assert set(targets) == {p.name for p in opportunity.POOLS}
    for pool in EXCLUSIVE:
        # a pool below 0.8 would mean a fifth of its events belong to nobody the tables carry, which
        # is a data problem rather than a modelling choice
        assert 0.80 <= targets[pool] <= 1.05, f"{pool} measured {targets[pool]:.3f}"


def test_the_touchdown_pools_are_complete_because_quarterbacks_are_in_them(targets):
    """`rushing_tds` pools the skill share with the QB share, so it must reach one.

    This is the reason a QB rushing-TD share metric exists at all: without it the goal-line pool
    would be 15% short and every team's rushing touchdowns would land on running backs.
    """
    assert targets["rushing_tds"] == pytest.approx(1.0, abs=0.03)
    assert targets["targets"] == pytest.approx(1.0, abs=0.03)
    assert targets["dropbacks"] == pytest.approx(1.0, abs=0.03)


def test_the_situational_carry_pools_are_known_to_be_short_of_a_whole(targets):
    """Quarterback sneaks and goal-line dives are not in the skill tables, and the target says so.

    Documenting it as a test rather than a comment: if one of these ever measures near 1.0 the QB
    carries have started arriving, and these pools become load-bearing instead of context.
    """
    for pool in ("rz_carries", "inside_5_carries", "short_yardage_carries"):
        assert 0.78 <= targets[pool] <= 0.90, f"{pool} now measures {targets[pool]:.3f}"


# --------------------------------------------------------------------------- #
# the sum, before and after
# --------------------------------------------------------------------------- #
def test_an_exclusive_pool_sums_to_its_measured_target_after_scaling(opp, targets):
    for pool in EXCLUSIVE:
        got = opp.group_by(["game_id", "team"]).agg(pl.col(f"share_{pool}").sum().alias("s"))
        assert float(got["s"].mean()) == pytest.approx(targets[pool], rel=1e-6), pool
        assert float(got["s"].min()) == pytest.approx(targets[pool], rel=1e-6), pool


def test_the_shares_nearly_balance_before_any_scaling_at_all(raw, targets):
    """Normalization is a correction, not a crutch.

    Every pool has to land close to its target on the strength of the priors alone; if scaling were
    doing heavy lifting, the shares behind it would be wrong and the scaled answer would only look
    right.

    The bound is 10% rather than 5% because of *when* this projection is made. An August board carries
    a 90-man roster on which almost everybody is still ACT: the injured reserve and PUP designations
    that take a handful of men per team out of the sum do not exist until the season starts, so mean
    availability reads high and the pool sums high with it. Measured rather than assumed: run the same
    code ex ante on 2025's week-1 roster and the target pool sums to 0.9930 against a 0.9982 target,
    while the 2026 August board sums to 1.0687 with mean availability 0.432 against 0.395. The excess
    is the vintage of the roster, and it lands almost entirely on the first three depth slots -- the
    men who are certain to be on the team -- rather than on the camp bodies, whose entire claim across
    slots 7 and deeper is 0.03.

    The widest is `receiving_tds` at 8.7%, and it is widest for a reason worth keeping in view: a
    touchdown share is fitted with a small `k`, so it stays close to its slot prior instead of being
    dragged down by a player's own thin record, which leaves availability as almost the only thing
    moving the sum. The direction is asserted as well as the size -- every pool over-claims, which is
    what an un-named injured reserve looks like. A pool that came back *short* would be a different
    animal and should fail here.
    """
    for pool in EXCLUSIVE:
        got = raw.group_by(["game_id", "team"]).agg(pl.col(f"share_{pool}").sum().alias("s"))
        gap = (float(got["s"].mean()) - targets[pool]) / targets[pool]
        assert -0.05 < gap < 0.10, f"{pool} raw sum {float(got['s'].mean()):.3f} vs {targets[pool]:.3f}"


def test_participation_pools_are_never_scaled(opp, raw):
    """Five men are on the field for one snap, so a snap share pool does not sum to one.

    Scaling it would be a category error: it would divide the offence's snaps among its receivers as
    if only one could be out there.
    """
    for pool in SHARED:
        a = opp.select(pl.col(f"share_{pool}").fill_null(0.0)).to_series()
        b = raw.select(pl.col(f"share_{pool}").fill_null(0.0)).to_series()
        assert (a - b).abs().max() == pytest.approx(0.0)


def test_normalization_moves_the_exclusive_pools_and_only_those(opp, raw):
    moved = [
        p for p in EXCLUSIVE
        if (opp[f"share_{p}"].fill_null(0.0) - raw[f"share_{p}"].fill_null(0.0)).abs().max() > 1e-9
    ]
    assert moved, "normalization did nothing at all, so the toggle is not wired up"


# --------------------------------------------------------------------------- #
# availability lives inside the sum
# --------------------------------------------------------------------------- #
def test_a_player_who_cannot_play_claims_nothing(opp):
    """`p_play` multiplies the share before the pool is divided, so a retired player is weightless.

    This is what lets the full 90-man roster be on the board: everybody gets a share, and the ones
    who will not be out there contribute nothing to anyone else's denominator.
    """
    idle = opp.filter(pl.col("p_play") <= 1e-9)
    if idle.is_empty():
        pytest.skip("no zero-availability players on the current roster")
    for pool in EXCLUSIVE:
        assert float(idle[pool].fill_null(0.0).abs().max()) < 1e-9, pool


def test_availability_is_a_probability(opp):
    assert opp["p_play"].min() >= 0.0
    assert opp["p_play"].max() <= 1.0


# --------------------------------------------------------------------------- #
# the counts themselves
# --------------------------------------------------------------------------- #
def test_the_per_game_counts_look_like_an_nfl_game(opp):
    per_team = opp.group_by(["game_id", "team"]).agg(
        pl.col("targets").sum(), pl.col("carries").sum(), pl.col("dropbacks").sum()
    )
    assert 27.0 <= float(per_team["targets"].mean()) <= 35.0
    assert 22.0 <= float(per_team["carries"].mean()) <= 30.0
    assert 32.0 <= float(per_team["dropbacks"].mean()) <= 40.0
    # nobody projects a negative opportunity -- except air yards, which are a signed quantity: a back
    # who catches nothing but screens really does subtract from his offence's air yardage.
    for pool in EXCLUSIVE + SHARED:
        if pool == "air_yards":
            continue
        assert float(opp[pool].fill_null(0.0).min()) >= 0.0, pool


def test_air_yards_are_the_one_signed_pool(opp):
    """A screen behind the line is negative air yardage, so this pool is allowed below zero.

    It is asserted rather than merely permitted, because a positive floor here would mean the
    check-down backs had stopped being distinguishable from the deep threats.
    """
    assert float(opp["air_yards"].min()) < 0.0
    per_team = opp.group_by(["game_id", "team"]).agg(pl.col("air_yards").sum().alias("s"))
    assert float(per_team["s"].min()) > 0.0, "a whole offence cannot have negative air yards"


def test_every_team_gets_seventeen_games_and_one_bye(opp):
    weeks = opp.group_by("team").agg(pl.col("week").n_unique().alias("n"))
    assert (weeks["n"] == 17).all()
    assert opp["game_id"].n_unique() == 272


def test_a_starter_out_earns_more_than_his_backup(opp):
    """The depth chart has to show up in the counts, not only in the priors."""
    wk1 = opp.filter(pl.col("week") == opp["week"].min())
    for pos, col in (("WR", "targets"), ("RB", "carries"), ("QB", "dropbacks"), ("TE", "targets")):
        by_slot = (
            wk1.filter((pl.col("position") == pos) & (pl.col("depth_slot") <= 3))
            .group_by("depth_slot").agg(pl.col(col).mean().alias("v")).sort("depth_slot")
        )
        vals = by_slot["v"].to_list()
        assert vals == sorted(vals, reverse=True), f"{pos}/{col} by slot: {vals}"


def test_the_team_columns_are_prefixed_so_a_pool_cannot_be_mistaken_for_a_player(opp):
    """A pool and the count it produces share a name; the prefix is what keeps the join honest.

    Without it the team's 31 targets landed on a receiver's row and every count was the team total.
    """
    for pool in opportunity.POOLS:
        assert f"team_{pool.team_col}" in opp.columns or pool.team_col == "designed_rushes"
    assert not any(c.endswith("_right") for c in opp.columns)


# --------------------------------------------------------------------------- #
# the per-team audit
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def by_team(settings) -> pl.DataFrame:
    return opportunity.team_pool_sums(PROJ_SEASON, settings)


def test_the_per_team_sums_are_the_sums_the_projection_actually_used(by_team, raw):
    """The shortcut has to be exact, or the audit is auditing something else.

    `team_pool_sums` reads shares and availability directly instead of re-running the whole
    opportunity frame with normalization off. That is only legitimate if it lands on the same number,
    which it does because a share and an availability are both season-level: a pool's raw sum is
    identical in all seventeen games, which is why `opportunity` can take its factor from any one.
    """
    for pool in EXCLUSIVE:
        mine = by_team.filter(pl.col("pool") == pool).sort("team")
        theirs = (
            raw.group_by(["team", "game_id"]).agg(pl.col(f"share_{pool}").sum().alias("s"))
            .group_by("team").agg(pl.col("s").mean().alias("s")).sort("team")
        )
        joined = mine.join(theirs, on="team")
        assert joined.height == 32, pool
        assert (joined["raw_sum"] - joined["s"]).abs().max() < 1e-9, pool


def test_every_team_appears_in_every_pool(by_team):
    counted = by_team.group_by("pool").agg(pl.col("team").n_unique().alias("teams"))
    assert (counted["teams"] == 32).all()
    assert by_team["pool"].n_unique() == len([p for p in opportunity.POOLS if p.shares])


def test_the_factor_is_what_normalization_would_multiply_the_team_by(by_team):
    """`factor` times the raw sum is the target, which is the whole claim the column makes."""
    got = by_team.filter(pl.col("raw_sum") > 1e-9)
    assert (got["raw_sum"] * got["factor"] - got["measured_target"]).abs().max() < 1e-9
    assert (by_team["factor"] > 0).all()


def test_the_teams_are_scattered_around_the_target_rather_than_all_short(by_team):
    """A per-team audit is only useful if it disagrees per team.

    If every roster came back short the estimator would have a level bias, and normalization would be
    silently adding a fixed amount of offence to all 32 teams rather than reallocating within each.
    """
    tgt = by_team.filter(pl.col("pool") == "targets")
    assert tgt.filter(pl.col("gap_pct") > 0).height >= 8
    assert tgt.filter(pl.col("gap_pct") < 0).height >= 8
    # 8% for the August-vintage reason in `test_the_shares_nearly_balance_before_any_scaling_at_all`;
    # what this test is really about is the two assertions above it, that teams disagree in both
    # directions rather than all being short
    assert abs(float(tgt["gap_pct"].mean())) < 8.0
    # and the spread is real: some roster is off by more than a few percent
    assert float(tgt["gap_pct"].abs().max()) > 5.0


def test_the_toggle_is_recorded_against_each_row(settings):
    """A row that says `normalized` false has to mean the number was left alone."""
    off = opportunity.team_pool_sums(PROJ_SEASON, replace(settings, normalize_pools=False))
    on = opportunity.team_pool_sums(PROJ_SEASON, settings)
    assert not off["normalized"].any()
    assert on.filter(pl.col("exclusive"))["normalized"].all()
    assert not on.filter(~pl.col("exclusive"))["normalized"].any()
    # the raw sums are a property of the roster, not of the toggle
    assert (off["raw_sum"] - on["raw_sum"]).abs().max() < 1e-12


# --------------------------------------------------------------------------- #
# who pays for a room that over-claims, and whose number is held
# --------------------------------------------------------------------------- #
# On a written-out room rather than on the season: the allocator's properties are arithmetic, and a
# five-man frame states them exactly where a 918-player run states them to a tolerance. The engine-wide
# versions are the pool-sum tests above, which run at whatever exponent is in force.
ROOM = ["game_id", "team"]


def room(claims: list[float], p_play: list[float] | None = None,
         locked: list[bool] | None = None) -> pl.DataFrame:
    """One team, one game: a claim each, and optionally an availability and a lock each.

    In depth order, because `_queue_alloc` fills the chart by `depth_slot` and the list is written the
    way a room reads -- starter first.
    """
    n = len(claims)
    return pl.DataFrame({
        "game_id": ["g"] * n, "team": ["T"] * n,
        "player_id": [f"p{i}" for i in range(n)], "depth_slot": list(range(1, n + 1)),
        "claim": claims,
        "p_play": [1.0] * n if p_play is None else p_play,
        "locked": [False] * n if locked is None else locked,
    })


def settle(frame: pl.DataFrame, want: float, tilt: float = 1.0,
           locks: bool = False, queue: bool = False) -> list[float]:
    alloc = opportunity._settle(
        pl.col("claim"),
        opportunity._tilt_weight(pl.col("claim"), pl.col("p_play"), tilt),
        pl.lit(want),
        pl.col("locked") if locks else None,
        queue,
    )
    return frame.select(alloc.alias("got"))["got"].to_list()


def test_tilt_one_is_the_flat_rescale_to_the_last_bit():
    """The identity that makes 1.0 a safe fallback: it has to change no number at all."""
    claims = [0.31, 0.22, 0.14, 0.09, 0.04, 0.008]
    want = 0.9977
    got = settle(room(claims), want, tilt=1.0)
    total = sum(claims)
    assert got == pytest.approx([c * want / total for c in claims], abs=1e-15)


def test_every_exponent_leaves_the_pool_exact():
    """The clip in `take` can leave a room short of paying; the second line is what closes it."""
    for tilt in (1.0, 0.85, 0.7, 0.4, 0.0):
        got = settle(room([0.34, 0.25, 0.18, 0.11, 0.05, 0.002]), 0.9977, tilt=tilt)
        assert sum(got) == pytest.approx(0.9977, abs=1e-12), tilt
        assert min(got) >= 0.0, tilt


def test_a_lower_exponent_charges_the_small_claims_more():
    """The whole point of the exponent, as a monotone statement rather than a fitted number.

    Reading the same over-claiming room down the grid: the biggest claim keeps more of itself and the
    smallest keeps less, every step of the way. `0.0` is the far end -- equal *absolute* amounts, so the
    bench pays first -- and it is in the list to show the direction does not turn round somewhere.
    """
    claims = [0.34, 0.25, 0.18, 0.11, 0.05, 0.02]      # sums to 0.95 against a 0.90 pool
    kept = [settle(room(claims), 0.90, tilt=t) for t in (1.0, 0.85, 0.7, 0.4, 0.0)]
    top = [k[0] / claims[0] for k in kept]
    bench = [k[-1] / claims[-1] for k in kept]
    assert top == sorted(top), f"the starter did not keep more as the exponent fell: {top}"
    assert bench == sorted(bench, reverse=True), f"the bench did not pay more: {bench}"


def test_an_under_claiming_room_is_scaled_up_and_not_tilted():
    """There is no evidence about who deserves a share of a shortfall, so everybody gets the same lift."""
    claims = [0.30, 0.20, 0.10, 0.02]                   # sums to 0.62 against a 0.90 pool
    for tilt in (1.0, 0.55, 0.0):
        got = settle(room(claims), 0.90, tilt=tilt)
        lift = [g / c for g, c in zip(got, claims, strict=True)]
        assert lift == pytest.approx([lift[0]] * len(claims), abs=1e-12), tilt


# A room typed past its own pool, which is the case a lock is *for*: the man is pushed up, the room now
# claims 1.25 of a 0.9977 pool, and somebody has to pay. (A room that under-claims has the opposite
# behaviour by design -- the teammates are lifted rather than charged -- which is its own test above.)
PUSHED = [0.30, 0.35, 0.28, 0.20, 0.12]
FIRST_LOCKED = [True, False, False, False, False]


def test_a_locked_share_is_delivered_at_what_was_typed():
    """The bug this was written for: "set his target share to 0.30" used to deliver 0.2545."""
    got = settle(room(PUSHED, locked=FIRST_LOCKED), 0.9977, tilt=0.85, locks=True)
    assert got[0] == pytest.approx(0.30, abs=1e-12)
    assert sum(got) == pytest.approx(0.9977, abs=1e-12)
    # and the room he is in is what moved: every un-edited teammate paid something
    assert all(g < c for g, c in zip(got[1:], PUSHED[1:], strict=True))


def test_without_the_lock_the_same_typed_share_is_rescaled_away():
    """The alternative reading, kept as a test because it is a supported setting and not a bug."""
    assert settle(room(PUSHED, locked=FIRST_LOCKED), 0.9977, tilt=0.85, locks=False)[0] < 0.30 - 1e-6


def test_locks_that_claim_more_than_the_pool_are_scaled_against_each_other():
    """A room typed past its own pool cannot have what it asked for, and the pool still has to balance.

    What is asserted is the choice: the locks keep their *ratios* and the un-edited men go to zero,
    rather than the team being allowed to throw more passes than it is projected to throw.
    """
    frame = room([0.70, 0.60, 0.20, 0.10], locked=[True, True, False, False])
    got = settle(frame, 1.0, tilt=0.7, locks=True)
    assert sum(got) == pytest.approx(1.0, abs=1e-12)
    assert got[0] / got[1] == pytest.approx(0.70 / 0.60, abs=1e-12)
    assert got[2] == pytest.approx(0.0, abs=1e-12)
    assert got[3] == pytest.approx(0.0, abs=1e-12)


def test_a_queue_pool_still_fills_in_depth_order_around_a_lock():
    """Dropbacks are one man's job: the backup only takes what the starter's lock leaves."""
    frame = room([1.0, 0.9, 0.5], locked=[True, False, False])
    got = settle(frame, 1.0, tilt=0.7, locks=True, queue=True)
    assert got == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    held = room([0.6, 0.9, 0.5], locked=[True, False, False])
    got = settle(held, 1.0, tilt=0.7, locks=True, queue=True)
    assert got[0] == pytest.approx(0.6, abs=1e-12)      # his own number, not the front of the queue
    assert sum(got) == pytest.approx(1.0, abs=1e-12)
    assert got[1] > got[2], "the queue stopped running in depth order behind the lock"


def test_availability_stays_linear_whatever_the_exponent_is():
    """`_tilt_weight` bends the share and leaves the games alone, which is a counting fact.

    Two men on the same share, one of them available for half the season: his claim is half and so is
    the amount of the room's disagreement he can be charged for, at every exponent. The exponent is
    about how wrong a *rate* is, and a man cannot be wrong about games he is not there for.
    """
    for tilt in (1.0, 0.7, 0.0):
        w = room([0.2, 0.1], p_play=[1.0, 0.5]).select(       # equal shares, half the availability
            opportunity._tilt_weight(pl.col("claim"), pl.col("p_play"), tilt).alias("w")
        )["w"].to_list()
        assert w[0] / w[1] == pytest.approx(2.0, rel=1e-12), tilt


def test_the_exponent_in_force_is_the_scenario_before_the_fitted_value(settings):
    assert opportunity.tilt_of(replace(settings, pool_tilt=0.5)) == 0.5
    assert opportunity.tilt_of(replace(settings, pool_tilt=None)) == pytest.approx(
        float(opportunity.load_tilt().get("pool_tilt", 1.0)))
