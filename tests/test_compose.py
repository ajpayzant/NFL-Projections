"""Composition: the identities that make a projection mean what it says.

Nothing here is fitted, so nothing here is allowed to be approximate. If a receiver's yards are not
his targets times his yards per target, the number on the board is not the number the engine believes
and no amount of accuracy work upstream can fix that. These tests are the arithmetic's contract.

The three that matter most are the ones the workbook could not state at all: a dropback ends in
exactly one of three ways, a quarterback's rushes split into designed runs and scrambles that add back
to his total, and a season is the sum of the games on the real schedule rather than a per-game number
multiplied by seventeen.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from src.config import LAST_COMPLETE_SEASON, PROJ_SEASON, Settings
from src.data import history
from src.model import compose, efficiency, opportunity


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def opp(settings) -> pl.DataFrame:
    return opportunity.opportunity(PROJ_SEASON, settings)


@pytest.fixture(scope="module")
def player_rates(settings) -> pl.DataFrame:
    return efficiency.rates(PROJ_SEASON, settings)


@pytest.fixture(scope="module")
def adj(opp, player_rates, settings) -> pl.DataFrame:
    """The frame composition reads, kept so identities can be checked against their own inputs."""
    return efficiency.adjusted(opp, player_rates, settings)


@pytest.fixture(scope="module")
def wk(settings, opp, player_rates) -> pl.DataFrame:
    return compose.weekly(PROJ_SEASON, settings, opp, player_rates)


@pytest.fixture(scope="module")
def yr(wk, settings) -> pl.DataFrame:
    return compose.seasonal(wk, PROJ_SEASON, settings)


def _joined(wk: pl.DataFrame, adj: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return wk.join(adj.select("game_id", "player_id", *cols), on=["game_id", "player_id"], how="inner")


# --------------------------------------------------------------------------- #
# count x rate, exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stat,count,rate", [
    ("receptions", "targets", "used_catch_rate"),
    ("receiving_yards", "targets", "used_yards_per_target"),
    ("completions", "attempts", "used_completion_pct"),
    ("passing_yards", "attempts", "used_yards_per_attempt"),
    ("interceptions", "attempts", "used_int_rate"),
    ("passing_air_yards", "attempts", "used_air_yards_per_attempt"),
])
def test_a_stat_is_its_count_times_its_rate(wk, adj, stat, count, rate):
    j = _joined(wk, adj, [rate])
    d = (pl.col(stat) - pl.col(count) * pl.col(rate).fill_null(0.0)).abs()
    assert float(j.select(d.max()).item()) < 1e-9, stat


def test_a_skill_players_rushing_yards_are_his_carries_times_his_yards_per_carry(wk, adj):
    j = _joined(wk, adj, ["used_yards_per_carry"]).filter(pl.col("position") != "QB")
    d = (pl.col("rushing_yards") - pl.col("carries") * pl.col("used_yards_per_carry").fill_null(0.0)).abs()
    assert float(j.select(d.max()).item()) < 1e-9


def test_touchdowns_come_from_the_team_pool_and_not_from_a_rate(wk, opp):
    """A red-zone share says who is out there; the touchdown itself is the team's, divided up.

    Projecting TDs from a per-target rate is how a workbook ends up with a league that scores 1,300
    passing touchdowns. Here the pool is the team's projection and the players cannot outrun it.
    """
    j = wk.join(
        opp.select("game_id", "player_id", "receiving_tds", "rushing_tds", "passing_tds"),
        on=["game_id", "player_id"], how="inner", suffix="_pool",
    )
    for col in ("receiving_tds", "rushing_tds", "passing_tds"):
        d = (pl.col(col) - pl.col(f"{col}_pool").fill_null(0.0)).abs()
        assert float(j.select(d.max()).item()) < 1e-9, col


# --------------------------------------------------------------------------- #
# the dropback identity
# --------------------------------------------------------------------------- #
def test_a_dropback_ends_in_an_attempt_a_sack_or_a_scramble(wk):
    """The three fates are fitted separately and rescaled to what they historically summed to.

    Without the rescale a quarterback whose sack rate and scramble rate both came in high lost pass
    attempts he was never going to lose, and his team's passing volume quietly fell short.
    """
    target = compose.measure_dropback_split()
    qb = wk.filter((pl.col("position") == "QB") & (pl.col("dropbacks") > 1e-6))
    assert qb.height > 0
    got = qb.select(
        ((pl.col("attempts") + pl.col("sacks") + pl.col("scrambles")) / pl.col("dropbacks")).alias("r")
    )
    assert float(got["r"].max()) == pytest.approx(target, abs=1e-9)
    assert float(got["r"].min()) == pytest.approx(target, abs=1e-9)


def test_the_fitted_attempt_and_sack_rates_still_decide_how_the_rest_is_divided(wk, adj):
    """Closing the identity must not throw away the rates; it only decides what they divide.

    The residual lands on attempts and sacks together, in exactly the proportion the two fitted rates
    give -- so a quarterback behind a bad line still takes more sacks than one behind a good one.
    """
    j = _joined(wk, adj, ["used_attempt_rate", "used_sack_rate"]).filter(
        (pl.col("position") == "QB") & (pl.col("dropbacks") > 1e-6)
    )
    want = pl.col("used_attempt_rate") / (pl.col("used_attempt_rate") + pl.col("used_sack_rate"))
    got = pl.col("attempts") / (pl.col("attempts") + pl.col("sacks"))
    assert float(j.select((got - want).abs().max()).item()) < 1e-9
    # and the spread across quarterbacks is real, not a rounding artefact
    by_qb = j.group_by("player").agg(
        (pl.col("sacks").sum() / pl.col("dropbacks").sum()).alias("sr")
    )
    assert float(by_qb["sr"].max()) - float(by_qb["sr"].min()) > 0.02


def test_the_measured_split_is_just_above_one_and_measured_rather_than_assumed():
    """1.003, because a handful of dropbacks end in an aborted snap or a fumbled exchange."""
    target = compose.measure_dropback_split()
    assert 1.0 <= target <= 1.01


# --------------------------------------------------------------------------- #
# the quarterback's legs
# --------------------------------------------------------------------------- #
def test_a_quarterbacks_designed_runs_and_scrambles_add_back_to_his_carries(wk):
    """The pool owns the total; the ratio of the two projections only decides the split.

    That ordering matters: the team's ground game stays balanced no matter how the split lands, which
    would not be true if designed runs and scrambles were each projected and then added.
    """
    qb = wk.filter(pl.col("position") == "QB")
    d = (pl.col("designed_rushes") + pl.col("scrambles") - pl.col("carries")).abs()
    assert float(qb.select(d.max()).item()) < 1e-9
    assert float(qb["designed_rushes"].min()) >= 0.0
    assert float(qb["scrambles"].min()) >= 0.0


def test_a_quarterbacks_rushing_yards_use_the_two_rates_and_not_a_blended_one(wk, adj):
    j = _joined(wk, adj, ["used_designed_rush_ypc", "used_scramble_ypc"]).filter(
        pl.col("position") == "QB"
    )
    d = (
        pl.col("rushing_yards")
        - pl.col("designed_rushes") * pl.col("used_designed_rush_ypc").fill_null(0.0)
        - pl.col("scrambles") * pl.col("used_scramble_ypc").fill_null(0.0)
    ).abs()
    assert float(j.select(d.max()).item()) < 1e-9
    parts = (pl.col("designed_rush_yards") + pl.col("scramble_yards") - pl.col("rushing_yards")).abs()
    assert float(j.select(parts.max()).item()) < 1e-9


def test_a_scrambling_quarterback_gains_more_per_carry_than_a_pocket_one(wk):
    """The split has to be visible in the output, not only in the arithmetic."""
    qb = (
        wk.filter(pl.col("position") == "QB")
        .group_by("player").agg(
            pl.col("carries").sum().alias("c"), pl.col("scrambles").sum().alias("s"),
            pl.col("rushing_yards").sum().alias("y"),
        )
        .filter(pl.col("c") > 20)
        .with_columns((pl.col("s") / pl.col("c")).alias("scr_frac"), (pl.col("y") / pl.col("c")).alias("ypc"))
    )
    assert qb.height > 20
    hi = qb.filter(pl.col("scr_frac") > pl.col("scr_frac").median())
    lo = qb.filter(pl.col("scr_frac") <= pl.col("scr_frac").median())
    assert float(hi["ypc"].mean()) > float(lo["ypc"].mean())


# --------------------------------------------------------------------------- #
# only skill players fumble on a touch, only quarterbacks on a dropback
# --------------------------------------------------------------------------- #
def test_the_fumble_denominator_follows_the_position(wk, adj):
    j = _joined(wk, adj, ["used_fumble_rate", "used_qb_fumble_rate"])
    skill = j.filter(pl.col("position") != "QB")
    d = (
        pl.col("fumbles_lost")
        - (pl.col("carries") + pl.col("receptions")) * pl.col("used_fumble_rate").fill_null(0.0)
    ).abs()
    assert float(skill.select(d.max()).item()) < 1e-9
    qb = j.filter(pl.col("position") == "QB")
    dq = (pl.col("fumbles_lost") - pl.col("dropbacks") * pl.col("used_qb_fumble_rate").fill_null(0.0)).abs()
    assert float(qb.select(dq.max()).item()) < 1e-9


# --------------------------------------------------------------------------- #
# fantasy points
# --------------------------------------------------------------------------- #
def test_fantasy_points_recompute_from_the_components(wk, settings):
    """Scoring is a setting, never a stored column, so the board must survive changing it."""
    again = wk.select(history.fantasy_points(wk.columns, settings.scoring))
    d = (wk["fantasy_points"] - again["fantasy_points"]).abs().max()
    assert float(d) == pytest.approx(0.0)


def test_a_fumble_costs_points_exactly_once(wk, settings):
    """History carries rushing and receiving fumbles apart; a projection carries one column.

    Scoring both would double the penalty, which is why `fantasy_points` only counts the combined
    column when the two components are absent.
    """
    assert "fumbles_lost" in wk.columns
    assert "rushing_fumbles_lost" not in wk.columns and "receiving_fumbles_lost" not in wk.columns
    zeroed = wk.with_columns(pl.lit(0.0).alias("fumbles_lost"))
    lost = zeroed.select(history.fantasy_points(zeroed.columns, settings.scoring))
    delta = float(lost["fantasy_points"].sum() - wk["fantasy_points"].sum())
    expected = -settings.scoring.fumble_lost * float(wk["fumbles_lost"].sum())
    assert delta == pytest.approx(expected, rel=1e-9)


def test_a_scoring_change_moves_the_board(wk, settings):
    half = replace(settings, scoring=replace(settings.scoring, reception=0.5))
    other = wk.select(history.fantasy_points(wk.columns, half.scoring))
    assert float(other["fantasy_points"].sum()) < float(wk["fantasy_points"].sum())


# --------------------------------------------------------------------------- #
# the season
# --------------------------------------------------------------------------- #
def test_a_season_is_the_sum_of_the_games_on_the_schedule(wk, yr):
    for col in ("targets", "carries", "receiving_yards", "rushing_yards", "passing_yards",
                "fantasy_points"):
        assert float(yr[col].sum()) == pytest.approx(float(wk[col].sum()), rel=1e-9), col
    assert yr.height == wk["player_id"].n_unique()
    assert (yr["weeks"] == 17).all()


def test_games_are_summed_availability_and_never_more_than_seventeen(yr):
    """A player expected to miss three weeks has fourteen games, and his per-game line is over those."""
    assert float(yr["games"].max()) <= 17.0 + 1e-9
    assert float(yr["games"].min()) >= 0.0
    d = (yr["fantasy_points"] / yr["games"].clip(1e-9) - yr["points_per_game"]).abs().max()
    assert float(d) < 1e-6


def test_no_stat_is_negative_except_the_one_that_is_allowed_to_be(yr):
    for col in compose.STAT_COLUMNS:
        if col in ("receiving_air_yards", "passing_air_yards"):
            continue  # air yards are signed: a screen behind the line subtracts
        assert float(yr[col].min()) >= -1e-9, col


def test_the_board_is_ranked_and_the_ranks_agree_with_the_points(yr):
    assert yr["overall_rank"].min() == 1
    assert (yr["overall_rank"].diff().drop_nulls() >= 0).all()
    for pos in ("QB", "RB", "WR", "TE"):
        p = yr.filter(pl.col("position") == pos).sort("position_rank")
        assert p["position_rank"][0] == 1
        assert (p["fantasy_points"].diff().drop_nulls() <= 1e-9).all(), pos


# --------------------------------------------------------------------------- #
# the players add up to the team
# --------------------------------------------------------------------------- #
def test_counts_drawn_from_a_pool_reconcile_with_the_team_that_owns_them(wk, settings):
    """The invariant the workbook broke: a team's projected targets and its receivers' must agree.

    The tolerance is the pool's measured target, not 1.0 -- carries come back 3% short because 3% of
    a team's rush attempts belong to nobody in the skill or passer tables. Forcing that to zero would
    invent the carries rather than report them.
    """
    check = compose.team_check(wk, PROJ_SEASON, settings)
    pooled = check.filter(pl.col("from_pool"))
    assert pooled.height >= 5
    for row in pooled.iter_rows(named=True):
        assert abs(row["gap_pct"]) < 4.0, row


def test_yardage_is_allowed_to_disagree_with_the_team_and_only_a_little(wk, settings):
    """Yards are a count times a rate, so they are not forced to match -- but a big gap is a bug.

    A roster of above-average receivers genuinely should project past its offence's recent yardage;
    scaling that away would hide a disagreement worth seeing.
    """
    check = compose.team_check(wk, PROJ_SEASON, settings)
    derived = check.filter(~pl.col("from_pool"))
    assert derived.height >= 3
    for row in derived.iter_rows(named=True):
        assert abs(row["gap_pct"]) < 6.0, row


def test_turning_normalization_off_breaks_the_reconciliation_and_that_is_the_point(settings):
    """The toggle has to do something visible, or it is decoration.

    Unscaled, the pools land within a few percent on the strength of the priors alone -- but they land
    there by luck of the roster rather than by construction, and the audit is what shows the difference.
    """
    off = replace(settings, normalize_pools=False)
    a = compose.team_check(None, PROJ_SEASON, settings).filter(pl.col("stat") == "targets")
    b = compose.team_check(None, PROJ_SEASON, off).filter(pl.col("stat") == "targets")
    assert float(a["worst_abs_gap"][0]) < float(b["worst_abs_gap"][0])


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def brd(yr, settings) -> pl.DataFrame:
    prev = history.player_seasons((LAST_COMPLETE_SEASON,), settings.scoring)
    return compose.board(yr, settings, prev)


def test_the_board_keeps_every_projected_player_and_adds_nobody(brd, yr):
    """A ranking board is a view of the projection, not a second population.

    The prior-season join is the risk: last season had players this one does not, and a join that let
    one of them through would put a man on the board with no projection behind him.
    """
    assert brd.height == yr.height
    assert set(brd["player_id"].to_list()) == set(yr["player_id"].to_list())
    assert brd["player_id"].n_unique() == brd.height


def test_the_drop_to_the_next_man_is_read_down_the_position_and_not_the_frame(brd):
    """`drop_next` is a window over the row below, so the frame's order is part of the arithmetic.

    Sorting the values without sorting the rows maps a sorted gap onto an unsorted player, which looks
    plausible and is wrong everywhere. This checks the gap against the position's own next man.
    """
    for pos in ("QB", "RB", "WR", "TE"):
        p = brd.filter(pl.col("position") == pos).sort("fantasy_points", descending=True)
        expected = p["fantasy_points"] - p["fantasy_points"].shift(-1)
        assert (p["drop_next"] - expected).abs().max() < 1e-9, pos
        # the last man at a position has nobody below him, and that is a null rather than a zero
        assert p["drop_next"][-1] is None
        assert (p["drop_next"].drop_nulls() >= -1e-9).all(), pos


def test_a_tier_is_a_block_of_the_position_in_projected_order(brd, settings):
    size = settings.tier_size
    assert brd["tier"].min() == 1
    for pos in ("QB", "RB", "WR", "TE"):
        p = brd.filter(pl.col("position") == pos)
        assert ((p["position_rank"] - 1) // size + 1 == p["tier"]).all(), pos
        assert p.filter(pl.col("tier") == 1).height <= size


def test_startable_is_the_positions_starter_count_and_nothing_else(brd, settings):
    for pos, n in settings.starters.items():
        p = brd.filter(pl.col("position") == pos)
        if p.is_empty():
            continue
        assert p.filter(pl.col("startable")).height == min(n, p.height), pos


def test_points_above_the_average_starter_sum_to_nothing_among_the_starters(brd):
    """`vs_starter` is a difference from a mean, so the starters' own deviations must cancel.

    The test that catches the mistake worth catching: computing the mean over everyone at the position
    rather than over the startable ones, which is a different and much less useful number.
    """
    for pos in ("QB", "RB", "WR", "TE"):
        p = brd.filter((pl.col("position") == pos) & pl.col("startable"))
        assert abs(float(p["vs_starter"].sum())) < 1e-6, pos
        assert brd.filter(pl.col("position") == pos)["vs_starter"].null_count() == 0


def test_a_move_needs_a_previous_team_to_have_moved_from(brd):
    """A player with no prior season has not changed teams; he has arrived."""
    fresh = brd.filter(pl.col("last_team").is_null())
    assert fresh.height > 0
    assert not fresh["changed_team"].any()
    assert fresh["delta_points"].null_count() == fresh.height
    moved = brd.filter(pl.col("changed_team"))
    assert (moved["last_team"] != moved["team"]).all()


def test_a_board_without_a_previous_season_still_ranks(yr, settings):
    """The comparison columns are optional; the ranking is not."""
    bare = compose.board(yr, settings)
    assert bare.height == yr.height
    assert "delta_points" not in bare.columns
    assert bare["vs_starter"].null_count() == 0
