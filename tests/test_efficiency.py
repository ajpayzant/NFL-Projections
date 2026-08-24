"""The rate layer: what a player does with an opportunity, and what the game does to that.

Two things are being defended here. First that a rate is a plausible football number for the position
holding it -- the workbook's failure mode was a backup with one long run reading as a 6.0 YPC back.
Second that the per-game adjustment stays a small, bounded, *documented* nudge: it is the team's
environment divided by the team's own average, so it must average out to roughly one over a season and
must never be the reason a projection looks the way it does.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from src.config import PROJ_SEASON, Settings
from src.model import efficiency, opportunity


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def player_rates(settings) -> pl.DataFrame:
    return efficiency.rates(PROJ_SEASON, settings)


@pytest.fixture(scope="module")
def opp(settings) -> pl.DataFrame:
    return opportunity.opportunity(PROJ_SEASON, settings)


@pytest.fixture(scope="module")
def adj(opp, player_rates, settings) -> pl.DataFrame:
    return efficiency.adjusted(opp, player_rates, settings)


# --------------------------------------------------------------------------- #
# the rates themselves
# --------------------------------------------------------------------------- #
def test_every_rate_is_estimated_for_the_positions_that_have_one(player_rates):
    for name in efficiency.RATE_METRICS:
        assert name in player_rates.columns, name


BANDS = {
    ("RB", "yards_per_carry"): (3.0, 5.6),
    ("WR", "catch_rate"): (0.45, 0.80),
    ("TE", "catch_rate"): (0.50, 0.85),
    ("RB", "catch_rate"): (0.60, 0.90),
    ("WR", "yards_per_target"): (5.0, 12.0),
    ("QB", "completion_pct"): (0.50, 0.75),
    ("QB", "yards_per_attempt"): (5.0, 9.0),
    ("QB", "int_rate"): (0.0, 0.06),
    ("QB", "sack_rate"): (0.02, 0.14),
}


@pytest.mark.parametrize("key,band", list(BANDS.items()))
def test_shrunk_rates_sit_inside_a_believable_band(player_rates, opp, key, band):
    """Shrinkage is what puts them here: unshrunk, a 12-target rookie would sit far outside.

    The bands are wide on purpose. They are not accuracy claims, they are a guard against the failure
    the workbook actually had -- a small sample taken at face value.
    """
    pos, name = key
    lo, hi = band
    ids = opp.filter(pl.col("position") == pos)["player_id"].unique()
    vals = player_rates.filter(pl.col("player_id").is_in(ids.implode()))[name].drop_nulls()
    if vals.is_empty():
        pytest.skip(f"no {pos} carries a fitted {name}")
    assert lo <= float(vals.min()), f"{pos} {name} floor {float(vals.min()):.3f} < {lo}"
    assert float(vals.max()) <= hi, f"{pos} {name} ceiling {float(vals.max()):.3f} > {hi}"


def test_a_scramble_is_worth_more_than_a_designed_quarterback_run(player_rates):
    """The whole reason the two are projected apart rather than as one rush-attempt rate.

    A scramble happens in broken coverage against a pass rush that has vacated its lanes; a designed
    run is a sneak or a read-option keeper into a set front. League-wide it is 7.25 yards against 3.05.
    """
    des = player_rates["designed_rush_ypc"].drop_nulls().mean()
    scr = player_rates["scramble_ypc"].drop_nulls().mean()
    assert scr > des * 1.5, f"scramble {scr:.2f} vs designed {des:.2f}"


# --------------------------------------------------------------------------- #
# the per-game factor
# --------------------------------------------------------------------------- #
def test_a_rate_the_team_layer_has_no_opinion_about_is_left_alone(adj):
    """Silence is the honest default: inventing a per-game catch-rate factor would be a made-up number.

    Only the rate families the team model actually projects per game -- yards per attempt, yards per
    carry, success rate -- get a multiplier. Everything else stays season-flat.
    """
    for name in efficiency.RATE_METRICS:
        if name in efficiency.CONTEXT_OF:
            continue
        col = adj[f"f_{name}"]
        assert float(col.min()) == 1.0 and float(col.max()) == 1.0, name


def test_the_factor_respects_its_clip(adj):
    lo, hi = efficiency.FACTOR_CLIP
    for name in efficiency.CONTEXT_OF:
        col = adj[f"f_{name}"]
        assert lo <= float(col.min()) and float(col.max()) <= hi, name


def test_the_factor_averages_out_over_a_season(adj):
    """A team plays a schedule, not one game: the multiplier is context, not a thumb on the scale.

    If a team's season mean drifted far from one, the per-game chain and the season-flat estimate it
    is divided by would be disagreeing about the same offence.
    """
    for name in efficiency.CONTEXT_OF:
        by_team = adj.group_by("team").agg(pl.col(f"f_{name}").mean().alias("m"))
        assert 0.93 <= float(by_team["m"].min()), f"{name}: {float(by_team['m'].min()):.3f}"
        assert float(by_team["m"].max()) <= 1.07, f"{name}: {float(by_team['m'].max()):.3f}"


def test_the_factor_actually_varies_between_a_teams_own_weeks(adj):
    """A schedule has to matter. If every week were identical the whole per-game layer is dead code."""
    spread = (
        adj.group_by(["team", "week"]).agg(pl.col("f_yards_per_attempt").mean().alias("f"))
        .group_by("team").agg((pl.col("f").max() - pl.col("f").min()).alias("range"))
    )
    assert float(spread["range"].median()) > 0.02


def test_turning_the_context_off_makes_every_factor_one(opp, player_rates, settings):
    off = efficiency.adjusted(opp, player_rates, replace(settings, use_context_factors=False))
    for name in efficiency.RATE_METRICS:
        assert float(off[f"f_{name}"].abs().max()) == 1.0, name
    have = [c for c in efficiency.RATE_METRICS if c in player_rates.columns]
    for name in have:
        d = (off[f"used_{name}"].fill_null(0.0) - off[name].fill_null(0.0)).abs().max()
        assert float(d) == pytest.approx(0.0), name


def test_adjusting_neither_adds_nor_drops_a_player_game(opp, adj):
    assert adj.height == opp.height
    assert adj["player_id"].n_unique() == opp["player_id"].n_unique()


# --------------------------------------------------------------------------- #
# the finding the engine is built around
# --------------------------------------------------------------------------- #
def test_a_players_own_efficiency_history_counts_for_less_than_his_own_usage(settings):
    """Opportunity is projectable and efficiency largely is not, and the fitted weights say so.

    This is the single most important thing the engine believes. It is asserted as a test so that a
    future change which quietly starts trusting small-sample yards per carry fails loudly.
    """
    from src.model import estimate, roster

    ros = roster.roster(PROJ_SEASON)
    detail = efficiency.rate_detail(PROJ_SEASON, settings, ros)
    shares = estimate.estimate(opportunity.SHARE_METRICS, ros, PROJ_SEASON, settings)
    rate_w = float(detail["own_weight"].mean())
    share_w = float(shares["own_weight"].mean())
    assert share_w > rate_w, f"shares {share_w:.3f} vs rates {rate_w:.3f}"
    # and the specific case that broke the workbook
    ypc = detail.filter(pl.col("metric") == "yards_per_carry")
    assert float(ypc["own_weight"].mean()) < 0.20
