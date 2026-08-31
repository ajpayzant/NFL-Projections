"""Priors, depth slots and the fitted artifacts.

These run against the real lake, so they assert football invariants -- a WR1 out-targets a WR5, a
first-round rookie out-targets a seventh-rounder -- rather than fixed numbers that would break on
every data refresh.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config import PROJ_SEASON, Settings
from src.data import depth, history, lake
from src.model import priors


@pytest.fixture(scope="module")
def slot_frame() -> pl.DataFrame:
    return priors.slots(tuple(range(2016, PROJ_SEASON + 1)))


@pytest.fixture(scope="module")
def pools() -> pl.DataFrame:
    return priors.team_pools()


# --------------------------------------------------------------------------- #
# depth charts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("season,when", [(2019, "preseason"), (2022, "preseason"), (PROJ_SEASON, "latest")])
def test_depth_slots_are_contiguous_within_a_team_and_position(season, when):
    ch = depth.depth_chart(season, when)
    counts = ch.group_by(["team", "position"]).agg(
        pl.col("depth_slot").min().alias("lo"),
        pl.col("depth_slot").max().alias("hi"),
        pl.len().alias("n"),
    )
    assert (counts["lo"] == 1).all()
    assert (counts["hi"] == counts["n"]).all()


def test_the_same_chart_yields_the_same_slots_every_time():
    """Two undrafted rookies in one tier tie on both real tiebreakers.

    Without a final deterministic key the slot order flipped between runs, which moved every prior
    built on it -- the fitted availability MAE drifted in the third decimal from one fit to the next.
    """
    depth.clear_cache()
    a = depth.depth_chart(PROJ_SEASON, "latest").sort("player_id")
    depth.clear_cache()
    b = depth.depth_chart(PROJ_SEASON, "latest").sort("player_id")
    assert a.select("player_id", "depth_slot").rows() == b.select("player_id", "depth_slot").rows()
    ties = a.group_by(["team", "position", "depth_tier"]).len().filter(pl.col("len") > 1)
    assert ties.height > 0, "no tiers with more than one player means the test proves nothing"


def test_every_team_has_a_charted_starter_at_every_position():
    ch = depth.depth_chart(PROJ_SEASON)
    starters = ch.filter(pl.col("depth_slot") == 1)
    for pos in ("QB", "RB", "WR", "TE"):
        assert starters.filter(pl.col("position") == pos)["team"].n_unique() == 32


def test_the_projection_season_uses_the_newest_chart_and_past_seasons_do_not():
    latest = depth.depth_chart(PROJ_SEASON, "latest")["snapshot"][0]
    pre = depth.depth_chart(PROJ_SEASON, "preseason")["snapshot"][0]
    assert latest >= pre


# --------------------------------------------------------------------------- #
# group priors
# --------------------------------------------------------------------------- #
def test_target_share_prior_falls_as_the_depth_slot_deepens(slot_frame):
    metric = priors.BY_NAME["target_share"]
    hist = history.skill_seasons(lake.history_seasons()).join(
        slot_frame.select("season", "player_id", "position", "slot_bucket"),
        on=["season", "player_id"], how="inner", suffix="_chart",
    ).with_columns(pl.col("position_chart").alias("position")).drop("position_chart")
    gp = priors.group_prior(metric, hist, before=PROJ_SEASON)
    for pos in ("WR", "RB", "TE"):
        vals = gp.filter(pl.col("position") == pos).sort("slot_bucket")["prior"].to_list()
        assert vals == sorted(vals, reverse=True), f"{pos} priors not decreasing: {vals}"


def test_priors_sit_inside_a_plausible_range(slot_frame):
    gp = priors.load_priors()
    if gp.is_empty():
        pytest.skip("no fitted priors on disk; run python -m src.model.priors --fit")
    shares = gp.filter(pl.col("metric").str.ends_with("_share"))
    assert shares["prior"].min() >= 0.0
    assert shares["prior"].max() <= 1.0
    ypc = gp.filter((pl.col("metric") == "yards_per_carry") & (pl.col("position") == "RB"))
    assert 3.5 <= ypc["prior"].min() <= ypc["prior"].max() <= 5.0


# --------------------------------------------------------------------------- #
# panels: the leakage guard, end to end
# --------------------------------------------------------------------------- #
def test_a_panel_never_reads_its_own_target_season(slot_frame, pools):
    """Corrupting the target season must not move a single blended observation."""
    metric = priors.BY_NAME["target_share"]
    hist = history.skill_seasons(lake.history_seasons())
    clean = priors.panel(metric, 2024, hist, slot_frame, pools, Settings())
    poisoned_hist = hist.with_columns(
        pl.when(pl.col("season") == 2024).then(pl.lit(9999.0)).otherwise(pl.col("targets")).alias("targets")
    )
    poisoned = priors.panel(metric, 2024, poisoned_hist, slot_frame, pools, Settings())
    joined = clean.join(poisoned, on="player_id", how="inner", suffix="_p")
    assert joined.height > 100
    assert (joined["obs"] - joined["obs_p"]).abs().max() in (None, 0.0)
    assert (joined["prior"] - joined["prior_p"]).abs().max() == pytest.approx(0.0)


def test_panel_covers_the_whole_charted_roster_not_only_players_with_history(slot_frame, pools):
    metric = priors.BY_NAME["target_share"]
    p = priors.panel(metric, 2025, history.skill_seasons(lake.history_seasons()), slot_frame, pools, Settings())
    no_history = p.filter(pl.col("obs").is_null())
    assert no_history.height > 0
    # and every one of them still gets a number, because the prior is defined by his job
    assert no_history["prior"].null_count() == 0


def test_share_errors_are_scored_against_a_real_team_pool(slot_frame, pools):
    metric = priors.BY_NAME["target_share"]
    p = priors.panel(metric, 2025, history.skill_seasons(lake.history_seasons()), slot_frame, pools, Settings())
    assert (p["den_t"] > 0).all()


# --------------------------------------------------------------------------- #
# rookies
# --------------------------------------------------------------------------- #
def test_rookie_curves_are_non_increasing_in_pick_for_volume_metrics():
    curves = priors.load_curves()
    if curves.is_empty():
        pytest.skip("no fitted curves on disk; run python -m src.model.priors --fit")
    for name in ("target_share", "carry_share", "route_participation"):
        for pos in ("WR", "RB", "TE"):
            vals = (
                curves.filter((pl.col("metric") == name) & (pl.col("position") == pos))
                .sort("pick_lo")["value"].drop_nulls().to_list()
            )
            assert vals == sorted(vals, reverse=True), f"{name}/{pos}: {vals}"


def test_first_round_receivers_start_far_above_late_round_ones():
    curves = priors.load_curves()
    if curves.is_empty():
        pytest.skip("no fitted curves on disk")
    wr = curves.filter((pl.col("metric") == "target_share") & (pl.col("position") == "WR")).sort("pick_lo")
    assert wr["value"][0] > 2 * wr["value"][-1]


def test_the_rookie_blend_is_fitted_on_the_grid_and_beats_the_slot_prior_on_shares():
    """The blend has to earn its place against both endpoints it nests, not just against one."""
    check = priors.load_rookie_check()
    if check.is_empty():
        pytest.skip("no fitted rookie blend on disk; run python -m src.model.priors --fit")
    for row in check.iter_rows(named=True):
        assert row["form"] in ("additive", "capital_ratio"), row["metric"]
        assert row["w"] in priors.ROOKIE_W_GRID, row["metric"]
        assert row["mae"] <= min(row["mae_slot_prior"], row["mae_draft_curve"]) + 1e-9, row
        assert row["gain_vs_slot_pct"] >= 0.0 and row["gain_vs_curve_pct"] >= 0.0, row
    shares = check.filter(pl.col("metric").str.ends_with("_share"))
    assert shares.height >= 5
    assert shares["gain_vs_slot_pct"].min() > 5.0, "draft capital should buy real opportunity"


def test_draft_capital_buys_opportunity_and_not_efficiency():
    """The central finding: a pick predicts the size of the job, barely the quality of the play."""
    check = priors.load_rookie_check()
    if check.is_empty():
        pytest.skip("no fitted rookie blend on disk")
    kinds = {m.name: m.kind for m in priors.METRICS}
    c = check.with_columns(pl.col("metric").replace_strict(kinds, default="rate").alias("kind"))
    shares, rates = c.filter(pl.col("kind") == "share"), c.filter(pl.col("kind") == "rate")
    if shares.is_empty() or rates.is_empty():
        pytest.skip("need both kinds fitted")
    assert shares["gain_vs_slot_pct"].mean() > 5 * rates["gain_vs_slot_pct"].mean()
    assert shares["w"].mean() > rates["w"].mean()


def test_a_rookie_prior_stays_inside_its_own_units():
    """The clip lives in the estimator, so a fit and a projection cannot disagree about bounds."""
    frame = pl.DataFrame({
        "slot": [0.20, 0.20, 0.90], "curve": [0.35, None, 0.95], "norm": [0.10, 0.10, 0.30]
    })
    out = frame.with_columns(
        priors.rookie_prior(pl.col("slot"), "capital_ratio", 1.0, bounded=True).alias("share"),
        priors.rookie_prior(pl.col("slot"), "capital_ratio", 1.0, bounded=False).alias("rate"),
    )
    assert out["share"].max() <= 1.0
    assert out["share"][1] == pytest.approx(0.20), "no curve means the slot prior, untouched"
    assert out["rate"].min() >= 0.0
    assert out["rate"][2] > 1.0, "a rate is not a share and must not be clipped at one"


def test_the_slot_norm_needs_enough_rookies_to_be_a_norm(slot_frame):
    norms = priors.load_slot_norm()
    if norms.is_empty():
        pytest.skip("no fitted slot norms on disk")
    assert norms["norm"].null_count() == 0
    assert (norms["norm"] > 0).all()
    # every norm must belong to a real (position, slot_bucket) job
    jobs = slot_frame.select("position", "slot_bucket").unique()
    assert norms.join(jobs, on=["position", "slot_bucket"], how="anti").height == 0


def test_isotonic_pooling_preserves_the_weighted_mean():
    values = [0.10, 0.20, 0.15]
    weights = [10.0, 10.0, 10.0]
    out = priors._isotonic_decreasing(values, weights)
    assert out == sorted(out, reverse=True)
    assert sum(v * w for v, w in zip(out, weights, strict=True)) == pytest.approx(
        sum(v * w for v, w in zip(values, weights, strict=True))
    )


def test_isotonic_leaves_an_already_ordered_curve_alone():
    values = [0.3, 0.2, 0.1]
    assert priors._isotonic_decreasing(values, [1.0, 1.0, 1.0]) == values


# --------------------------------------------------------------------------- #
# team pools
# --------------------------------------------------------------------------- #
def test_team_pools_match_the_team_game_table(pools):
    tg = history.team_seasons(lake.history_seasons())
    joined = pools.join(tg, on=["season", "team"], how="inner")
    assert joined.height > 300
    # the pool is built from player rows, the team table from play-by-play; they must agree
    assert (joined["team_targets"] - joined["targets"]).abs().max() <= 2
    assert (joined["team_carries"] - joined["carries"]).abs().max() <= 2
    assert (joined["team_games"] == joined["games"]).all()


# --------------------------------------------------------------------------- #
# the record, as opposed to the estimate of it
# --------------------------------------------------------------------------- #
# `estimate.season_history` is what the app puts beside an override knob: the same ratio the estimator
# blends, but per season, unweighted and unshrunk. It is a different claim from `obs` and these check it
# stays one -- a "record" that had quietly been recency-weighted would be indistinguishable on screen.
HISTORY_METRICS = ("target_share", "yards_per_carry", "dropback_share")


@pytest.fixture(scope="module")
def record() -> pl.DataFrame:
    from src.model import estimate

    return estimate.season_history(HISTORY_METRICS, seasons=tuple(range(2021, 2026)))


def test_the_record_is_one_row_per_player_season_per_metric_with_its_denominator(record):
    assert set(record["metric"].unique().to_list()) == set(HISTORY_METRICS)
    assert set(record["season"].unique().to_list()) <= set(range(2021, 2026))
    assert record.height > 2000
    assert record["n"].null_count() == 0
    # a value is num / n wherever there was any opportunity at all
    got = record.filter(pl.col("n") > 0)
    assert (got["value"] - got["num"] / got["n"]).abs().max() == pytest.approx(0.0, abs=1e-12)
    # and it is not weighted or clipped: shares reach the top of their range, rates exceed one
    shares = record.filter(pl.col("kind") == "share")
    assert shares["value"].max() > 0.9
    assert record.filter(pl.col("metric") == "yards_per_carry")["value"].max() > 5.0


def test_the_record_is_the_players_own_seasons_and_only_the_ones_asked_for(record):
    from src.model import estimate

    who = record.filter(pl.col("metric") == "target_share").group_by("player_id").len()
    many = who.filter(pl.col("len") >= 3)["player_id"].to_list()[:5]
    assert many, "expected somebody with three seasons of targets in five"
    mine = estimate.season_history(("target_share",), tuple(many), seasons=(2024, 2025))
    assert set(mine["player_id"].unique().to_list()) <= set(many)
    assert set(mine["season"].unique().to_list()) <= {2024, 2025}


def test_a_metric_nobody_has_heard_of_is_skipped_rather_than_raising():
    from src.model import estimate

    got = estimate.season_history(("not_a_metric",))
    assert got.is_empty()
    assert "value" in got.columns, "the empty frame keeps its schema so a page can select from it"


def test_the_record_disagrees_with_the_recency_weighted_estimate_it_sits_beside(record):
    """The whole reason both exist: a career average and a trend are different arguments.

    If they agreed everywhere, one of them would be redundant. What is asserted is that the record
    spans the estimate rather than reproducing it -- somebody's own seasons must straddle his `obs`.
    """
    from src.model import estimate, roster

    ros = roster.roster(PROJ_SEASON)
    own = estimate.own_rate(priors.BY_NAME["target_share"], PROJ_SEASON, Settings())
    mine = (
        record.filter(pl.col("metric") == "target_share")
        .group_by("player_id").agg(pl.col("value").min().alias("lo"),
                                   pl.col("value").max().alias("hi"), pl.len().alias("seasons"))
        .filter(pl.col("seasons") >= 3)
        .join(own, on="player_id", how="inner")
        .join(ros.select("player_id"), on="player_id", how="inner")
    )
    assert mine.height > 100
    inside = mine.filter((pl.col("obs") >= pl.col("lo") - 1e-9) & (pl.col("obs") <= pl.col("hi") + 1e-9))
    assert inside.height == mine.height, "a weighted average must lie inside the seasons it averages"
    assert mine.filter((pl.col("hi") - pl.col("lo")) > 0.05).height > 50, "no trends at all"
