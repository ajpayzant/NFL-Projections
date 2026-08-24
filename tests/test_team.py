"""The team environment: season shape, the per-game factor chain, and the 2026 schedule.

Like the other suites these run against the real lake and assert invariants rather than fixed
numbers, so a data refresh cannot break them for the wrong reason. The two that matter most are the
leakage guard (a projection for season S must not read season S) and the composition guard (a season
roll-up must equal the sum of its games).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.config import LAST_COMPLETE_SEASON, PROJ_SEASON, REG_WEEKS, Settings
from src.model import team


@pytest.fixture(scope="module")
def rows() -> pl.DataFrame:
    return team.game_rows(PROJ_SEASON)


@pytest.fixture(scope="module")
def shape() -> pl.DataFrame:
    return team.season_shape(PROJ_SEASON)


@pytest.fixture(scope="module")
def env() -> pl.DataFrame:
    return team.game_environment(PROJ_SEASON)


@pytest.fixture(scope="module")
def fit() -> dict[str, dict]:
    f = team.load_shape_fit()
    if not f:
        pytest.skip("no fitted shape on disk; run python -m src.model.team --fit")
    return f


# --------------------------------------------------------------------------- #
# the schedule the projection is built on
# --------------------------------------------------------------------------- #
def test_every_team_plays_seventeen_games_across_eighteen_weeks_with_one_bye(rows):
    per_team = rows.group_by("team").agg(pl.len().alias("games"), pl.col("week").n_unique())
    assert per_team.height == 32
    assert (per_team["games"] == 17).all()
    weeks = set(rows["week"].unique().to_list())
    assert weeks == set(range(1, REG_WEEKS + 1))
    # two rows per game, one per side, and nobody plays twice in a week
    assert rows.group_by("game_id").len()["len"].unique().to_list() == [2]
    assert rows.group_by(["team", "week"]).len()["len"].max() == 1


def test_the_two_sides_of_a_game_are_mirror_images(rows):
    a = rows.select("game_id", "team", "opponent", "is_home", "market_spread", "market_points")
    b = a.select(pl.all().name.suffix("_o")).rename({"game_id_o": "game_id"})
    j = a.join(b, on="game_id").filter(pl.col("team") != pl.col("team_o"))
    assert j.height == rows.height
    assert (j["team"] == j["opponent_o"]).all()
    # exactly one home side, except at a neutral site where neither is
    home = j.group_by("game_id").agg(pl.col("is_home").sum().alias("n"))
    assert set(home["n"].to_list()) <= {0, 1}
    lined = j.drop_nulls("market_spread")
    assert (lined["market_spread"] + lined["market_spread_o"]).abs().max() == pytest.approx(0.0)


def test_no_projection_row_is_missing_the_environment_it_needs(rows):
    """Roof drives the weather factors, so a null there would silently neutralise them."""
    assert rows["roof"].null_count() == 0
    assert rows["is_indoor"].null_count() == 0
    indoor = rows.filter("is_indoor")
    assert indoor.height > 0
    assert indoor["temp"].null_count() == indoor.height
    assert indoor["wind"].null_count() == indoor.height
    outdoor = rows.filter(~pl.col("is_indoor"))
    assert outdoor["temp"].null_count() == 0, "climate table should cover every outdoor venue"


def test_historical_rows_look_like_projection_rows_and_hide_their_actuals(rows):
    hist = team.historical_rows(LAST_COMPLETE_SEASON)
    assert set(team.ROW_SCHEMA) <= set(hist.columns)
    assert hist.select(team.ROW_SCHEMA).schema == rows.select(team.ROW_SCHEMA).schema
    # anything the season actually did is prefixed, so no fit can reach it by accident
    outcomes = {"points", "plays", "carries", "targets", "pass_attempts"}
    assert not outcomes & set(hist.columns)
    assert {f"a_{c}" for c in outcomes} <= set(hist.columns)


# --------------------------------------------------------------------------- #
# stage 1: season shape
# --------------------------------------------------------------------------- #
def test_the_estimate_lies_between_the_league_mean_and_the_teams_own_form(shape):
    s = shape.drop_nulls(["estimate", "league_mean", "recent_form"])
    lo = pl.min_horizontal("league_mean", "recent_form")
    hi = pl.max_horizontal("league_mean", "recent_form")
    off = s.with_columns((pl.col("estimate") < lo - 1e-9).alias("under"),
                         (pl.col("estimate") > hi + 1e-9).alias("over"))
    assert off["under"].sum() == 0
    assert off["over"].sum() == 0


def test_keep_and_w_stay_inside_the_grid(fit):
    for key, rec in fit.items():
        assert 0.0 <= rec["keep"] <= 1.0, key
        assert 0.5 <= rec["w"] <= 1.0, key


def test_a_metric_that_lost_out_of_sample_carries_nothing_forward(fit):
    reverted = [k for k, v in fit.items() if v["reverted_to_league_mean"]]
    assert reverted, "the point of the auto-revert is that some metrics fail; none did"
    for key in reverted:
        assert fit[key]["keep"] == 0.0, key
        assert fit[key]["gain_vs_mean_pct"] <= 0.0 + 1e-9, key
    for key, rec in fit.items():
        if rec["keep"] > 0:
            assert rec["gain_vs_mean_pct"] > 0, f"{key} beats nothing yet still carries form"


def test_a_reverted_metric_is_the_same_number_for_every_team(shape, fit):
    reverted = [k for k, v in fit.items() if v["reverted_to_league_mean"]]
    if not reverted:
        pytest.skip("nothing reverted")
    side, metric = reverted[0].split(".", 1)
    vals = shape.filter((pl.col("side") == side) & (pl.col("metric") == metric))["estimate"]
    assert vals.n_unique() == 1


def test_season_shape_never_reads_the_season_it_projects():
    """The stage-1 analogue of the priors leakage guard: corrupt 2025 and 2025 must not move."""
    seasons = tuple(s for s in range(2016, LAST_COMPLETE_SEASON + 1))
    clean = team.shape_panel(seasons)
    target = LAST_COMPLETE_SEASON
    poisoned = clean.with_columns(
        pl.when(pl.col("season") == target).then(pl.lit(9999.0)).otherwise(pl.col("value")).alias("value")
    )
    a = team._fit_rows(poisoned, [target], require_actual=False)
    b = team._fit_rows(clean, [target], require_actual=False)
    j = a.join(b, on=["season", "team", "side", "metric"], suffix="_clean")
    assert j.height > 1000
    for col in ("last", "prev", "lmean1", "lmean2"):
        assert (j[col] - j[f"{col}_clean"]).abs().max() == pytest.approx(0.0), col


def test_shape_wide_gives_every_team_an_offence_and_a_defence(shape):
    wide = team.shape_wide(PROJ_SEASON)
    assert wide.height == 32
    assert any(c.startswith("off_") for c in wide.columns)
    assert any(c.startswith("def_") for c in wide.columns)
    assert wide.select(pl.exclude("team")).null_count().max_horizontal()[0] == 0


# --------------------------------------------------------------------------- #
# stage 2: the factor chain
# --------------------------------------------------------------------------- #
def test_every_splits_factors_average_one_under_their_own_counts():
    """A split may move a game but must not move the league: otherwise the chain inflates."""
    f = team.load_factors()
    if f.is_empty():
        pytest.skip("no fitted factors on disk")
    splits = f.filter(pl.col("kind") == "split")
    level = splits.group_by(["metric", "term"]).agg(
        ((pl.col("value") * pl.col("n")).sum() / pl.col("n").sum()).alias("level")
    )
    assert (level["level"] - 1.0).abs().max() < 1e-6


def test_a_metric_whose_context_lost_has_no_factor_rows_and_so_a_flat_chain():
    f, scores = team.load_factors(), team.load_context_scores()
    if f.is_empty() or scores.is_empty():
        pytest.skip("no fitted context on disk")
    dropped = scores.filter(~pl.col("context_used"))["metric"].to_list()
    for metric in dropped:
        assert f.filter(pl.col("metric") == metric).height == 0
    assert (scores.filter("context_used")["gain_pct"] > 0).all()


def test_context_can_be_switched_off_entirely(rows):
    off = team.game_environment(PROJ_SEASON, Settings(use_context_factors=False), rows=rows)
    factors = [c for c in off.columns if c.startswith("f_")]
    assert factors
    for c in factors:
        assert (off[c] - 1.0).abs().max() == pytest.approx(0.0)


def test_factors_are_bounded_and_finite(env):
    for c in [c for c in env.columns if c.startswith("f_")]:
        v = env[c].to_numpy()
        assert np.isfinite(v).all(), c
        assert v.min() >= team.FACTOR_FLOOR - 1e-9, c
        assert v.max() < 3.0, c


def test_marginal_and_partial_factors_both_exist_and_differ():
    f = team.load_factors()
    if f.is_empty():
        pytest.skip("no fitted factors on disk")
    assert f["marginal"].null_count() == 0
    # the raw home/away points split is much bigger than what survives pricing the line in
    venue = f.filter((pl.col("metric") == "points") & (pl.col("term") == "venue"))
    home = venue.filter(pl.col("bucket") == "home")
    assert home["marginal"][0] > home["value"][0] > 1.0


def test_the_defence_term_reads_from_predictable_ratings(fit):
    """Every rating the chain multiplies must be one stage 1 actually carries forward."""
    for rating in team.DEF_RATINGS:
        rec = fit.get(f"defense.{rating}") or fit.get(f"offense.{rating}")
        assert rec is not None, rating
        assert not rec["reverted_to_league_mean"], f"{rating} is league-average for every team"


# --------------------------------------------------------------------------- #
# composition: games must add up to the season
# --------------------------------------------------------------------------- #
def test_a_season_is_exactly_the_sum_of_its_games(env):
    se = team.season_environment(PROJ_SEASON)
    assert se.height == 32
    assert (se["games"] == 17).all()
    per_game = env.group_by("team").agg(
        pl.col("carries").sum().alias("carries"), pl.col("targets").sum().alias("targets"),
        pl.col("points").sum().alias("points"),
    )
    j = se.join(per_game, on="team", suffix="_sum")
    for c in ("carries", "targets", "points"):
        assert (j[c] - j[f"{c}_sum"]).abs().max() == pytest.approx(0.0, abs=1e-6)


def test_projected_volumes_are_recognisably_football(env):
    per_team = env.group_by("team").agg(pl.col("points").sum(), pl.col("plays").sum(),
                                        pl.col("carries").sum(), pl.col("pass_attempts").sum())
    assert 250 <= per_team["points"].min() and per_team["points"].max() <= 600
    assert 900 <= per_team["plays"].min() and per_team["plays"].max() <= 1200
    assert 300 <= per_team["carries"].min() and per_team["carries"].max() <= 650
    assert 400 <= per_team["pass_attempts"].min() and per_team["pass_attempts"].max() <= 750


def test_renormalising_the_schedule_leaves_the_flat_season_total_alone(rows):
    """With the schedule renormalised a team's 17 factors average 1, so only the shape moves."""
    on = team.season_environment(PROJ_SEASON, Settings(schedule_renormalise=True))
    assert (on["schedule_factor"] - 1.0).abs().max() < 1e-9
    for c in ("carries", "targets"):
        assert (on[c] - on[f"{c}_flat"]).abs().max() < 0.5


def test_the_schedule_moves_teams_in_both_directions(env):
    se = team.season_environment(PROJ_SEASON)
    assert se["schedule_factor"].min() < 1.0 < se["schedule_factor"].max()
    # and by a believable amount: no team's schedule is worth ten percent of its offence
    assert (se["schedule_factor"] - 1.0).abs().max() < 0.10


# --------------------------------------------------------------------------- #
# the market
# --------------------------------------------------------------------------- #
def test_the_market_weight_is_fitted_on_week_one_only():
    m = team.load_market()
    if not m:
        pytest.skip("no fitted market on disk")
    assert 0.0 <= m["market_weight"] <= 1.0
    assert m["n"] < m["n_all_weeks"], "week-1 fit should see far fewer games than all weeks"
    # blending must beat both of the things it blends, or there is no reason to blend
    assert m["mae"] <= min(m["mae_market_only"], m["mae_model_only"]) + 1e-9


def test_points_uses_the_blend_and_keeps_the_chains_own_answer_beside_it(env):
    assert "points" in env.columns and "points_chain" in env.columns
    lined = env.drop_nulls("market_points")
    assert lined.height > 0
    assert (lined["points"] - lined["points_chain"]).abs().max() > 0.1


def test_teams_without_a_posted_line_still_get_a_level_consistent_with_the_ones_that_do(env):
    """The market offset exists so a team's lined and unlined games sit on one scale."""
    gap = env.group_by("team").agg(
        pl.col("points").filter(pl.col("market_points").is_not_null()).mean().alias("lined"),
        pl.col("points").filter(pl.col("market_points").is_null()).mean().alias("unlined"),
    ).drop_nulls()
    assert gap.height > 20
    assert (gap["lined"] - gap["unlined"]).abs().max() < 6.0
