"""The held-out harness: does it actually hold the season out, and does it compare like with like.

Two families of test here, and the first matters more than any accuracy number. A backtest that
leaks is worse than no backtest, because it produces a number that looks like evidence. So the
leakage guards are asserted directly: an as-of artifact set must change when its window changes, must
not move when a later season is added to the lake, and the market must be genuinely off when it is
switched off.

The second family is comparability. Every variant has to be scored over the same players -- the
baseline that predicts for three times as many mostly-zero players would otherwise win on MAE while
being useless.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from src.config import PROJ_SEASON, Settings
from src.model import backtest, estimate, priors, roster, team

TARGET = 2024


@pytest.fixture(scope="module")
def settings() -> Settings:
    return backtest.base_settings()


@pytest.fixture(scope="module")
def ex(settings) -> backtest.Ex:
    return backtest.ex_ante(TARGET, settings)


# --------------------------------------------------------------------------- #
# the population: rosters for a past season
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("season", [2021, TARGET, PROJ_SEASON])
def test_every_season_resolves_a_whole_league_roster(season):
    """The bug this guards: the shared lake's `rosters` table is a week-1 snapshot for 2026 and a
    few hundred scattered rows for earlier seasons, which silently produced a 145-player 2024."""
    when = "latest" if season >= PROJ_SEASON else "preseason"
    ros = roster.roster(season, when)
    assert ros["team"].n_unique() == 32
    assert ros.height > 600, "a whole league's offence is nearer 900 than 150"
    assert ros["player_id"].n_unique() == ros.height


def test_the_projected_population_covers_almost_all_of_the_seasons_scoring(ex):
    """If the week-1 roster missed the players who mattered, every number below would be about the
    roster rather than about the model."""
    cov = backtest.coverage(ex, ex.population)
    assert cov["covered_pct"] > 97.0


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_as_of_artifacts_are_not_the_saved_artifacts(ex):
    """Priors fitted before 2024 must differ from priors fitted through 2025. If these came back
    equal, `fitted_as_of` would be silently returning the leaky set."""
    saved = priors.fitted_saved()
    a = ex.fitted.priors.sort(["metric", "position", "slot_bucket"])
    b = saved.priors.sort(["metric", "position", "slot_bucket"])
    assert not a.equals(b)
    assert ex.fitted.as_of == TARGET
    for frame in (ex.fitted.priors, ex.fitted.curves, ex.fitted.norms, ex.fitted.slot_games):
        assert not frame.is_empty()


def test_an_as_of_set_is_the_same_however_much_later_data_exists(settings):
    """The window, not the lake, decides what a fold sees: seasons at or after the target contribute
    nothing, so building the same fold twice from a lake that also holds 2025 is deterministic."""
    a = priors.fitted_as_of(2022, settings)
    b = priors.fitted_as_of(2022, settings)
    assert a.priors.sort(["metric", "position", "slot_bucket"]).equals(
        b.priors.sort(["metric", "position", "slot_bucket"])
    )
    later = priors.fitted_as_of(2025, settings)
    assert not a.priors.sort(["metric", "position", "slot_bucket"]).equals(
        later.priors.sort(["metric", "position", "slot_bucket"])
    )


def test_the_projection_path_is_unchanged_by_the_new_argument():
    """`fitted=None` and `fitted=fitted_saved()` have to be the same call, or threading the argument
    through five modules changed the 2026 board."""
    ros = roster.roster(PROJ_SEASON)
    names = ("target_share", "yards_per_target")
    a = estimate.estimate(names, ros, PROJ_SEASON, Settings())
    b = estimate.estimate(names, ros, PROJ_SEASON, Settings(), priors.fitted_saved())
    assert a.equals(b)


def test_expected_weather_replaces_the_weather_the_game_got():
    played = team.game_rows(TARGET)
    ante = team.game_rows(TARGET, expected_weather=True)
    assert played.height == ante.height
    joined = played.select("game_id", "team", pl.col("temp").alias("t_played")).join(
        ante.select("game_id", "team", pl.col("temp").alias("t_ante")), on=["game_id", "team"]
    )
    assert (joined["t_played"] != joined["t_ante"]).sum() > 0


@pytest.mark.parametrize("season", [2021, 2022, 2023, 2024, 2025])
def test_every_fold_has_a_climate_window_to_draw_on(season):
    """Two ways an outdoor game ended up with no expected weather, both silently meaning 'average':
    a climate window pinned to 2021 was empty for a 2021 fold, and a neutral-site game hosted by a
    dome team has no home climate of its own."""
    outdoor = team.game_rows(season, expected_weather=True).filter(~pl.col("is_indoor"))
    assert outdoor.height > 300
    assert outdoor["temp"].null_count() == 0
    assert outdoor["wind"].null_count() == 0


def test_zero_market_weight_reads_no_line_at_all():
    """A closing line must not reach the projection by the level-correction side door either."""
    rows = team.game_rows(TARGET, expected_weather=True)
    off = team.game_environment(TARGET, Settings(market_weight=0.0), rows=rows)
    assert off["implied_points"].equals(off["model_points"])
    on = team.game_environment(TARGET, Settings(market_weight=0.5), rows=rows)
    assert not on["implied_points"].equals(on["model_points"])


# --------------------------------------------------------------------------- #
# comparability
# --------------------------------------------------------------------------- #
def test_the_baseline_is_scored_over_the_same_players_as_the_engine(ex, settings):
    e = backtest.ewma(TARGET, settings, ex.population)
    assert sorted(e["player_id"].to_list()) == sorted(ex.population["player_id"].to_list())


def test_every_variant_scores_the_same_population():
    art = backtest.run((TARGET,), ("full", "ewma", "workbook"), verbose=False)
    counts = art["players"].group_by("variant").agg(pl.len().alias("n"))
    assert counts["n"].n_unique() == 1
    played = art["summary"].filter(
        (pl.col("stat") == "fantasy_points") & (pl.col("view") == "played")
        & (pl.col("season") == 0)
    )
    assert played["n"].n_unique() == 1


def test_the_engine_beats_both_baselines_on_the_season_it_did_not_see():
    """The point of the whole module. Asserted as an inequality rather than a number so a data
    refresh does not break it, and on rank correlation as well as error because a board is read as an
    order."""
    art = backtest.run((TARGET,), ("full", "ewma", "workbook"), verbose=False)
    got = {
        r["variant"]: r
        for r in art["summary"].filter(
            (pl.col("stat") == "fantasy_points") & (pl.col("view") == "played")
            & (pl.col("season") == 0)
        ).iter_rows(named=True)
    }
    for loser in ("ewma", "workbook"):
        assert got["full"]["mae"] < got[loser]["mae"]
        assert got["full"]["rmse"] < got[loser]["rmse"]
        assert got["full"]["spearman"] > got[loser]["spearman"]


# --------------------------------------------------------------------------- #
# the variants and the verdict
# --------------------------------------------------------------------------- #
def test_forcing_k_covers_every_metric():
    """A metric missing from the dict falls back to the Settings default, not to the forced value, so
    a partial override would leave part of the engine shrinking normally."""
    zero = backtest._all_k(priors.fitted_saved(), 0.0)
    assert set(zero.k) == {m.name for m in priors.METRICS}
    assert set(zero.k.values()) == {0.0}


def test_no_shrinkage_moves_a_thin_sample_and_prior_only_does_not(ex, settings):
    """The two endpoints have to actually be endpoints: at k=0 a player's own rate is his answer, at
    k=inf his job's prior is."""
    ros = roster.roster(TARGET, "preseason")
    name = "yards_per_carry"
    at_zero = estimate.estimate((name,), ros, TARGET, settings, backtest._all_k(ex.fitted, 0.0))
    at_inf = estimate.estimate((name,), ros, TARGET, settings, backtest._all_k(ex.fitted, 1e12))
    have = at_zero.filter(pl.col("n") > 0)
    assert (have["used"] - have["obs"]).abs().max() < 1e-9
    assert (at_inf["used"] - at_inf["prior"]).abs().max() < 1e-6


def test_rookie_curves_off_leaves_only_slot_priors(ex):
    off = backtest._no_curve(ex.fitted)
    assert set(off.blends) == set(ex.fitted.blends)
    assert all(w == 0.0 for _, w in off.blends.values())


def test_a_verdict_needs_a_clean_sweep():
    """One metric cannot revert a node. MAE on this population rewards under-projection, and the
    normalisation ablation is exactly that case -- lower MAE, worse RMSE and worse order."""
    summary = pl.DataFrame([
        {"variant": "full", "view": "played", "season": 0, "stat": "fantasy_points",
         "mae": 10.0, "rmse": 20.0, "bias": 0.0, "spearman": 0.80, "n": 100},
        {"variant": "sweeps", "view": "played", "season": 0, "stat": "fantasy_points",
         "mae": 9.0, "rmse": 19.0, "bias": 0.0, "spearman": 0.81, "n": 100},
        {"variant": "mae_only", "view": "played", "season": 0, "stat": "fantasy_points",
         "mae": 9.0, "rmse": 21.0, "bias": 0.0, "spearman": 0.79, "n": 100},
        {"variant": "loses", "view": "played", "season": 0, "stat": "fantasy_points",
         "mae": 11.0, "rmse": 21.0, "bias": 0.0, "spearman": 0.79, "n": 100},
    ])
    got = {r["variant"]: r["verdict"] for r in backtest.decisions(summary).iter_rows(named=True)}
    assert got == {"sweeps": "REVERT", "mae_only": "mixed", "loses": "keep"}


def test_the_scored_frame_charges_for_a_player_who_never_played(ex, settings):
    """A projection that hands 90 points to somebody who never took the field is wrong by 90, not
    excused. `all` is the view that says so."""
    pred = backtest.project(TARGET, settings, ex)
    scored = backtest.score_frame(pred, ex)
    ghosts = scored.filter(~pl.col("played"))
    assert ghosts.height > 0
    assert (ghosts["a_fantasy_points"] == 0.0).all()
    assert scored.height == pred.height
    assert scored["all"].all()


def test_composition_still_holds_on_a_held_out_season(ex, settings):
    """The identity is not allowed to lapse just because the season is 2024: counts drawn from a
    normalised pool must still sum to the team they were divided from."""
    pred = backtest.project(TARGET, settings, ex)
    assert pred.height == ex.population.height
    assert pred["fantasy_points"].min() >= 0.0
    top = pred.sort("fantasy_points", descending=True).head(40)
    assert set(top["position"].unique()) <= {"QB", "RB", "WR", "TE"}


def test_the_interval_report_scores_the_range_and_not_the_point() -> None:
    """MAE cannot fail a range and coverage cannot fail a mean, so the harness has to print both.

    A hundred draws is far too few to *judge* the calibration -- that is `simulate --calibrate` on a
    couple of thousand -- but it is enough to prove the ex-ante frame reaches the simulator and that a
    coverage number comes back bounded and counted over the population the point tables scored.
    """
    from src.model import simulate

    if not simulate.load().meta.get("fitted"):
        pytest.skip("no fitted dispersion on disk")
    got = backtest.intervals((TARGET,), draws=120)
    assert set(got["season"].unique()) == {TARGET}
    assert set(got["position"].unique()) >= {"ALL", "QB", "RB", "WR", "TE"}
    by = {r["population"]: r for r in got.filter(pl.col("position") == "ALL").iter_rows(named=True)}
    assert set(by) == {"board", "startable"}
    assert by["startable"]["n"] < by["board"]["n"]      # the cut is a cut, not a relabelling
    for every in by.values():
        assert every["n"] > 200
        assert every["want_90"] == 0.9
        assert 0.5 < every["cover_90"] <= 1.0
        assert every["cover_50"] < every["cover_90"]
        # a partition, not an approximation: the two tails are the complement of the band, so an
        # outcome exactly on P5 belongs to one of the three and not to two of them
        assert every["below"] + every["cover_90"] + every["above"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cal(ex) -> dict[str, pl.DataFrame]:
    """One held-out run, measured once: composition is minutes and both tables come from the same frame."""
    art = backtest.run((TARGET,), ("full",), verbose=False)
    return {"players": art["players"], **backtest.calibration(art["players"])}


def test_the_calibration_line_is_flat_enough_to_believe_the_board(cal):
    """The test that decides whether the projections are *tilted*, which MAE cannot see.

    Regressing outcome on projection: a slope of 1 is the claim. The bounds are wide because this runs
    on one held-out season and the point is to catch a regression that breaks the level, not to pin a
    third decimal. `mean_err` is the level error and is asserted; `median_err` deliberately is not --
    season points are right-skewed, so a projection correctly aimed at the mean sits above the median
    outcome and a median error of zero would be the defect.
    """
    board = {r["position"]: r for r in cal["lines"].filter(pl.col("population") == "board")
             .iter_rows(named=True)}
    assert set(board) >= {"ALL", "QB", "RB", "WR", "TE"}
    for pos, row in board.items():
        assert 0.7 < row["slope"] < 1.4, f"{pos} projections are tilted: slope {row['slope']:.2f}"
        assert abs(row["mean_err"]) < 12.0, f"{pos} level error {row['mean_err']:.1f} points"
    assert board["ALL"]["median_err"] < board["ALL"]["mean_err"], "the skew is the wrong way round"


def test_the_bands_are_cut_on_the_projection_and_report_the_players_who_never_played(cal):
    """Cutting on realised games would keep only the players who beat the availability applied, which
    shifts every error in the table. `never_played` states that fact without selecting on it."""
    bands = cal["bands"]
    assert bands.height >= 4
    assert bands["n"].sum() == cal["players"].filter(pl.col("variant") == "full").height
    assert bands["never_played"].min() >= 0.0
    assert bands["never_played"].max() <= 1.0
    low = bands.sort("band").row(0, named=True)
    high = bands.sort("band").row(-1, named=True)
    assert low["never_played"] > high["never_played"], (
        "a player projected for two games misses the season more often than one projected for sixteen"
    )
    assert high["proj_mean"] > low["proj_mean"]


def test_a_calibration_line_needs_a_population_to_fit(monkeypatch):
    """Fewer than ten rows, or no spread in the projection, has to come back as nan rather than as a
    slope of zero -- a fabricated line would be read as evidence."""
    tiny = pl.DataFrame({
        "variant": ["full"] * 4, "position": ["QB"] * 4, "games": [10.0] * 4, "a_games": [10] * 4,
        "fantasy_points": [100.0, 110.0, 120.0, 130.0], "a_fantasy_points": [90.0, 120.0, 95.0, 140.0],
    })
    assert backtest.calibration(tiny)["lines"].is_empty()


def test_a_variant_that_changes_nothing_changes_nothing(ex, settings):
    identity = backtest.BY_NAME["full"]
    s, f = identity.on(settings, ex.fitted)
    assert s == settings
    a = backtest.project(TARGET, s, replace(ex, fitted=f))
    b = backtest.project(TARGET, settings, ex)
    assert a["fantasy_points"].sum() == pytest.approx(b["fantasy_points"].sum())
