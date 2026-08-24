"""The Monte Carlo's invariants, on a frame small enough to reason about by hand.

A simulation is the hardest part of the engine to check, because almost any bug produces numbers that
look like plausible ranges. So the tests here are not "does it run" -- they are the handful of
properties that have to hold or the ranges mean nothing:

- the median lands on the point projection, so a range is a range *around* the board rather than a
  second and quieter projection;
- a wider dispersion is a wider interval, monotonically;
- volume drives volatility, so a twelve-target receiver is a safer bet than a two-target one at the
  same projection;
- the pool stays whole per draw, which is what makes teammates anti-correlated and an override
  zero-sum;
- the same seed is the same answer, because a floor that moves on a rerun cannot be argued with;
- and a share moved on the team page moves the player's range, which is the point of the feature.

The fixtures are two teams of four, one week, built by hand. Nothing here reads the lake.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from src.config import Settings
from src.model import simulate
from src.model.simulate import Dispersion


def _weekly(weeks: int = 1, target_shares=(0.30, 0.25, 0.20, 0.10), p_play: float = 1.0,
            team_targets: float = 40.0, ypt: float = 8.0, td_rate: float = 0.05,
            rec_rate: float = 0.65) -> pl.DataFrame:
    """One or more weeks of a two-team league: four receivers each, on a fixed target pool.

    Written in the projection's own form -- counts already multiplied by `p_play`, points already
    scored -- because that is what `run` is handed, and building it any other way would test a frame
    the app never produces.
    """
    rows = []
    for w in range(1, weeks + 1):
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            for i, share in enumerate(target_shares):
                targets = share * team_targets * p_play
                yards = targets * ypt
                rows.append({
                    "player_id": f"{team}{i}", "player": f"{team} {i}", "position": "WR",
                    "team": team, "opponent": opp, "game_id": f"{w}_AAA_BBB", "week": w,
                    "p_play": p_play, "targets": targets, "carries": 0.0, "attempts": 0.0,
                    "receptions": targets * rec_rate, "receiving_yards": yards,
                    "receiving_tds": targets * td_rate,
                    "fantasy_points": targets * rec_rate + yards * 0.1 + targets * td_rate * 6,
                })
    return pl.DataFrame(rows)


def _disp(share: float = 0.0, rate: float = 0.0, usage: float = 0.0, event: float = 0.0,
          team_week: float = 0.0, team_season: float = 0.0, rho: float = 0.0,
          games: tuple[float, ...] = (1.0,)) -> Dispersion:
    """A dispersion with every sigma under the caller's control, so one channel can be moved alone."""
    chans = [c.name for c in simulate.CHANNELS]
    groups = [g.name for g in simulate.GROUPS]
    return Dispersion(
        team_week={c: team_week for c in chans},
        team_rho={c: rho for c in chans},
        team_season={c: team_season for c in chans},
        usage_cv2={f"{p}:{c}": usage for p in simulate.POSITIONS for c in chans},
        event_cv2={f"{p}:{g}": event for p in simulate.POSITIONS for g in groups},
        share_season={f"{p}:{t}:{c}": share for p in simulate.POSITIONS for t in simulate.TIERS
                      for c in chans},
        rate_season={f"{p}:{t}:{g}": rate for p in simulate.POSITIONS for t in simulate.TIERS
                     for g in groups},
        games_ratio={f"{p}:{t}": list(games) for p in simulate.POSITIONS for t in simulate.TIERS},
        boom={p: 15.0 for p in simulate.POSITIONS},
        bust={p: 5.0 for p in simulate.POSITIONS},
    )


SETTINGS = Settings()
DRAWS = 4000


# --------------------------------------------------------------------------- #
# the median is the projection
# --------------------------------------------------------------------------- #
def test_with_no_dispersion_the_only_spread_left_is_the_count_and_the_mean_is_the_projection() -> None:
    """The degenerate case pins the arithmetic: no shocks, no injuries, nothing left to be unsure of.

    Not quite nothing. Weekly share variance carries a `1/expected_count` Poisson term that no sigma
    can switch off, and correctly so -- a two-target week is lumpy whatever we know about the player.
    So the claim here is the one that has to hold exactly (the mean is the projection) plus the shape
    of what remains: on this much volume the residual spread is a couple of per cent, and it is the
    count rather than a leak, which the volume test below confirms by making it move.
    """
    wk = _weekly(weeks=3, td_rate=0.0, team_targets=4_000.0, ypt=1.0)
    got = simulate.run(wk, SETTINGS, _disp(), draws=1000).season.sort("player_id")
    assert got["sim_mean"].to_numpy() == pytest.approx(got["projected"].to_numpy(), rel=3e-3)
    assert got["p50"].to_numpy() == pytest.approx(got["projected"].to_numpy(), rel=1e-2)
    # the residual is the count and nothing else, so it is bounded by the Poisson span of the
    # thinnest workload in the frame and it is that player who carries it
    cv = (got["range"] / got["sim_mean"]).to_numpy()
    thin = wk.filter(pl.col("player_id") == "AAA3")["targets"].sum()
    assert cv.max() < 3.3 / np.sqrt(thin)
    assert cv.argmax() in (3, 7)


def test_a_touchdown_is_a_poisson_draw_and_so_survives_every_sigma_being_zero() -> None:
    """Not a defect: at four expected scores a season the spread *is* the count, not our ignorance."""
    sim = simulate.run(_weekly(weeks=17), SETTINGS, _disp(), draws=2000)
    got = sim.season.sort("player_id")
    assert (got["p95"] > got["p5"]).all()
    assert got["sim_mean"].to_numpy() == pytest.approx(got["projected"].to_numpy(), rel=0.02)


def test_the_simulated_mean_stays_on_the_projection_when_the_shocks_are_real() -> None:
    """Every shock is mean one, so the total is the projection's total and not a re-levelled one.

    Checked in aggregate rather than per player: normalisation makes an individual's mean drift a
    little, on purpose, but the pool it is drawn from cannot move.
    """
    wk = _weekly(weeks=6, p_play=0.9)
    d = _disp(share=0.4, rate=0.2, usage=0.3, event=1.0, team_week=0.2, team_season=0.1,
              games=(0.6, 0.9, 1.0, 1.1, 1.4))
    sim = simulate.run(wk, SETTINGS, d, draws=DRAWS)
    assert sim.season["sim_mean"].sum() == pytest.approx(sim.season["projected"].sum(), rel=0.02)


def test_the_draws_are_in_the_row_order_of_the_season_frame() -> None:
    """The frame is sorted by median and the draws are not, unless somebody keeps them together.

    This is the one alignment in `Sim` that nothing else can catch: every quantile column is computed
    inside the sort, so a permuted `points_draws` still produces a season table that reads perfectly
    while `points_draws[i]` belongs to a different player. Asserted per row against the quantiles the
    frame published, which is the only statement of the invariant that cannot be satisfied by accident.
    """
    d = _disp(share=0.3, rate=0.2, usage=0.3, event=1.0, team_week=0.2, games=(0.7, 1.0, 1.2))
    sim = simulate.run(_weekly(weeks=6, p_play=0.9), SETTINGS, d, draws=1000)
    assert sim.season["p50"].is_sorted(descending=True)
    for i, row in enumerate(sim.season.iter_rows(named=True)):
        mine = sim.points_draws[i]
        assert float(np.median(mine)) == pytest.approx(row["p50"], abs=1e-3)
        assert float(mine.mean()) == pytest.approx(row["sim_mean"], rel=1e-4)
        assert sim.index[row["player_id"]] == i
        assert sim.draws_of(row["player_id"]) is not None
        assert float(np.quantile(mine, 0.95)) == pytest.approx(row["p95"], abs=1e-3)


def test_the_median_of_the_top_player_is_near_his_projection() -> None:
    wk = _weekly(weeks=17, p_play=0.94)
    d = _disp(share=0.3, rate=0.2, usage=0.3, event=1.0, team_week=0.2,
              games=(0.7, 0.9, 1.0, 1.05, 1.15))
    got = simulate.run(wk, SETTINGS, d, draws=DRAWS).season.sort("projected", descending=True)
    top = got.row(0, named=True)
    assert top["p50"] == pytest.approx(top["projected"], rel=0.10)
    assert top["p5"] < top["p50"] < top["p95"]


# --------------------------------------------------------------------------- #
# more dispersion is a wider interval
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["share", "rate", "usage", "team_week"])
def test_every_sigma_widens_the_interval_it_belongs_to(field: str) -> None:
    wk = _weekly(weeks=8)
    narrow = simulate.run(wk, SETTINGS, _disp(**{field: 0.05}), draws=2000).season.sort("player_id")
    wide = simulate.run(wk, SETTINGS, _disp(**{field: 0.60}), draws=2000).season.sort("player_id")
    assert (wide["range"] > narrow["range"]).all()
    assert (wide["sim_sd"] > narrow["sim_sd"]).all()


def test_an_injury_distribution_lowers_the_floor_without_moving_the_mean() -> None:
    """What availability is for: the floor is a season he missed, not a season he was bad in."""
    wk = _weekly(weeks=17, p_play=0.95)
    healthy = simulate.run(wk, SETTINGS, _disp(games=(1.0,)), draws=2000).season.sort("player_id")
    fragile = simulate.run(wk, SETTINGS, _disp(games=(0.2, 0.6, 1.0, 1.3, 1.4)),
                           draws=2000).season.sort("player_id")
    assert (fragile["p5"] < healthy["p5"]).all()
    assert fragile["sim_mean"].sum() == pytest.approx(healthy["sim_mean"].sum(), rel=0.03)
    assert (fragile["games_p5"] < healthy["games_p5"]).all()


# --------------------------------------------------------------------------- #
# volume is what separates the safe projection from the volatile one
# --------------------------------------------------------------------------- #
def test_the_same_projection_on_more_volume_is_the_safer_one() -> None:
    """The plan's own test of the volatility score, and the reason a rate is a per-event dispersion.

    Two receivers projected for identical points: one on twelve targets a game at eight yards each,
    one on three at thirty-two. Per-event noise averages out over twelve events and does not over
    three, so the same projection has to come back with a narrower range on the volume.
    """
    high = _weekly(weeks=17, target_shares=(0.30, 0.30, 0.20, 0.20), team_targets=40.0, ypt=8.0,
                   td_rate=0.05, rec_rate=0.65)
    # a quarter of the volume at four times the yield, and the per-target rates lifted to match, so
    # the two fixtures are the same points and differ in nothing but how lumpy they are
    low = _weekly(weeks=17, target_shares=(0.30, 0.30, 0.20, 0.20), team_targets=10.0, ypt=32.0,
                  td_rate=0.20, rec_rate=2.60)
    d = _disp(event=1.2, usage=0.3)
    a = simulate.run(high, SETTINGS, d, draws=2000).season.sort("player_id")
    b = simulate.run(low, SETTINGS, d, draws=2000).season.sort("player_id")
    assert a["projected"].to_numpy() == pytest.approx(b["projected"].to_numpy(), rel=1e-6)
    assert (a["volatility"] < b["volatility"]).all()


# --------------------------------------------------------------------------- #
# the pool is whole, so teammates trade
# --------------------------------------------------------------------------- #
def test_teammates_are_anti_correlated_under_normalisation_and_not_without_it() -> None:
    """The share shock is zero-sum inside a receiving room, which is a testable consequence.

    Normalisation on, one receiver's good season is another's bad one. Off, the two are independent.
    Two teams in the fixture, so the same test also says the effect is inside a team rather than
    across the league.
    """
    wk = _weekly(weeks=17)
    d = _disp(share=0.4, usage=0.3)
    on = simulate.run(wk, SETTINGS, d, draws=3000)
    off = simulate.run(wk, replace(SETTINGS, normalize_pools=False), d, draws=3000)

    def rho(sim, a: str, b: str) -> float:
        return float(np.corrcoef(sim.draws_of(a), sim.draws_of(b))[0, 1])

    assert rho(on, "AAA0", "AAA1") < -0.15
    assert abs(rho(off, "AAA0", "AAA1")) < 0.10
    assert abs(rho(on, "AAA0", "BBB0")) < 0.10


def test_the_game_correlation_ties_the_two_teams_together_with_the_sign_it_was_given() -> None:
    """`team_rho` is measured with a sign, so a negative one has to come back negative."""
    wk = _weekly(weeks=17)
    same = simulate.run(wk, SETTINGS, _disp(team_week=0.5, rho=0.9), draws=3000)
    opposed = simulate.run(wk, SETTINGS, _disp(team_week=0.5, rho=-0.9), draws=3000)

    def across(sim) -> float:
        a = sim.draws_of("AAA0") + sim.draws_of("AAA1")
        b = sim.draws_of("BBB0") + sim.draws_of("BBB1")
        return float(np.corrcoef(a, b)[0, 1])

    assert across(same) > 0.3
    assert across(opposed) < -0.3


# --------------------------------------------------------------------------- #
# reproducibility, and the questions the app asks
# --------------------------------------------------------------------------- #
def test_the_same_seed_is_the_same_floor() -> None:
    wk = _weekly(weeks=4)
    d = _disp(share=0.3, rate=0.2, usage=0.3, event=1.0, games=(0.5, 1.0, 1.25))
    a = simulate.run(wk, SETTINGS, d, draws=500)
    b = simulate.run(wk, SETTINGS, d, draws=500)
    assert a.season["p5"].to_list() == b.season["p5"].to_list()
    assert simulate.run(wk, SETTINGS, d, draws=500, seed=7).season["p5"].to_list() != \
        a.season["p5"].to_list()


def test_chunking_the_draws_does_not_change_the_answer_it_only_bounds_the_memory() -> None:
    """Including the detail frames: those span every draw rather than accumulating within a chunk."""
    wk = _weekly(weeks=4)
    d = _disp(share=0.3, usage=0.3, event=1.0)
    whole = simulate.run(wk, SETTINGS, d, draws=1200, chunk=1200, detail=("AAA0",))
    split = simulate.run(wk, SETTINGS, d, draws=1200, chunk=400, detail=("AAA0",))
    assert whole.season["p50"].to_numpy() == pytest.approx(split.season["p50"].to_numpy(), rel=0.05)
    assert whole.weekly["p50"].to_numpy() == pytest.approx(split.weekly["p50"].to_numpy(), rel=0.10)
    assert (split.weekly["mean"] > 0).all()
    assert whole.stats["mean"].to_numpy() == pytest.approx(split.stats["mean"].to_numpy(), rel=0.05)


def test_p_over_reads_the_draws_rather_than_interpolating_the_quantiles() -> None:
    wk = _weekly(weeks=17)
    sim = simulate.run(wk, SETTINGS, _disp(share=0.3, usage=0.3, event=1.0), draws=2000)
    row = sim.season.filter(pl.col("player_id") == "AAA0").row(0, named=True)
    assert sim.p_over("AAA0", row["p50"]) == pytest.approx(0.5, abs=0.03)
    assert sim.p_over("AAA0", row["p95"]) == pytest.approx(0.05, abs=0.02)
    assert sim.p_over("AAA0", row["p5"]) == pytest.approx(0.95, abs=0.02)
    assert np.isnan(sim.p_over("nobody", 100.0))


def test_the_detail_players_get_a_week_by_week_range_and_a_range_per_stat() -> None:
    wk = _weekly(weeks=17)
    sim = simulate.run(wk, SETTINGS, _disp(share=0.3, usage=0.4, event=1.0), draws=1000,
                       detail=("AAA0",))
    assert sim.weekly["player_id"].unique().to_list() == ["AAA0"]
    assert sim.weekly.height == 17
    assert (sim.weekly["p95"] > sim.weekly["p5"]).all()

    stats = dict(zip(sim.stats["stat"], sim.stats["mean"], strict=True))
    want = wk.filter(pl.col("player_id") == "AAA0")["targets"].sum()
    assert stats["targets"] == pytest.approx(want, rel=0.05)
    assert set(sim.stats["stat"]) >= {"targets", "receptions", "receiving_yards", "receiving_tds"}
    assert (sim.stats["p95"] > sim.stats["p5"]).all()


def test_the_distribution_is_binned_for_plotting_with_the_lump_at_zero_kept_apart() -> None:
    """A histogram of a fragile season is bimodal, and a plot that smooths that away is a wrong plot."""
    wk = _weekly(weeks=17, p_play=0.8)
    sim = simulate.run(wk, SETTINGS, _disp(share=0.3, usage=0.3, games=(0.0, 0.3, 1.0, 1.3)),
                       draws=2000)
    h = sim.histogram("AAA0", bins=30)
    assert h["share"].sum() == pytest.approx(1.0, abs=0.02)
    assert h["cumulative"][-1] == pytest.approx(h["share"].sum())
    assert h["points"].is_sorted()
    # the first bin is the seasons he never played, and this sample has some
    assert h["points"][0] < 1.0
    assert h["share"][0] > 0.0
    assert simulate.run(wk, SETTINGS, _disp(), draws=8).histogram("nobody").is_empty()


def test_each_of_the_detail_players_weeks_carries_its_own_boom_and_bust_rate() -> None:
    wk = _weekly(weeks=17)
    sim = simulate.run(wk, SETTINGS, _disp(usage=0.4, event=1.0), draws=1000, detail=("AAA0",))
    got = sim.weekly
    assert {"boom_rate", "bust_rate"} <= set(got.columns)
    assert ((got["boom_rate"] >= 0.0) & (got["boom_rate"] <= 1.0)).all()
    # the fixture's 21-point weeks sit over the 15-point boom line and well over the 5-point bust one
    assert got["boom_rate"].min() > 0.5
    assert got["bust_rate"].max() < 0.15
    # and against a line nobody can reach, no week booms
    high = simulate.run(wk, SETTINGS, replace_thresholds(_disp(usage=0.4), boom=1e6, bust=1e6),
                        draws=200, detail=("AAA0",)).weekly
    assert high["boom_rate"].max() == pytest.approx(0.0)
    assert high["bust_rate"].min() == pytest.approx(1.0)


def test_moving_a_share_moves_that_players_range_and_his_teammates_with_it() -> None:
    """The plan's own acceptance test, at the level of the simulation: the range follows the edit.

    Nothing is re-fitted here. The edited weekly frame is a different frame, and the sim shocks the
    frame it is handed, which is why an override reaches the floor and the ceiling for free.
    """
    wk = _weekly(weeks=17)
    d = _disp(share=0.3, usage=0.3, event=1.0)
    before = simulate.run(wk, SETTINGS, d, draws=2000).season

    # hand the fourth receiver's share to the first, in the counts, the way an override would
    moved = wk.with_columns(
        pl.when(pl.col("player_id") == "AAA0").then(pl.col("targets") * 1.5)
        .when(pl.col("player_id") == "AAA3").then(pl.col("targets") * 0.5)
        .otherwise(pl.col("targets")).alias("targets")
    )
    moved = moved.with_columns(
        (pl.col("targets") * 0.65).alias("receptions"),
        (pl.col("targets") * 8.0).alias("receiving_yards"),
        (pl.col("targets") * 0.05).alias("receiving_tds"),
    ).with_columns(
        (pl.col("receptions") + pl.col("receiving_yards") * 0.1 + pl.col("receiving_tds") * 6)
        .alias("fantasy_points")
    )
    after = simulate.run(moved, SETTINGS, d, draws=2000).season

    def of(frame: pl.DataFrame, pid: str) -> dict:
        return frame.filter(pl.col("player_id") == pid).row(0, named=True)

    up, down = of(after, "AAA0"), of(after, "AAA3")
    was_up, was_down = of(before, "AAA0"), of(before, "AAA3")
    assert up["p50"] > was_up["p50"] * 1.2 and up["p95"] > was_up["p95"] * 1.2
    assert down["p50"] < was_down["p50"] * 0.8 and down["p5"] < was_down["p5"] * 0.8
    # the untouched teammate is untouched, because the edit was zero-sum inside the room
    assert of(after, "AAA1")["p50"] == pytest.approx(of(before, "AAA1")["p50"], rel=0.05)
    assert of(after, "BBB0")["p50"] == pytest.approx(of(before, "BBB0")["p50"], rel=0.05)


# --------------------------------------------------------------------------- #
# boom, bust and the fitted artifact
# --------------------------------------------------------------------------- #
def test_boom_and_bust_are_rates_against_the_thresholds_they_were_given() -> None:
    wk = _weekly(weeks=17)
    d = _disp(usage=0.4, event=1.0)
    row = simulate.run(wk, SETTINGS, d, draws=2000).season.filter(pl.col("player_id") == "AAA0") \
        .row(0, named=True)
    assert 0.0 <= row["bust_rate"] <= 1.0 and 0.0 <= row["boom_rate"] <= 1.0
    # a threshold nobody can miss is a bust rate of one, and one nobody can reach is a boom rate of nil
    high = simulate.run(wk, SETTINGS, replace_thresholds(d, boom=1e6, bust=1e6), draws=200).season
    assert high["bust_rate"].max() == pytest.approx(1.0)
    assert high["boom_rate"].max() == pytest.approx(0.0)


def replace_thresholds(d: Dispersion, boom: float, bust: float) -> Dispersion:
    return replace(d, boom={p: boom for p in simulate.POSITIONS},
                   bust={p: bust for p in simulate.POSITIONS})


def test_the_fitted_dispersion_round_trips_through_json(tmp_path) -> None:
    d = _disp(share=0.3, rate=0.2, games=(0.5, 1.0, 1.5))
    path = tmp_path / "dispersion.json"
    d.save(path)
    back = simulate.load(path)
    assert back.k_share_season("WR", "starter", "targets") == d.k_share_season("WR", "starter",
                                                                              "targets")
    assert back.games_ratio["WR:depth"] == [0.5, 1.0, 1.5]
    # the sample is resampled from, so it is stored as a fixed number of quantiles of what was given
    got = back.games_sample("WR", "depth")
    assert got.size == simulate.ATOMS
    assert (got.min(), got.max()) == pytest.approx((0.5, 1.5), abs=0.01)
    assert float(got.mean()) == pytest.approx(1.0, abs=0.02)
    assert back.thresholds("RB") == d.thresholds("RB")


def test_an_unfitted_dispersion_still_produces_a_range(tmp_path) -> None:
    """The app should show a range on a fresh checkout, and say the numbers are not fitted."""
    d = simulate.load(tmp_path / "nothing.json")
    assert d.meta == {"fitted": False}
    got = simulate.run(_weekly(weeks=4), SETTINGS, d, draws=200).season
    assert (got["p95"] > got["p5"]).all()


def test_the_calibration_scale_is_the_only_thing_that_moves_a_fitted_sigma() -> None:
    d = _disp(share=0.3, rate=0.2)
    twice = replace(d, scale={p: 2.0 for p in simulate.POSITIONS})
    assert twice.k_share_season("WR", "starter", "targets") == pytest.approx(0.6)
    assert twice.k_rate_season("WR", "starter", "rec_yards") == pytest.approx(0.4)
    # transient dispersion is directly observed, so calibration leaves it alone
    assert twice.k_usage("WR", "targets") == d.k_usage("WR", "targets")
    assert twice.k_team_week("targets") == d.k_team_week("targets")


def test_the_calibration_statistic_is_a_distance_from_uniform() -> None:
    """What the scale is fitted on, so it is worth pinning: uniform scores low, piled-up scores high."""
    rng = np.random.default_rng(0)
    flat = simulate._cvm(rng.random(2000))
    low = simulate._cvm(rng.random(2000) * 0.3)               # every outcome under its own median
    narrow = simulate._cvm(np.clip(rng.normal(0.5, 0.05, 2000), 0, 1))   # intervals far too wide
    assert flat < 0.5                                         # 1/6 in expectation under uniformity
    assert low > 20 * flat
    assert narrow > 20 * flat
    assert np.isnan(simulate._cvm(np.array([0.5])))


def test_a_tie_in_the_draws_counts_as_half_so_a_touchdown_count_is_not_read_as_a_bias() -> None:
    """The mid-PIT correction. A discrete outcome sits *inside* its own atom, not at the bottom of it."""
    draws = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (1, 1))   # one player, four equally likely values
    assert simulate._pit(draws, np.array([0.0]))[0] == pytest.approx(0.125)
    assert simulate._pit(draws, np.array([3.0]))[0] == pytest.approx(0.875)
    assert simulate._pit(draws, np.array([1.5]))[0] == pytest.approx(0.5)


def test_coverage_counts_the_outcomes_that_landed_inside_the_interval() -> None:
    """Against a made-up set of actuals, so the arithmetic is checkable rather than merely plausible."""
    wk = _weekly(weeks=17)
    sim = simulate.run(wk, SETTINGS, _disp(share=0.3, usage=0.3, event=1.0), draws=2000)
    ids = sim.season["player_id"].to_list()
    # every actual placed exactly at that player's own median: full 50% coverage, no bias
    actual = pl.DataFrame({"player_id": ids,
                           "a_fantasy_points": sim.season["p50"].to_list(),
                           "played": [True] * len(ids)})
    got = simulate.coverage(sim, actual, min_n=4).filter(pl.col("position") == "ALL") \
        .row(0, named=True)
    assert got["cover_90"] == pytest.approx(1.0)
    assert got["cover_50"] == pytest.approx(1.0)
    assert got["pit_mean"] == pytest.approx(0.5, abs=0.02)
    assert got["median_bias"] == pytest.approx(0.0, abs=0.5)

    # and every actual above the 95th: nothing covered, all of it above
    high = actual.with_columns(pl.Series("a_fantasy_points", sim.season["p95"].to_list()) * 1.5)
    got = simulate.coverage(sim, high, min_n=4).filter(pl.col("position") == "ALL").row(0, named=True)
    assert got["cover_90"] == pytest.approx(0.0)
    assert got["above"] == pytest.approx(1.0)
    assert got["median_bias"] > 0


def test_the_width_criterion_ignores_a_shift_and_the_shape_criterion_does_not() -> None:
    """Why the scale is fitted on the PIT's spread rather than on its distance from uniform.

    Both statistics are needed and they answer different questions. Slide a perfectly-spread PIT
    sideways -- a pure bias in the median, nothing wrong with the width -- and `_cvm` degrades sharply,
    which is what makes it reach for a wider sigma as the remedy. `_pit_iqr` does not move, so the
    location error survives to be reported instead of absorbed.
    """
    n = 2000
    flat = (np.arange(n) + 0.5) / n
    shifted = np.clip(flat + 0.15, 0.0, 1.0)
    assert simulate._pit_iqr(flat) == pytest.approx(0.5, abs=0.01)
    assert simulate._pit_iqr(shifted) == pytest.approx(0.5, abs=0.01)
    assert simulate._cvm(shifted) > 20 * simulate._cvm(flat)

    # and it still fails a width that is genuinely wrong, in both directions
    too_narrow = np.where(flat < 0.5, 0.02, 0.98)          # everything in the tails
    too_wide = 0.5 + (flat - 0.5) * 0.1                    # everything piled on the median
    assert simulate._pit_iqr(too_narrow) > 0.9
    assert simulate._pit_iqr(too_wide) < 0.1
    assert np.isnan(simulate._pit_iqr(np.array([0.4, 0.6])))


def test_the_coverage_population_is_cut_on_the_projection_and_not_on_the_outcome() -> None:
    """Selecting on the outcome is the way to make a range look calibrated when it is not.

    `played` keeps only the seasons that beat the `p_play` the projection applied, so it drops the left
    tail the model predicted and reads the survivors as an upside the model missed. Shown here with the
    survivorship dialled to the extreme -- every outcome under its own median discarded -- because that
    is the same operation `played` performs, only more of it.
    """
    wk = _weekly(weeks=17)
    sim = simulate.run(wk, SETTINGS, _disp(share=0.3, usage=0.3, event=1.0), draws=2000)
    ids = sim.season["player_id"].to_list()
    actual = pl.DataFrame({"player_id": ids, "a_fantasy_points": sim.season["p50"].to_list(),
                           "played": [True] * len(ids)})

    cut = sim.season["projected"].median()
    trimmed = simulate.coverage(sim, actual, min_projected=cut, min_n=1) \
        .filter(pl.col("position") == "ALL").row(0, named=True)
    whole = simulate.coverage(sim, actual, min_n=1).filter(pl.col("position") == "ALL") \
        .row(0, named=True)
    assert 0 < trimmed["n"] < whole["n"]
    assert trimmed["pit_mean"] == pytest.approx(whole["pit_mean"], abs=0.05)   # a cut, not a shift

    # now select on the outcome instead, and watch the PIT move without the model changing at all
    survivors = actual.with_columns(
        pl.Series("a_fantasy_points", sim.season["p95"].to_list()).alias("a_fantasy_points"))
    biased = simulate.coverage(sim, survivors, min_n=1).filter(pl.col("position") == "ALL") \
        .row(0, named=True)
    assert biased["pit_mean"] > whole["pit_mean"] + 0.3
    assert biased["below"] < whole["below"] + 1e-9
