"""The adjustment layer: an edit does what it says, says what it did, and can be taken back.

The workbook's `BASE / x Adj / USED` triplet was trustworthy because all three numbers were on screen
at once. The equivalent guarantees here are the ones worth testing: an edit is applied where it
belongs, reported whether or not it found anything, still subject to the pool it is a share of, and
removable one knob at a time without disturbing the others.

Every test below is either an invariant of the data model or a football fact -- a share taken from one
receiver is a share his teammates lose -- so a data refresh cannot break them without something
genuinely being wrong. The full-projection tests are module-scoped fixtures: three runs, not thirty.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.config import PROJ_SEASON, REG_WEEKS, Settings
from src.model import opportunity, overrides
from src.model.overrides import Override, Scenario


# --------------------------------------------------------------------------- #
# the data model
# --------------------------------------------------------------------------- #
def test_an_edit_has_to_be_editable() -> None:
    with pytest.raises(ValueError, match="not editable"):
        Scenario().set(Override("player", "x", "fantasy_points", "set", 400.0))
    with pytest.raises(ValueError, match="level must be"):
        Override("universe", "x", "target_share")
    with pytest.raises(ValueError, match="mode must be"):
        Override("player", "x", "target_share", "increase")


def test_setting_the_same_knob_twice_replaces_rather_than_stacks() -> None:
    sc = Scenario().set(
        Override("player", "p1", "target_share", "set", 0.20),
        Override("player", "p1", "target_share", "set", 0.30),
    )
    assert len(sc.items) == 1
    assert sc.items[0].value == 0.30


def test_a_week_makes_it_a_different_knob() -> None:
    sc = Scenario().set(
        Override("team", "PHI", "targets", "multiply", 1.1),
        Override("team", "PHI", "targets", "multiply", 1.2, week=3),
    )
    assert len(sc.items) == 2


def test_a_reset_takes_one_knob_and_leaves_the_rest() -> None:
    sc = Scenario().set(
        Override("player", "p1", "target_share", "set", 0.30),
        Override("player", "p1", "catch_rate", "set", 0.70),
        Override("player", "p2", "target_share", "set", 0.10),
    )
    left = sc.clear(level="player", key="p1", field_name="target_share")
    assert {(o.key, o.field) for o in left.items} == {("p1", "catch_rate"), ("p2", "target_share")}
    assert {(o.key, o.field) for o in left.clear(level="player", key="p1").items} == \
        {("p2", "target_share")}
    assert sc.clear().items == ()


def test_a_scenario_survives_a_round_trip_through_json() -> None:
    sc = Scenario(name="mine").set(
        Override("player", "p1", "target_share", "set", 0.30, base=0.24, note="alpha"),
        Override("team", "PHI", "carries", "multiply", 0.9, week=5),
    ).patch_league(normalize_pools=False, k_scale=0.0)
    back = Scenario.from_json(sc.to_json())
    assert back == sc
    assert json.loads(sc.to_json())["overrides"][0]["base"] == 0.24


def test_the_digest_is_the_content_and_not_the_name_or_the_clock() -> None:
    sc = Scenario(name="a").set(Override("player", "p1", "target_share", "set", 0.30))
    renamed = sc.rename("b")
    assert renamed.digest == sc.digest
    assert renamed.name == "b"
    # the note and the recorded base are annotation, not content: they cannot cost a cached run
    annotated = Scenario(name="a").set(
        Override("player", "p1", "target_share", "set", 0.30, base=0.19, note="changed my mind")
    )
    assert annotated.digest == sc.digest
    assert Scenario().digest != sc.digest


def test_the_order_edits_were_made_in_does_not_change_the_digest() -> None:
    a = Scenario().set(Override("player", "p1", "target_share", "set", 0.30),
                       Override("team", "PHI", "carries", "multiply", 1.1))
    b = Scenario().set(Override("team", "PHI", "carries", "multiply", 1.1),
                       Override("player", "p1", "target_share", "set", 0.30))
    assert a.digest == b.digest


def test_a_league_knob_at_its_default_is_not_an_edit() -> None:
    base = Settings()
    same = Scenario().patch_league(normalize_pools=base.normalize_pools,
                                   use_context_factors=base.use_context_factors,
                                   market_weight=base.market_weight,
                                   scoring=overrides.DEFAULT_SCORING)
    assert same.is_baseline
    assert same.digest == Scenario().digest
    off = Scenario().patch_league(normalize_pools=False)
    assert not off.is_baseline
    assert off.patch_league(normalize_pools=True).is_baseline


def test_recency_comes_back_from_json_as_the_tuple_the_engine_expects() -> None:
    sc = Scenario.from_json(Scenario().patch_league(recency=(4.0, 2.0, 1.0)).to_json())
    assert overrides.settings_for(sc).recency == (4.0, 2.0, 1.0)


def test_the_league_patch_reaches_settings_and_nothing_else_does() -> None:
    sc = Scenario().patch_league(normalize_pools=False, context_k=50.0, scoring="half_ppr")
    st = overrides.settings_for(sc)
    assert st.normalize_pools is False
    assert st.context_k == 50.0
    assert st.scoring.reception == 0.5
    assert overrides.settings_for(Scenario()) == Settings()
    with pytest.raises(ValueError, match="not league-editable"):
        Scenario().patch_league(status_availability={})


def test_k_scale_moves_every_shrinkage_constant_and_leaves_one_untouched_alone() -> None:
    from src.model import priors

    fitted = priors.fitted_saved()
    doubled = overrides.fitted_for(Scenario().patch_league(k_scale=2.0), fitted)
    assert overrides.fitted_for(Scenario(), fitted) is fitted
    # every metric, so none of them silently falls back to the Settings default and escapes the scale
    assert set(doubled.k) == {m.name for m in priors.METRICS}
    for name in doubled.k:
        if name in fitted.k:
            assert doubled.k[name] == pytest.approx(2.0 * fitted.k[name])
    assert all(k == 0.0 for k in overrides.fitted_for(Scenario().patch_league(k_scale=0.0),
                                                      fitted).k.values())


# --------------------------------------------------------------------------- #
# applying one edit to one frame
# --------------------------------------------------------------------------- #
@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame({
        "team": ["PHI", "PHI", "DAL"],
        "week": [1, 2, 1],
        "targets": [34.0, 30.0, 32.0],
    })


def test_set_and_multiply_mean_what_they_say(frame) -> None:
    sc = Scenario().set(Override("team", "PHI", "targets", "set", 40.0))
    out, log = overrides.apply(frame, sc, "team", "team", week_col="week")
    assert out["targets"].to_list() == [40.0, 40.0, 32.0]
    assert log[0]["rows"] == 2 and log[0]["applied"] is True

    sc = Scenario().set(Override("team", "PHI", "targets", "multiply", 0.5))
    out, _ = overrides.apply(frame, sc, "team", "team", week_col="week")
    assert out["targets"].to_list() == [17.0, 15.0, 32.0]


def test_a_week_edits_that_week_only(frame) -> None:
    sc = Scenario().set(Override("team", "PHI", "targets", "set", 20.0, week=2))
    out, log = overrides.apply(frame, sc, "team", "team", week_col="week")
    assert out["targets"].to_list() == [34.0, 20.0, 32.0]
    assert log[0]["rows"] == 1


def test_provenance_records_what_it_was_and_what_it_became(frame) -> None:
    sc = Scenario().set(Override("team", "PHI", "targets", "multiply", 2.0, base=31.0))
    _, log = overrides.apply(frame, sc, "team", "team", week_col="week")
    rec = log[0]
    assert rec["base_recorded"] == 31.0                 # what it was worth when the edit was made
    assert rec["base_now"] == pytest.approx(32.0)       # what it was worth when it ran
    assert rec["used_now"] == pytest.approx(64.0)


def test_an_edit_that_finds_nothing_says_so_instead_of_vanishing(frame) -> None:
    sc = Scenario().set(
        Override("team", "NOPE", "targets", "set", 40.0),
        Override("team", "PHI", "air_yards", "set", 300.0),
        Override("team", "PHI", "carries", "set", 20.0, week=99),
    )
    out, log = overrides.apply(frame, sc, "team", "team", week_col="week")
    assert out.equals(frame)
    assert len(log) == 3
    assert not any(r["applied"] for r in log)
    assert [r["reason"] for r in log] == ["no rows for NOPE", "no such column in this frame",
                                         "no such column in this frame"]


def test_a_frame_without_weeks_reports_a_weekly_edit_rather_than_applying_it_everywhere() -> None:
    frame = pl.DataFrame({"team": ["PHI"], "targets": [34.0]})
    sc = Scenario().set(Override("team", "PHI", "targets", "set", 40.0, week=3))
    out, log = overrides.apply(frame, sc, "team", "team")
    assert out.equals(frame)
    assert log[0]["reason"] == "this frame has no weeks to target"


def test_fields_narrows_an_edit_to_the_frame_that_owns_it(frame) -> None:
    sc = Scenario().set(Override("team", "PHI", "targets", "set", 40.0))
    out, log = overrides.apply(frame, sc, "team", "team", week_col="week", fields=("carries",))
    assert out.equals(frame)
    assert log == []


def test_availability_is_capped_at_the_season_and_active_weeks_follows_it() -> None:
    part = pl.DataFrame({"expected_games": [8.0, 25.0], "active_weeks": [0.9, 0.9]})
    out = overrides._rederive_availability(part)
    assert out["expected_games"].to_list() == [8.0, float(REG_WEEKS - 1)]
    assert out["active_weeks"].to_list() == [pytest.approx(8.0 / 17.0), 1.0]


# --------------------------------------------------------------------------- #
# a whole projection under a scenario
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def base() -> overrides.Run:
    return overrides.run(Scenario(), PROJ_SEASON)


@pytest.fixture(scope="module")
def target(base) -> dict:
    """The receiver the edits are about: the top projected one, so his room is a real receiving room."""
    return base.board.filter(pl.col("position") == "WR").row(0, named=True)


@pytest.fixture(scope="module")
def edited(target) -> overrides.Run:
    return overrides.run(
        Scenario(name="t").set(
            Override("player", target["player_id"], "target_share", "set", 0.32),
            Override("player", target["player_id"], "expected_games", "set", 17.0),
        ),
        PROJ_SEASON,
    )


def test_the_baseline_run_is_the_engines_own_answer(base) -> None:
    from src.model import compose

    board = compose.board(compose.seasonal(compose.weekly(PROJ_SEASON, Settings()), PROJ_SEASON,
                                           Settings()), Settings())
    mine = base.board.select("player_id", "fantasy_points").sort("player_id")
    theirs = board.select("player_id", "fantasy_points").sort("player_id")
    assert mine.height == theirs.height
    assert (mine["fantasy_points"] - theirs["fantasy_points"]).abs().max() < 1e-9
    assert base.provenance.is_empty()
    assert base.digest == Scenario().digest


def test_every_edit_is_reported_against_the_frame_it_landed_on(edited, target) -> None:
    prov = edited.provenance
    assert prov.filter(pl.col("applied")).height >= 2
    stages = set(prov.filter(pl.col("applied"))["stage"].to_list())
    assert {"shares", "availability"} <= stages
    assert set(prov["key"].unique().to_list()) == {target["player_id"]}


def test_a_share_edit_is_the_share_the_projection_used(edited, target) -> None:
    row = edited.shares.filter(pl.col("player_id") == target["player_id"]).row(0, named=True)
    assert row["target_share"] == pytest.approx(0.32)


def test_a_games_edit_carries_into_the_games_projected(edited, target) -> None:
    part = edited.part.filter(pl.col("player_id") == target["player_id"]).row(0, named=True)
    assert part["expected_games"] == pytest.approx(17.0)
    assert part["active_weeks"] == pytest.approx(1.0)
    got = edited.board.filter(pl.col("player_id") == target["player_id"]).row(0, named=True)
    assert got["games"] == pytest.approx(17.0, abs=0.02)


def test_normalisation_takes_from_the_teammates_what_it_gives_the_player(base, edited, target) -> None:
    """The whole reason to edit inside the engine rather than in a spreadsheet.

    A team's targets are a fixed quantity, so a share handed to one receiver has to come from the
    others. `net` near zero with a large `abs` is that fact showing up in the diff.
    """
    moved = overrides.diff(base.board, edited.board).filter(pl.col("team") == target["team"])
    him = moved.filter(pl.col("player_id") == target["player_id"]).row(0, named=True)
    others = moved.filter(pl.col("player_id") != target["player_id"])
    assert him["d_targets"] > 0.0
    assert others.height > 0
    assert others["d_targets"].sum() < 0.0
    per_game = edited.opp.filter(pl.col("week") == edited.opp["week"].min())
    team_sum = (
        per_game.filter(pl.col("team") == target["team"])["share_targets"].sum()
    )
    assert team_sum == pytest.approx(opportunity.measure_targets()["targets"], abs=1e-6)


def test_an_edit_moves_nobody_on_another_team(base, edited, target) -> None:
    moved = overrides.diff(base.board, edited.board)
    assert set(moved["team"].unique().to_list()) == {target["team"]}


def test_with_normalisation_off_the_teammates_keep_what_they_had(base, target) -> None:
    off = Scenario().patch_league(normalize_pools=False)
    without = overrides.run(off, PROJ_SEASON)
    with_edit = overrides.run(
        off.set(Override("player", target["player_id"], "target_share", "set", 0.32)), PROJ_SEASON
    )
    moved = overrides.diff(without.board, with_edit.board)
    assert moved.height == 1
    assert moved["player_id"][0] == target["player_id"]


def test_the_diff_hides_rounding_and_nothing_else(base) -> None:
    assert overrides.diff(base.board, base.board).is_empty()
    assert overrides.summary(base.board, base.board).is_empty()


def test_the_summary_counts_players_who_moved_rather_than_players(base, edited, target) -> None:
    got = overrides.summary(base.board, edited.board)
    moved = overrides.diff(base.board, edited.board)
    assert got["players_moved"].sum() == moved.height
    assert got["players_moved"].sum() < base.board.height
    wr = got.filter(pl.col("position") == "WR").row(0, named=True)
    assert wr["abs_points_moved"] >= abs(wr["net_points_moved"])
    assert wr["biggest_gain"] > 0.0 > wr["biggest_loss"]


def test_a_stale_edit_is_reported_and_changes_nothing(base) -> None:
    sc = Scenario().set(Override("player", "00-9999999", "target_share", "set", 0.40))
    got = overrides.run(sc, PROJ_SEASON)
    assert not got.provenance["applied"].any()
    assert overrides.diff(base.board, got.board).is_empty()


# --------------------------------------------------------------------------- #
# scenarios on disk
# --------------------------------------------------------------------------- #
def test_a_saved_scenario_comes_back_the_same(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    sc = Scenario(name="my rankings").set(
        Override("player", "p1", "target_share", "set", 0.30, base=0.24)
    ).patch_league(k_scale=0.5)
    overrides.save(sc)
    back = overrides.load("my rankings")
    assert back.items == sc.items
    assert back.k_scale == 0.5
    assert back.digest == sc.digest
    assert back.created and back.updated          # provenance: when it was written
    assert overrides.names() == ["my rankings"]
    assert overrides.delete("my rankings") is True
    assert overrides.names() == []
    assert overrides.delete("my rankings") is False


def test_a_name_that_is_not_a_filename_still_saves(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    overrides.save(Scenario(name="12-team / half PPR: *mine*"))
    assert overrides.names() == ["12-team / half PPR: *mine*"]
    assert overrides.load("12-team / half PPR: *mine*").is_baseline


def test_loading_a_scenario_that_is_not_there_is_empty_rather_than_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    got = overrides.load("nope")
    assert got.name == "nope"
    assert got.is_baseline
