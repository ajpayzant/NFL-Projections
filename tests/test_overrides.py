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


def test_un_overriding_a_group_of_players_takes_all_of_their_edits_and_nobody_else_s() -> None:
    """The other end of a bulk edit. Overriding selectively is only usable if un-overriding is too."""
    sc = Scenario().set(
        Override("player", "p1", "target_share", "set", 0.30),
        Override("player", "p1", "target_share", "set", 0.25, week=4),
        Override("player", "p1", "catch_rate", "set", 0.70),
        Override("player", "p2", "target_share", "set", 0.10),
        Override("player", "p3", "carry_share", "set", 0.40),
        Override("team", "PHI", "targets", "multiply", 1.1),
    )
    left = sc.clear_keys("player", ["p1", "p3"])
    assert {(o.level, o.key) for o in left.items} == {("player", "p2"), ("team", "PHI")}
    # a team of the same name is a different key: the level is part of the question
    assert sc.clear_keys("team", ["p1"]).items == sc.items


def test_un_overriding_nobody_leaves_the_scenario_identical() -> None:
    """No new `updated`, so the digest does not move and the app recomputes nothing."""
    sc = Scenario().set(Override("player", "p1", "target_share", "set", 0.30))
    assert sc.clear_keys("player", []) is sc
    assert sc.clear_keys("player", ["someone_else"]) is sc


def test_the_edit_count_per_player_is_what_a_drop_all_button_promises() -> None:
    sc = Scenario().set(
        Override("player", "p1", "target_share", "set", 0.30),
        Override("player", "p1", "target_share", "set", 0.25, week=4),
        Override("player", "p2", "target_share", "set", 0.10),
        Override("team", "PHI", "targets", "multiply", 1.1),
    )
    assert sc.counts("player") == {"p1": 2, "p2": 1}
    assert sc.counts("team") == {"PHI": 1}
    for key, n in sc.counts("player").items():
        assert len(sc.clear_keys("player", [key]).items) == len(sc.items) - n


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
        Scenario().patch_league(simulation_draws=10)


def test_a_status_factor_is_patched_into_the_stated_set_and_not_over_it() -> None:
    """The knob that replaces two hundred hand edits, and the merge that stops it costing nine others.

    A scenario that disagrees about the practice squad should record the practice squad and nothing
    else: stored whole, patching `DEV` would drop `CUT` and `RET` from the dict, and `availability`
    would hand both of them its `default=1.0` fallback -- so saying "the practice squad does not play
    here" would also have said "released players do", which is the opposite of the intent and would
    have shown up as a hundred and fifty cut men back on the board with a full season each.
    """
    base = Settings().status_availability
    sc = Scenario().patch_league(status_availability={"DEV": 0.0})
    # recorded as the one disagreement, so the scenario reads as one disagreement
    assert sc.league["status_availability"] == {"DEV": 0.0}
    got = overrides.settings_for(sc).status_availability
    assert got["DEV"] == 0.0
    assert {k: v for k, v in got.items() if k != "DEV"} == {k: v for k, v in base.items()
                                                            if k != "DEV"}
    # a second disagreement joins the first rather than replacing it
    both = sc.patch_league(status_availability={**sc.league["status_availability"], "RES": 0.0})
    assert overrides.settings_for(both).status_availability["DEV"] == 0.0
    assert overrides.settings_for(both).status_availability["RES"] == 0.0
    # set back to what it always was, it is not an edit -- the baseline has to stay the baseline
    assert sc.patch_league(status_availability=dict(base)).is_baseline
    # and it survives the JSON round trip a saved scenario is
    again = Scenario.from_json(both.to_json())
    assert overrides.settings_for(again).status_availability == \
        overrides.settings_for(both).status_availability
    assert again.digest == both.digest


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


def test_an_edit_on_a_cell_with_no_number_in_it_is_reported_rather_than_written() -> None:
    """A completion percentage on a running back: the row exists, the number does not.

    Every editing surface filters these out before writing one, but a saved scenario outlives the
    roster it was written against -- a receiver who has since become the emergency quarterback, or a
    metric a refresh no longer estimates for him -- so the engine has to answer for it rather than
    raise. It used to reach `float(None)` and take the whole projection down.
    """
    frame = pl.DataFrame({"player_id": ["p1", "p2"], "completion_pct": [None, 0.64]},
                         schema={"player_id": pl.String, "completion_pct": pl.Float64})
    sc = Scenario().set(Override("player", "p1", "completion_pct", "set", 0.68))
    out, log = overrides.apply(frame, sc, "player", "player_id")
    assert out.equals(frame)
    assert not log[0]["applied"]
    assert log[0]["reason"] == "no number for this player to change"


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
# playing time
# --------------------------------------------------------------------------- #
# The one knob that is not a share of anything the engine divides. Nothing splits the snap pool -- five men
# are on the field for the same snap -- so by the rule that governs every other share a snap share would be
# evidence, and it was: a reader could type one and no projected number moved. It is spent instead as a
# multiplier on every *other* claim the man makes, because that is what a snap share means: more of the
# game, and therefore more of everything he is out there to claim. The exclusive pools then settle, so the
# targets and carries he gains come off the teammates he is on the field instead of.
#
# Four things have to hold for that to be trustworthy, and each is a test below: the multiplier is exactly
# one until somebody types (or the whole board would drift); typing the estimated value is a no-op; the edit
# reaches the stat line; and the pool it lands in still ties out to its measured target afterwards.
@pytest.fixture(scope="module")
def snaps(base) -> dict:
    """A second receiver: enough snaps to have a rate, and enough room left to raise it into."""
    men = base.shares.join(
        base.roster.select("player_id", "player", "position", "team", "depth_slot"), on="player_id"
    )
    mine = men.filter(
        (pl.col("position") == "WR") & (pl.col("depth_slot") == 2) & (pl.col("snap_share") > 0.35)
    )
    if mine.is_empty():
        pytest.skip("no second receiver with a real snap share")
    return mine.sort("snap_share").row(0, named=True)


def test_playing_time_is_exactly_one_until_somebody_types(base) -> None:
    col = base.shares[overrides.PLAYING_TIME]
    assert col.null_count() == 0
    assert col.min() == 1.0 and col.max() == 1.0


def test_typing_the_estimated_snap_share_changes_nothing(base, snaps) -> None:
    """The multiplier is a ratio to the model's own number, so the identity edit has to be the identity.

    Not a formality: the measured share already contains his measured playing time, so if this drifted
    then every unedited player would be projected off a number nobody chose.
    """
    same = overrides.run(
        Scenario(name="same").set(
            Override("player", snaps["player_id"], "snap_share", "set", float(snaps["snap_share"]))),
        PROJ_SEASON,
    )
    assert same.shares.filter(pl.col("player_id") == snaps["player_id"])[
        overrides.PLAYING_TIME][0] == pytest.approx(1.0)
    assert overrides.diff(base.board, same.board).is_empty()


@pytest.fixture(scope="module")
def more_snaps(snaps) -> overrides.Run:
    return overrides.run(
        Scenario(name="snaps").set(
            Override("player", snaps["player_id"], "snap_share", "multiply", 1.25)),
        PROJ_SEASON,
    )


def test_more_snaps_is_more_of_everything_he_claims(base, snaps, more_snaps) -> None:
    pid = snaps["player_id"]
    assert more_snaps.shares.filter(pl.col("player_id") == pid)[
        overrides.PLAYING_TIME][0] == pytest.approx(1.25)
    was = base.board.filter(pl.col("player_id") == pid).row(0, named=True)
    now = more_snaps.board.filter(pl.col("player_id") == pid).row(0, named=True)
    # the snaps themselves, which is the share he typed
    assert now["offense_snaps"] > was["offense_snaps"] * 1.2
    # and the claims he makes in them, which is the point: a share he never touched moved with his snaps
    for stat in ("targets", "receptions", "receiving_yards", "fantasy_points"):
        assert now[stat] > was[stat] * 1.1, stat


def test_what_he_gains_his_room_gives_up(base, snaps, more_snaps) -> None:
    """The edit is conserved, which is what makes it a projection rather than a wish."""
    moved = overrides.diff(base.board, more_snaps.board).filter(pl.col("team") == snaps["team"])
    him = moved.filter(pl.col("player_id") == snaps["player_id"]).row(0, named=True)
    others = moved.filter(pl.col("player_id") != snaps["player_id"])
    assert him["d_targets"] > 0.0
    assert others.height > 0
    assert others["d_targets"].sum() == pytest.approx(-him["d_targets"], rel=0.02)
    week = more_snaps.opp.filter(pl.col("week") == more_snaps.opp["week"].min())
    team_sum = week.filter(pl.col("team") == snaps["team"])["share_targets"].sum()
    assert team_sum == pytest.approx(opportunity.measure_targets()["targets"], abs=1e-6)


def test_a_share_he_typed_himself_is_not_scaled_by_his_snaps(base, snaps) -> None:
    """`lock_edited_shares` means an override is the number used. Two edits do not multiply into one.

    Otherwise a reader who stated both a snap share and a target share would get neither: the engine
    would take the target share he asked for and then quietly scale it by the other thing he asked for.
    """
    pid = snaps["player_id"]
    stated = float(base.shares.filter(pl.col("player_id") == pid)["target_share"][0])
    both = overrides.run(
        Scenario(name="both").set(
            Override("player", pid, "snap_share", "multiply", 1.3),
            Override("player", pid, "target_share", "set", stated),
        ),
        PROJ_SEASON,
    )
    assert both.shares.filter(pl.col("player_id") == pid)["target_share"][0] == pytest.approx(stated)
    was = base.board.filter(pl.col("player_id") == pid).row(0, named=True)
    now = both.board.filter(pl.col("player_id") == pid).row(0, named=True)
    assert now["targets"] == pytest.approx(was["targets"], rel=0.02)   # the number he stated, unscaled
    assert now["offense_snaps"] > was["offense_snaps"] * 1.2           # the snaps he stated, honoured


def test_a_quarterbacks_playing_time_reaches_the_queue(base) -> None:
    """The dropback pool is filled in depth order rather than by tilt, so it needs its own check."""
    qbs = base.shares.join(base.roster.select("player_id", "position", "team", "depth_slot"),
                           on="player_id").filter(
        (pl.col("position") == "QB") & (pl.col("depth_slot") == 2) & (pl.col("qb_snap_share") > 0.3))
    if qbs.is_empty():
        pytest.skip("no backup quarterback with a projected snap share")
    him = qbs.row(0, named=True)
    less = overrides.run(
        Scenario(name="bench").set(
            Override("player", him["player_id"], "qb_snap_share", "multiply", 0.5)),
        PROJ_SEASON,
    )
    was = base.board.filter(pl.col("player_id") == him["player_id"]).row(0, named=True)
    now = less.board.filter(pl.col("player_id") == him["player_id"]).row(0, named=True)
    assert now["attempts"] < was["attempts"] * 0.75
    assert now["passing_yards"] < was["passing_yards"] * 0.75
    # the team still throws the ball the same number of times: the man ahead of him takes it back
    mine = base.board.filter(pl.col("team") == him["team"])["attempts"].sum()
    theirs = less.board.filter(pl.col("team") == him["team"])["attempts"].sum()
    assert theirs == pytest.approx(mine, rel=0.01)


def test_playing_time_cannot_be_pushed_past_the_cap(base, snaps) -> None:
    """A man cannot be on the field three times as much as he was measured at, whatever is typed."""
    silly = overrides.run(
        Scenario(name="silly").set(
            Override("player", snaps["player_id"], "snap_share", "multiply", 50.0)),
        PROJ_SEASON,
    )
    assert silly.shares.filter(pl.col("player_id") == snaps["player_id"])[
        overrides.PLAYING_TIME][0] == pytest.approx(overrides.PLAYING_TIME_CAP)


# --------------------------------------------------------------------------- #
# one game rather than the season
# --------------------------------------------------------------------------- #
# A per-game edit is the same edit as a season one at a different grain, so what is tested here is that
# the grain is real: the week it names moves, the other sixteen do not, the season total follows because
# it is their sum, and the team's books still balance in the week that was edited.
WEEK = 5


@pytest.fixture(scope="module")
def busy(base) -> dict:
    """The most-used back in week 5, so his absence is a hole somebody has to fill."""
    wk = base.weekly.filter((pl.col("week") == WEEK) & (pl.col("position") == "RB"))
    return wk.sort("carries", descending=True).row(0, named=True)


@pytest.fixture(scope="module")
def benched(busy) -> overrides.Run:
    """He does not play in week 5. The one thing a season-long games count cannot say."""
    return overrides.run(
        Scenario(name="out").set(
            Override("player", busy["player_id"], "p_play", "set", 0.0, week=WEEK)),
        PROJ_SEASON,
    )


def test_a_per_game_availability_edit_needs_the_week_it_is_about() -> None:
    with pytest.raises(ValueError, match="needs a week"):
        Scenario().set(Override("player", "x", "p_play", "set", 0.0))


def test_a_per_game_edit_is_applied_at_the_game_stage(benched, busy) -> None:
    prov = benched.provenance
    assert prov.height == 1, "one edit, one record -- no stage may claim it twice"
    row = prov.row(0, named=True)
    assert (row["stage"], row["field"], row["week"], row["applied"]) == ("game", "p_play", WEEK, True)
    assert row["rows"] == 1, "one player-game, not seventeen"
    assert row["base_now"] > 0.0 and row["used_now"] == 0.0


def test_the_week_named_is_the_only_week_that_moves(base, benched, busy) -> None:
    def his(run, week):
        f = run.weekly.filter((pl.col("player_id") == busy["player_id"]) & (pl.col("week") == week))
        return float(f["carries"][0])

    assert his(benched, WEEK) == pytest.approx(0.0)
    others = [w for w in base.weekly["week"].unique().to_list() if w != WEEK]
    for week in others:
        if base.weekly.filter((pl.col("player_id") == busy["player_id"])
                              & (pl.col("week") == week)).is_empty():
            continue          # his bye
        assert his(benched, week) == pytest.approx(his(base, week), abs=1e-9)


def test_the_room_takes_the_work_he_is_not_there_for(base, benched, busy) -> None:
    """The reason the edit is applied before the pool is divided rather than after.

    A team's carries in a game are a fixed quantity. Removing one back does not remove the carries; it
    hands them to whoever else is on the field, which is what the projection has to say too.
    """
    def team_carries(run):
        f = run.weekly.filter((pl.col("team") == busy["team"]) & (pl.col("week") == WEEK))
        return float(f["carries"].sum())

    assert team_carries(benched) == pytest.approx(team_carries(base), rel=1e-6)

    def mate_carries(run):
        f = run.weekly.filter((pl.col("team") == busy["team"]) & (pl.col("week") == WEEK)
                              & (pl.col("player_id") != busy["player_id"]))
        return float(f["carries"].sum())

    assert mate_carries(benched) > mate_carries(base)


def test_the_season_total_is_the_sum_of_the_games_including_the_edited_one(base, benched, busy) -> None:
    was, now = (r.season_frame.filter(pl.col("player_id") == busy["player_id"]).row(0, named=True)
                for r in (base, benched))
    assert now["carries"] < was["carries"]
    assert now["games"] == pytest.approx(was["games"] - base_p_play(base, busy), abs=1e-6)
    weeks = benched.weekly.filter(pl.col("player_id") == busy["player_id"])
    assert now["carries"] == pytest.approx(float(weeks["carries"].sum()), rel=1e-9)


def base_p_play(run: overrides.Run, who: dict) -> float:
    f = run.weekly.filter((pl.col("player_id") == who["player_id"]) & (pl.col("week") == WEEK))
    return float(f["p_play"][0])


def test_a_per_game_rate_edit_moves_that_game_and_leaves_the_rest(base, busy) -> None:
    got = overrides.run(
        Scenario().set(Override("player", busy["player_id"], "yards_per_carry", "set", 9.0,
                                week=WEEK)),
        PROJ_SEASON,
    )
    row = got.provenance.row(0, named=True)
    assert (row["stage"], row["applied"], row["used_now"]) == ("game rates", True, 9.0)

    def yards(run, week):
        f = run.weekly.filter((pl.col("player_id") == busy["player_id"]) & (pl.col("week") == week))
        return float(f["rushing_yards"][0])

    # the game factor still applies on top, so the answer is 9.0 x that factor rather than exactly 9.0
    carries = float(got.weekly.filter((pl.col("player_id") == busy["player_id"])
                                      & (pl.col("week") == WEEK))["carries"][0])
    assert yards(got, WEEK) / carries == pytest.approx(9.0, rel=0.30)
    assert yards(got, WEEK) > yards(base, WEEK)
    weeks = [w for w in (WEEK + 1, WEEK - 1) if w >= 1]
    for week in weeks:
        if base.weekly.filter((pl.col("player_id") == busy["player_id"])
                              & (pl.col("week") == week)).is_empty():
            continue
        assert yards(got, week) == pytest.approx(yards(base, week), abs=1e-9)


def test_a_season_edit_and_a_per_game_edit_on_one_field_are_both_kept(busy) -> None:
    """Two grains of the same claim: 24% of the targets all year, 40% of them in week 5."""
    sc = Scenario().set(
        Override("player", busy["player_id"], "target_share", "set", 0.24),
        Override("player", busy["player_id"], "target_share", "set", 0.40, week=WEEK),
    )
    assert len(sc.items) == 2
    got = overrides.run(sc, PROJ_SEASON)
    prov = got.provenance.filter(pl.col("applied"))
    assert set(prov["stage"].to_list()) == {"shares", "game"}
    per_week = got.opp.filter(pl.col("player_id") == busy["player_id"]).sort("week")
    edited = per_week.filter(pl.col("week") == WEEK)["target_share"][0]
    rest = per_week.filter(pl.col("week") != WEEK)["target_share"]
    assert float(edited) == pytest.approx(0.40)
    assert rest.max() == pytest.approx(0.24) and rest.min() == pytest.approx(0.24)


def test_a_week_on_a_season_grain_field_is_reported_by_exactly_one_stage(busy) -> None:
    """A chart is not a per-game thing. Saying so once is the requirement; twice is a log arguing."""
    got = overrides.run(
        Scenario().set(Override("player", busy["player_id"], "depth_slot", "set", 1.0, week=WEEK)),
        PROJ_SEASON,
    )
    assert got.provenance.height == 1
    row = got.provenance.row(0, named=True)
    assert (row["stage"], row["applied"]) == ("season grain", False)
    assert "season" in row["reason"]


def test_the_stages_split_the_edits_between_them_rather_than_sharing_them() -> None:
    """`weeks` is what keeps a season stage and a game stage from both claiming one edit."""
    frame = pl.DataFrame({"player_id": ["p1", "p1"], "week": [1, 2], "target_share": [0.2, 0.2]})
    sc = Scenario().set(
        Override("player", "p1", "target_share", "set", 0.3),
        Override("player", "p1", "target_share", "set", 0.5, week=2),
    )
    season, log = overrides.apply(frame, sc, "player", "player_id", week_col="week", weeks="never")
    assert [r["week"] for r in log] == [None]
    assert season["target_share"].to_list() == [0.3, 0.3]
    game, log = overrides.apply(season, sc, "player", "player_id", week_col="week", weeks="only")
    assert [r["week"] for r in log] == [2]
    assert game["target_share"].to_list() == [0.3, 0.5]
    with pytest.raises(ValueError, match="any, only or never"):
        overrides.apply(frame, sc, "player", "player_id", weeks="sometimes")


# --------------------------------------------------------------------------- #
# moving somebody on the depth chart
# --------------------------------------------------------------------------- #
# A slot is not a number applied to a projection, it is the row every prior is read off, so these tests
# are about the two things that separate a move from a relabelling: the room stays a legal chart, and the
# man who moved is *re-priced* off where he landed rather than keeping what his old slot was worth.
@pytest.fixture(scope="module")
def room(base) -> dict:
    """A quarterback room with a starter and a backup, which is where a promotion is most visible."""
    qbs = (base.part.filter((pl.col("position") == "QB") & (pl.col("depth_slot") <= 2))
           .sort(["team", "depth_slot"]))
    for team in qbs["team"].unique(maintain_order=True).to_list():
        mine = qbs.filter(pl.col("team") == team)
        if mine.height == 2:
            return {"team": team, "starter": mine.row(0, named=True), "backup": mine.row(1, named=True)}
    raise AssertionError("no team has two projected quarterbacks")


@pytest.fixture(scope="module")
def promoted(room) -> overrides.Run:
    return overrides.run(
        Scenario(name="promotion").set(
            Override("player", room["backup"]["player_id"], overrides.DEPTH_FIELD, "set", 1.0,
                     base=2.0)
        ),
        PROJ_SEASON,
    )


def test_a_depth_slot_is_an_override_the_scenario_accepts() -> None:
    sc = Scenario().set(Override("player", "p1", overrides.DEPTH_FIELD, "set", 1.0))
    assert len(sc.items) == 1
    assert overrides.DEPTH_FIELD not in overrides.PLAYER_FIELDS   # no numeric stage may touch it
    assert overrides.DEPTH_FIELD in overrides.FIELDS["player"]


def test_a_promotion_swaps_the_two_men_and_leaves_the_rest_of_the_room_alone(promoted, room) -> None:
    after = (promoted.roster.filter((pl.col("team") == room["team"]) & (pl.col("position") == "QB"))
             .sort(overrides.DEPTH_FIELD))
    slots = dict(zip(after["player_id"].to_list(), after[overrides.DEPTH_FIELD].to_list(), strict=True))
    assert slots[room["backup"]["player_id"]] == 1
    assert slots[room["starter"]["player_id"]] == 2
    assert sorted(slots.values()) == list(range(1, after.height + 1))


def test_the_chart_is_still_a_chart_everywhere_after_a_move(promoted) -> None:
    counted = promoted.roster.group_by("team", "position").agg(
        pl.col(overrides.DEPTH_FIELD).min().alias("lo"),
        pl.col(overrides.DEPTH_FIELD).max().alias("hi"),
        pl.col(overrides.DEPTH_FIELD).n_unique().alias("distinct"),
        pl.len().alias("n"),
    )
    assert (counted["lo"] == 1).all()
    assert (counted["hi"] == counted["n"]).all()
    assert (counted["distinct"] == counted["n"]).all()


def test_the_slots_the_priors_are_keyed_on_follow_the_move(promoted, room) -> None:
    from src.model import priors, roster

    moved = promoted.roster.filter(pl.col("player_id") == room["backup"]["player_id"]).row(0, named=True)
    assert moved["slot_bucket"] == min(1, priors.SLOT_CAP["QB"])
    assert moved["avail_slot"] == min(1, roster.AVAIL_SLOT_CAP)
    demoted = promoted.roster.filter(pl.col("player_id") == room["starter"]["player_id"]).row(0,
                                                                                              named=True)
    assert demoted["slot_bucket"] == min(2, priors.SLOT_CAP["QB"])


def test_a_promoted_backup_is_priced_as_a_starter_rather_than_handed_a_starters_share(
    base, promoted, room
) -> None:
    """The point of the whole stage: his availability and his share are re-estimated off the new slot.

    A relabelling would move the share -- he is first in the quarterback queue now -- and leave his
    expected games where a backup's were. Both have to move, in opposite directions for the two men.
    """
    up = promoted.part.filter(pl.col("player_id") == room["backup"]["player_id"]).row(0, named=True)
    down = promoted.part.filter(pl.col("player_id") == room["starter"]["player_id"]).row(0, named=True)
    assert up["expected_games"] > room["backup"]["expected_games"]
    assert down["expected_games"] < room["starter"]["expected_games"]

    def share(run: overrides.Run, pid: str) -> float:
        return float(run.shares.filter(pl.col("player_id") == pid)["dropback_share"][0])

    assert share(promoted, room["backup"]["player_id"]) > share(base, room["backup"]["player_id"])
    assert share(promoted, room["starter"]["player_id"]) < share(base, room["starter"]["player_id"])


def test_the_promotion_moves_the_board_and_the_demotion_moves_it_back(base, promoted, room) -> None:
    moved = overrides.diff(base.board, promoted.board)
    ids = set(moved["player_id"].to_list())
    assert room["backup"]["player_id"] in ids
    assert room["starter"]["player_id"] in ids
    him = moved.filter(pl.col("player_id") == room["backup"]["player_id"]).row(0, named=True)
    them = moved.filter(pl.col("player_id") == room["starter"]["player_id"]).row(0, named=True)
    assert him["d_fantasy_points"] > 0.0 > them["d_fantasy_points"]


def test_sending_the_starter_down_promotes_the_man_behind_him(room) -> None:
    """The direction a re-sort cannot do, and the reason the room is rebuilt rather than re-ranked.

    Nobody asks for the slot a demoted starter leaves, so ranking by the wanted slot left him first
    anyway. The man behind him has to be pulled up into it.
    """
    sc = Scenario().set(Override("player", room["starter"]["player_id"], overrides.DEPTH_FIELD,
                                 "set", 2.0, base=1.0))
    got = overrides.run(sc, PROJ_SEASON)
    after = got.roster.filter((pl.col("team") == room["team"]) & (pl.col("position") == "QB"))
    slots = dict(zip(after["player_id"].to_list(), after[overrides.DEPTH_FIELD].to_list(), strict=True))
    assert slots[room["starter"]["player_id"]] == 2
    assert slots[room["backup"]["player_id"]] == 1
    assert got.provenance.filter(pl.col("stage") == "depth")["used_now"][0] == pytest.approx(2.0)


def test_a_slot_past_the_end_of_the_room_is_the_back_of_it(room) -> None:
    """Asking for slot 30 in a four-man room is a demotion to last, not a hole in the chart."""
    sc = Scenario().set(Override("player", room["starter"]["player_id"], overrides.DEPTH_FIELD,
                                 "set", 30.0))
    got = overrides.run(sc, PROJ_SEASON)
    after = got.roster.filter((pl.col("team") == room["team"]) & (pl.col("position") == "QB"))
    him = after.filter(pl.col("player_id") == room["starter"]["player_id"]).row(0, named=True)
    assert him[overrides.DEPTH_FIELD] == after.height
    assert sorted(after[overrides.DEPTH_FIELD].to_list()) == list(range(1, after.height + 1))


def test_a_move_is_reported_as_its_own_stage_in_slots(promoted, room) -> None:
    rows = promoted.provenance.filter(pl.col("stage") == "depth")
    assert rows.height == 1
    got = rows.row(0, named=True)
    assert got["key"] == room["backup"]["player_id"]
    assert got["applied"]
    assert got["base_now"] == pytest.approx(2.0)     # slots, not values
    assert got["used_now"] == pytest.approx(1.0)


def test_a_depth_slot_is_set_rather_than_multiplied() -> None:
    sc = Scenario().set(Override("player", "p1", overrides.DEPTH_FIELD, "multiply", 0.5))
    got = overrides.run(sc, PROJ_SEASON)
    rows = got.provenance.filter(pl.col("stage") == "depth")
    assert rows.height == 1
    assert not rows["applied"][0]
    assert "not multiplied" in rows["reason"][0]


def test_a_move_of_somebody_who_is_not_there_says_so(base) -> None:
    sc = Scenario().set(Override("player", "00-9999999", overrides.DEPTH_FIELD, "set", 1.0))
    got = overrides.run(sc, PROJ_SEASON)
    rows = got.provenance.filter(pl.col("stage") == "depth")
    assert rows.height == 1
    assert not rows["applied"][0]
    assert "no rows" in rows["reason"][0]
    assert overrides.diff(base.board, got.board).is_empty()


def test_two_men_sent_to_the_same_slot_are_ordered_by_the_later_edit(room) -> None:
    """`Scenario.set` keys on the override id, so item order is not recency -- `at` has to be."""
    first, second = room["starter"]["player_id"], room["backup"]["player_id"]
    sc = Scenario().set(
        Override("player", first, overrides.DEPTH_FIELD, "set", 1.0, at="2026-01-01T00:00:00"),
        Override("player", second, overrides.DEPTH_FIELD, "set", 1.0, at="2026-01-02T00:00:00"),
    )
    got = overrides.run(sc, PROJ_SEASON)
    after = got.roster.filter(pl.col("player_id").is_in([first, second]))
    slots = dict(zip(after["player_id"].to_list(), after[overrides.DEPTH_FIELD].to_list(), strict=True))
    assert slots[second] == 1        # the more recent ask wins the slot
    assert slots[first] == 2


# --------------------------------------------------------------------------- #
# releasing a player: the one edit that removes a row instead of changing a number
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cut_room(base) -> dict:
    """A receiving room with somebody behind the man being released, so the promotion is visible."""
    wrs = (base.board.filter(pl.col("position") == "WR").select("player_id", "team", "depth_slot")
           .sort(["team", "depth_slot"]))
    for team in wrs["team"].unique(maintain_order=True).to_list():
        mine = wrs.filter(pl.col("team") == team)
        if mine.height >= 4:
            return {"team": team, "gone": mine.row(1, named=True), "behind": mine.row(2, named=True),
                    "room": mine}
    raise AssertionError("no team has four projected receivers")


@pytest.fixture(scope="module")
def released(cut_room) -> overrides.Run:
    return overrides.run(
        Scenario(name="release").set(
            Override("player", cut_room["gone"]["player_id"], overrides.ROSTER_FIELD, "set",
                     overrides.OFF_ROSTER)
        ),
        PROJ_SEASON,
    )


def test_a_release_is_an_override_the_scenario_accepts() -> None:
    sc = Scenario().set(Override("player", "p1", overrides.ROSTER_FIELD, "set", 0.0))
    assert len(sc.items) == 1
    # no numeric stage may touch it: it is a row, not a cell
    assert overrides.ROSTER_FIELD not in overrides.PLAYER_FIELDS
    assert overrides.ROSTER_FIELD not in overrides.GAME_ALL_FIELDS
    assert overrides.ROSTER_FIELD in overrides.FIELDS["player"]


def test_a_released_player_is_in_no_frame_the_run_produced(released, cut_room) -> None:
    """Removed rather than projected small -- nothing downstream can count him."""
    pid = cut_room["gone"]["player_id"]
    for name in ("roster", "part", "shares", "rates", "opp", "adj", "weekly", "season_frame",
                 "board"):
        frame = getattr(released, name)
        assert "player_id" in frame.columns, name
        assert frame.filter(pl.col("player_id") == pid).is_empty(), name


def test_the_room_closes_up_behind_him_and_is_still_a_chart(released, cut_room) -> None:
    after = (released.roster.filter((pl.col("team") == cut_room["team"])
                                    & (pl.col("position") == "WR"))
             .sort(overrides.DEPTH_FIELD))
    assert after.height == cut_room["room"].height - 1
    assert after[overrides.DEPTH_FIELD].to_list() == list(range(1, after.height + 1))
    behind = after.filter(pl.col("player_id") == cut_room["behind"]["player_id"]).row(0, named=True)
    assert behind[overrides.DEPTH_FIELD] == cut_room["behind"]["depth_slot"] - 1
    # the slots the priors and the survival curve are keyed on followed him up
    assert behind["slot_bucket"] <= behind[overrides.DEPTH_FIELD]
    assert behind["avail_slot"] == min(behind[overrides.DEPTH_FIELD], 12)


def test_his_pool_goes_to_the_room_and_the_team_still_adds_up(base, released, cut_room) -> None:
    """A released receiver's targets are not lost: normalisation hands them to the men who are left."""
    mine = pl.col("team") == cut_room["team"]
    before = float(base.board.filter(mine)["targets"].sum())
    after = float(released.board.filter(mine)["targets"].sum())
    assert after == pytest.approx(before, rel=1e-6)
    moved = overrides.diff(base.board, released.board)
    assert set(moved["team"].unique().to_list()) == {cut_room["team"]}
    assert moved.filter(pl.col("player_id") == cut_room["behind"]["player_id"])["d_targets"][0] > 0.0


def test_a_release_is_reported_as_its_own_stage(released, cut_room) -> None:
    rows = released.provenance.filter(pl.col("stage") == "roster")
    assert rows.height == 1
    got = rows.row(0, named=True)
    assert got["key"] == cut_room["gone"]["player_id"]
    assert got["applied"] and got["rows"] == 1
    assert got["base_now"] == pytest.approx(1.0)
    assert got["used_now"] == pytest.approx(overrides.OFF_ROSTER)


def test_a_roster_spot_is_set_rather_than_multiplied() -> None:
    sc = Scenario().set(Override("player", "p1", overrides.ROSTER_FIELD, "multiply", 0.0))
    rows = overrides.run(sc, PROJ_SEASON).provenance.filter(pl.col("stage") == "roster")
    assert rows.height == 1
    assert not rows["applied"][0]
    assert "not multiplied" in rows["reason"][0]


def test_only_zero_releases_him(base, cut_room) -> None:
    sc = Scenario().set(Override("player", cut_room["gone"]["player_id"], overrides.ROSTER_FIELD,
                                 "set", 1.0))
    got = overrides.run(sc, PROJ_SEASON)
    rows = got.provenance.filter(pl.col("stage") == "roster")
    assert not rows["applied"][0]
    assert "nothing to apply" in rows["reason"][0]
    assert got.board.height == base.board.height


def test_releasing_somebody_who_is_not_there_says_so(base) -> None:
    sc = Scenario().set(Override("player", "00-9999999", overrides.ROSTER_FIELD, "set", 0.0))
    got = overrides.run(sc, PROJ_SEASON)
    rows = got.provenance.filter(pl.col("stage") == "roster")
    assert not rows["applied"][0]
    assert "no rows" in rows["reason"][0]
    assert overrides.diff(base.board, got.board).is_empty()


def test_a_week_scoped_release_is_reported_rather_than_guessed_at(base) -> None:
    pid = base.board["player_id"][0]
    sc = Scenario().set(Override("player", pid, overrides.ROSTER_FIELD, "set", 0.0, week=5))
    got = overrides.run(sc, PROJ_SEASON)
    assert got.provenance.filter(pl.col("stage") == "roster").is_empty()
    grain = got.provenance.filter(pl.col("stage") == "season grain")
    assert grain.height == 1
    assert not grain["applied"][0]
    assert "p_play" in grain["reason"][0]
    assert not got.board.filter(pl.col("player_id") == pid).is_empty()


def test_a_room_cannot_be_emptied_and_the_last_man_left_is_told_why(base) -> None:
    """A pool with nobody to divide it is not a projection, so the deepest man in the room stays."""
    counts = base.board.group_by("team", "position").len().sort("len")
    small = counts.filter(pl.col("position") == "QB").row(0, named=True)
    room = (base.board.filter((pl.col("team") == small["team"]) & (pl.col("position") == "QB"))
            .sort("depth_slot"))
    sc = Scenario().set(*[Override("player", pid, overrides.ROSTER_FIELD, "set", 0.0)
                          for pid in room["player_id"].to_list()])
    got = overrides.run(sc, PROJ_SEASON)
    left = got.roster.filter((pl.col("team") == small["team"]) & (pl.col("position") == "QB"))
    assert left.height == 1
    assert left["player_id"][0] == room["player_id"][-1]       # the deepest one, priced as a starter
    assert left[overrides.DEPTH_FIELD][0] == 1
    kept = got.provenance.filter((pl.col("stage") == "roster") & ~pl.col("applied")).row(0, named=True)
    assert "would empty" in kept["reason"]


def test_a_release_and_a_promotion_are_placed_in_the_room_that_is_left(base, cut_room) -> None:
    """The chart is settled after the release, so a slot asked for is a slot in the smaller room."""
    gone, behind = cut_room["gone"]["player_id"], cut_room["behind"]["player_id"]
    sc = Scenario().set(
        Override("player", gone, overrides.ROSTER_FIELD, "set", 0.0),
        Override("player", behind, overrides.DEPTH_FIELD, "set", 1.0),
    )
    got = overrides.run(sc, PROJ_SEASON)
    room = (got.roster.filter((pl.col("team") == cut_room["team"]) & (pl.col("position") == "WR"))
            .sort(overrides.DEPTH_FIELD))
    assert room.filter(pl.col("player_id") == behind)[overrides.DEPTH_FIELD][0] == 1
    assert room.filter(pl.col("player_id") == gone).is_empty()
    assert room[overrides.DEPTH_FIELD].to_list() == list(range(1, room.height + 1))


def test_the_release_changes_the_digest_and_comes_back_from_json(cut_room) -> None:
    sc = Scenario().set(Override("player", cut_room["gone"]["player_id"], overrides.ROSTER_FIELD,
                                 "set", 0.0))
    assert sc.digest != Scenario().digest
    again = Scenario.from_json(sc.to_json())
    assert again.digest == sc.digest
    assert again.items[0].field == overrides.ROSTER_FIELD


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


# --------------------------------------------------------------------------- #
# picking the work back up
# --------------------------------------------------------------------------- #
# The app holds the live scenario in Streamlit session state, which does not survive a browser reload
# or a server restart. A preseason pass across thirty-two teams is hours of typing, so the scenario is
# written to disk on every edit and the *name* of what was written is remembered. These are the
# guarantees that behaviour rests on.
def test_saving_remembers_what_was_saved(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    assert overrides.last_used() is None
    overrides.save(Scenario(name="preseason pass").set(
        Override("player", "p1", "target_share", value=0.3)))
    assert overrides.last_used() == "preseason pass"


def test_the_pointer_is_not_offered_as_a_scenario_to_load(tmp_path, monkeypatch) -> None:
    """It lives in the same directory, so `names()` has to know the difference."""
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    overrides.save(Scenario(name="mine"))
    assert overrides.pointer_path().is_file()
    assert overrides.names() == ["mine"]


def test_a_pointer_at_a_deleted_scenario_is_not_a_pointer(tmp_path, monkeypatch) -> None:
    """Otherwise the next session opens on a file that is not there and starts from nothing silently."""
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    overrides.save(Scenario(name="gone soon"))
    assert overrides.last_used() == "gone soon"
    overrides.delete("gone soon")
    assert overrides.last_used() is None


def test_the_newest_scenario_is_offered_first(tmp_path, monkeypatch) -> None:
    """Alphabetical stops being useful once a league-wide pass has produced a dozen of them."""
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    overrides.save(Scenario(name="aaa first"))
    overrides.save(Scenario(name="zzz second"))
    assert overrides.names()[0] == "zzz second"


def test_a_summary_says_how_big_each_saved_scenario_is(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    overrides.save(Scenario(name="two edits").set(
        Override("player", "p1", "target_share", value=0.3),
        Override("player", "p2", "carry_share", value=0.1)))
    got = overrides.summaries()
    assert [(s["name"], s["edits"]) for s in got] == [("two edits", 2)]
    assert got[0]["updated"]


def test_a_scenario_saved_over_keeps_the_date_it_was_created(tmp_path, monkeypatch) -> None:
    """`created` against `updated` is what tells you a pass has been revisited rather than started."""
    monkeypatch.setattr(overrides, "SCENARIOS", tmp_path)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)
    overrides.save(Scenario(name="revisited"))
    first = overrides.load("revisited")
    overrides.save(first.set(Override("player", "p1", "target_share", value=0.3)))
    again = overrides.load("revisited")
    assert again.created == first.created
    assert len(again.items) == 1
