"""The app's own logic: the parts of `app/ui.py` that are not Streamlit calls.

Almost everything in `ui` is chrome, and chrome is checked by running the pages -- `scripts/smoke_app.py`
does that in a real session, including clicking a reset and typing a multiplier. What is left, and what
is here, is the translation between an edited grid and a scenario: the one piece of the editing path
that a test can pin down exactly, and the one that would silently record the wrong `base` or the wrong
week if it were wrong.

`import ui` outside `streamlit run` warns about session state and then works, which is why the module
keeps a fallback for the no-session case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import ui  # noqa: E402

from src.model import compose, opportunity, overrides, roster  # noqa: E402
from src.model.overrides import Override, Scenario  # noqa: E402


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch) -> None:
    """Every edit is autosaved now, so a test run must not write into the user's own scenarios.

    `ui.live()` also reads the pointer file on a session's first access, which without this would open
    the tests on whatever the user last worked on.
    """
    room = tmp_path / "scenarios"
    room.mkdir()
    monkeypatch.setattr(overrides, "SCENARIOS", room)
    monkeypatch.setattr(overrides, "ensure_dirs", lambda: None)


@pytest.fixture(autouse=True)
def clean(sandbox) -> None:
    """Every test starts from the engine's own answer, since the live scenario is module state."""
    ui._state().pop(ui.LIVE, None)
    ui.set_live(Scenario())


@pytest.fixture
def before() -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": ["p1", "p2"],
        "player": ["A", "B"],
        "target_share": [0.24, 0.10],
        "catch_rate": [0.68, 0.62],
    })


FIELDS = ("target_share", "catch_rate")


def test_a_grid_nobody_touched_produces_nothing(before) -> None:
    assert ui.edits_from_grid(before, before, "player", "player_id", FIELDS) == []


def test_a_changed_cell_becomes_a_set_edit_on_that_player_alone(before) -> None:
    after = before.with_columns(
        pl.when(pl.col("player_id") == "p1").then(0.32).otherwise(pl.col("target_share"))
        .alias("target_share")
    )
    got = ui.edits_from_grid(before, after, "player", "player_id", FIELDS)
    assert len(got) == 1
    o = got[0]
    assert (o.level, o.key, o.field, o.mode, o.value) == ("player", "p1", "target_share", "set", 0.32)
    assert o.base == 0.24            # what it was worth before the edit, for the diff
    assert o.week is None


def test_the_recorded_base_stays_the_engines_number_across_a_second_edit(before) -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30, base=0.24))
    after = before.with_columns(pl.Series("target_share", [0.35, 0.10]))
    got = ui.edits_from_grid(before, after, "player", "player_id", FIELDS)
    assert got[0].value == 0.35
    assert got[0].base == 0.24


def test_a_weekly_grid_keys_each_edit_to_its_week() -> None:
    was = pl.DataFrame({"team": ["PHI"] * 3, "week": [1, 2, 3], "targets": [34.0, 30.0, 31.0]})
    now = was.with_columns(pl.Series("targets", [34.0, 40.0, 31.0]))
    got = ui.edits_from_grid(was, now, "team", "team", ("targets",), week_col="week")
    assert len(got) == 1
    assert (got[0].week, got[0].value, got[0].base) == (2, 40.0, 30.0)


def test_a_grid_of_a_different_shape_is_ignored_rather_than_misaligned(before) -> None:
    """Row order is the only thing tying the two frames together, so a different height is not an edit."""
    assert ui.edits_from_grid(before, before.head(1), "player", "player_id", FIELDS) == []


# --------------------------------------------------------------------------- #
# the team sheet
# --------------------------------------------------------------------------- #
# A grid of a whole roster is the one place where the number on screen is not necessarily the engine's
# own: a promotion renumbers the room, so the man pushed back is showing a slot nobody typed. That makes
# `base_frame` load-bearing rather than a nicety -- without it the sheet records "what it was" as the
# consequence of somebody else's edit -- and `require_base` the difference between refusing a
# quarterback's rate on a running back and writing down an edit the engine will only report unapplied.
def test_the_recorded_base_is_the_engines_number_rather_than_what_was_on_screen(before) -> None:
    engine = pl.DataFrame({"player_id": ["p1", "p2"], "target_share": [0.19, 0.10]})
    after = before.with_columns(pl.Series("target_share", [0.32, 0.10]))
    got = ui.edits_from_grid(before, after, "player", "player_id", FIELDS, base_frame=engine)
    assert (got[0].value, got[0].base) == (0.32, 0.19)


def test_an_existing_edit_still_outranks_the_base_frame(before) -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30, base=0.24))
    engine = pl.DataFrame({"player_id": ["p1"], "target_share": [0.19]})
    after = before.with_columns(pl.Series("target_share", [0.35, 0.10]))
    got = ui.edits_from_grid(before, after, "player", "player_id", FIELDS, base_frame=engine)
    assert got[0].base == 0.24


def test_a_cell_the_engine_has_no_number_for_is_refused_rather_than_recorded() -> None:
    """A completion percentage on a running back. The engine reports such an edit unapplied."""
    was = pl.DataFrame({"player_id": ["p1"], "completion_pct": [None], "catch_rate": [0.6]},
                       schema={"player_id": pl.String, "completion_pct": pl.Float64,
                               "catch_rate": pl.Float64})
    now = was.with_columns(pl.Series("completion_pct", [0.68]))
    fields = ("completion_pct", "catch_rate")
    assert ui.edits_from_grid(was, now, "player", "player_id", fields, require_base=True) == []
    # without the guard it is an edit with no base, which is what the sheet must not write
    loose = ui.edits_from_grid(was, now, "player", "player_id", fields)
    assert len(loose) == 1 and loose[0].base is None


def test_every_editable_player_field_is_reachable_from_some_column_set() -> None:
    """Otherwise the groups are quietly not a division of the whole, and a knob has no home.

    The sheet is a season sheet -- one row per player, no week -- so the per-game-only fields are not
    its business and are reached from the game editor instead. Nor is the release, which removes the row
    a sheet would have shown it in rather than filling a cell of it: its home is the depth chart's own
    control. Both are excluded by name rather than by a looser assertion, so adding a field still has to
    be a decision about where it can be edited.
    """
    grouped = {f for group in ui.SHEET_GROUPS.values() for f in group} | set(ui.SHEET_ALWAYS)
    editable = (set(overrides.FIELDS["player"]) - set(overrides.GAME_ONLY_FIELDS)
                - set(overrides.ROSTER_FIELDS))
    assert not editable - grouped, "no column set reaches these"
    assert not grouped - editable, "these are not fields the engine takes"


def test_the_slot_and_the_games_lead_every_column_set() -> None:
    for column_set in ui.SHEET_COLUMN_SETS:
        assert ui.sheet_fields(column_set, ["WR"])[:2] == ui.SHEET_ALWAYS


def test_the_default_column_set_follows_the_positions_on_screen() -> None:
    wr = ui.sheet_fields(ui.SHEET_KNOBS, ["WR"])
    qb = ui.sheet_fields(ui.SHEET_KNOBS, ["QB"])
    assert "target_share" in wr and "dropback_share" not in wr
    assert "dropback_share" in qb and "catch_rate" not in qb
    both = ui.sheet_fields(ui.SHEET_KNOBS, ["QB", "WR"])
    assert "dropback_share" in both and "target_share" in both
    # and the shortlist is a shortlist: naming no position is not the same as everything editable
    assert len(both) < len(ui.sheet_fields(ui.SHEET_EVERYTHING))


def test_a_pool_wins_over_the_participation_column_of_the_same_name() -> None:
    """`snap_share` is both, and the sheet must show the one the projection divides with."""
    part = pl.DataFrame({"player_id": ["p1"], "depth_slot": [1], "expected_games": [16.0],
                         "snap_share": [0.11]})
    shares = pl.DataFrame({"player_id": ["p1"], "snap_share": [0.88]})
    rates = pl.DataFrame({"player_id": ["p1"], "catch_rate": [0.6]})
    got = ui._sheet_values(part, shares, rates, ("depth_slot", "expected_games", "snap_share",
                                                 "catch_rate"))
    assert got["snap_share"][0] == 0.88
    assert got["depth_slot"][0] == 1 and got["catch_rate"][0] == 0.6


# --------------------------------------------------------------------------- #
# the game sheet
# --------------------------------------------------------------------------- #
# The roster sheet's failure mode is the wrong `base`; this one's is the wrong *grain*. A sheet of one
# week that wrote season-wide overrides would look identical on screen and be wrong about sixteen games
# in order to be right about one, so what is pinned here is that the fields offered are fields a week
# can carry, and that a cell typed on a week's sheet comes back keyed to that week.
def test_a_game_sheet_offers_only_fields_a_week_can_carry() -> None:
    for column_set in ui.SHEET_COLUMN_SETS:
        fields = ui.game_fields(column_set, ["QB", "RB", "WR", "TE"])
        assert fields, column_set
        assert not set(fields) - set(overrides.GAME_ALL_FIELDS), column_set
        # the two season knobs the roster sheet always leads with, and the reason this is a second set
        assert overrides.DEPTH_FIELD not in fields and ui.GAMES_METRIC not in fields


def test_the_chance_he_plays_leads_every_game_column_set() -> None:
    """It is the knob the surface exists for: he is out this week, and the room takes the work."""
    assert ui.GAME_SHEET_ALWAYS == overrides.GAME_ONLY_FIELDS
    for column_set in ui.SHEET_COLUMN_SETS:
        fields = ui.game_fields(column_set, ["RB"])
        assert fields[0] == "p_play"
        assert fields.count("p_play") == 1


def test_a_cell_typed_on_a_game_sheet_is_keyed_to_that_game_alone() -> None:
    was = pl.DataFrame({"player_id": ["p1", "p2"], "week": [5, 5],
                        "p_play": [0.89, 0.74], "target_share": [0.24, 0.10]})
    now = was.with_columns(pl.Series("p_play", [0.0, 0.74]))
    got = ui.edits_from_grid(was, now, "player", "player_id", ("p_play", "target_share"),
                             week_col="week")
    assert len(got) == 1
    o = got[0]
    assert (o.level, o.key, o.field, o.value, o.week) == ("player", "p1", "p_play", 0.0, 5)
    assert o.base == 0.89


def test_the_same_man_in_two_weeks_is_two_edits_rather_than_one() -> None:
    was = pl.DataFrame({"player_id": ["p1", "p1"], "week": [5, 6], "p_play": [0.89, 0.89]})
    now = was.with_columns(pl.Series("p_play", [0.0, 0.5]))
    got = ui.edits_from_grid(was, now, "player", "player_id", ("p_play",), week_col="week")
    assert sorted((o.week, o.value) for o in got) == [(5, 0.0), (6, 0.5)]
    assert len({o.id for o in got}) == 2


def test_a_game_edit_records_the_unedited_engine_number_rather_than_the_room_it_left() -> None:
    """The second edit in a game is made against a screen the first edit moved.

    Rule a starter out and his teammates' shares rise, because the pool was divided without him. Typing
    over one of those risen numbers must record what the engine had before any of it, or resetting the
    second edit would put the man back on the first edit's answer rather than on the estimator.
    """
    was = pl.DataFrame({"player_id": ["p2"], "week": [5], "target_share": [0.31]})   # risen
    engine = pl.DataFrame({"player_id": ["p2"], "week": [5], "target_share": [0.19]})
    now = was.with_columns(pl.Series("target_share", [0.35]))
    got = ui.edits_from_grid(was, now, "player", "player_id", ("target_share",), week_col="week",
                             base_frame=engine)
    assert (got[0].value, got[0].base, got[0].week) == (0.35, 0.19, 5)


def test_the_game_reconciliation_is_the_models_own_mapping_without_the_yardage() -> None:
    """One game's version of `compose.team_check`, so the pairs come from the model rather than a page."""
    assert set(ui.GAME_RECONCILE) <= set(compose.RECONCILE)
    assert all(not c.startswith("team_") for c in ui.GAME_RECONCILE.values())
    # a count out of a normalised pool must match; yards are a count times a rate and are not forced to
    assert "targets" in ui.GAME_RECONCILE and "carries" in ui.GAME_RECONCILE
    assert "receiving_yards" not in ui.GAME_RECONCILE and "rushing_yards" not in ui.GAME_RECONCILE


def test_a_games_two_rows_come_out_in_a_stable_order_even_with_nobody_at_home() -> None:
    """A neutral-site game has no home team, and a fixture still has to be written one way round."""
    env = pl.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "team": ["LAC", "ARI", "JAX", "NE"],
        "is_home": [True, False, False, False],
    })
    got = ui._sided(env)
    assert got.filter(pl.col("game_id") == "g1").sort("side")["team"].to_list() == ["LAC", "ARI"]
    assert sorted(got.filter(pl.col("game_id") == "g2")["side"].to_list()) == [0, 1]


# --------------------------------------------------------------------------- #
# the posted lines, as a check
# --------------------------------------------------------------------------- #
# The whole value of this surface is that the two numbers on it are independent. `model_points` on the
# environment has already had the team's market offset added, so comparing *that* to the line it was
# fitted against would show a model that agrees with the market because it was moved to.
def test_the_market_check_compares_the_line_to_the_models_unblended_estimate() -> None:
    v = ui.View(payload=Scenario(name="market").content_json())
    gaps = ui.market_gaps(v)
    if gaps.is_empty():                     # a season with no lines on file has nothing to check
        pytest.skip("no posted lines for this season")
    env = ui.environment(v).filter(pl.col("has_market"))
    joined = gaps.join(env.select("week", "team", pl.col("model_points").alias("blended")),
                       on=["week", "team"], how="inner")
    assert joined.height == gaps.height
    # the offset is taken back out, and the gap is that estimate against the line
    assert ((joined["own_points"] + joined["market_offset"] - joined["blended"]).abs() < 1e-9).all()
    assert ((joined["own_points"] - joined["market_points"] - joined["gap"]).abs() < 1e-9).all()
    # and what the projection runs on lies between the two, because it is a blend of them
    lo = pl.min_horizontal("market_points", "blended")
    hi = pl.max_horizontal("market_points", "blended")
    assert joined.filter((pl.col("implied_points") < lo - 1e-6)
                         | (pl.col("implied_points") > hi + 1e-6)).is_empty()


def test_the_team_rollup_counts_only_lined_games_and_carries_one_offset() -> None:
    """A team's gap is estimated from its lined games alone; the offset it carries is one number."""
    v = ui.View(payload=Scenario(name="market").content_json())
    per, gaps = ui.market_by_team(v), ui.market_gaps(v)
    if per.is_empty():
        pytest.skip("no posted lines for this season")
    assert int(per["lined"].sum()) == gaps.height
    assert (per["lined"] <= per["games"]).all()
    one = gaps.group_by("team").agg(pl.col("market_offset").n_unique().alias("n"))
    assert one["n"].max() == 1


def test_the_weight_the_page_reports_is_the_one_the_engine_will_use() -> None:
    """`None` means "use what was fitted", and a reader checking a projection needs the number."""
    from src.model import team as team_model

    fitted = float(team_model.load_market()["market_weight"])
    assert ui.fitted_market_weight() == fitted
    assert ui.market_weight_used(ui.View(payload=Scenario(name="a").content_json())) == fitted
    off = Scenario(name="b").patch_league(market_weight=0.0)
    assert ui.market_weight_used(ui.View(payload=off.content_json())) == 0.0
    # the sidebar dial: any weight between the two ends survives the round trip, and the fitted number
    # itself is stored as unset so a refit still moves it
    part = Scenario(name="c").patch_league(market_weight=0.25)
    assert ui.market_weight_used(ui.View(payload=part.content_json())) == 0.25
    same = Scenario(name="d").patch_league(market_weight=0.25).patch_league(market_weight=None)
    assert "market_weight" not in same.league
    assert ui.market_weight_used(ui.View(payload=same.content_json())) == fitted


def test_the_view_is_the_scenarios_content_and_settings_follow_it() -> None:
    ui.set_live(Scenario(name="mine").patch_league(normalize_pools=False, scoring="half_ppr"))
    v = ui.view()
    assert v.payload == ui.live().content_json()
    assert v.scenario.digest == ui.live().digest
    assert v.settings.normalize_pools is False
    assert v.settings.scoring.reception == 0.5
    assert v.scoring == "half_ppr"


def test_renaming_the_live_scenario_does_not_change_what_is_cached() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    first = ui.view().payload
    ui.set_live(ui.live().rename("consensus"))
    assert ui.view().payload == first


# --------------------------------------------------------------------------- #
# the stat line
# --------------------------------------------------------------------------- #
# The columns a page shows for a position, and the sentence it prints for a player. Pure formatting, and
# worth pinning because the failure is silent: a stat line that quietly drops the touchdowns still
# renders, and a reader has no way to know a column is missing rather than zero.
QB_ROW = {
    "position": "QB", "attempts": 546.0, "completions": 361.7, "passing_yards": 4180.4,
    "passing_tds": 31.2, "interceptions": 9.4, "carries": 42.3, "rushing_yards": 210.8,
    "rushing_tds": 2.6, "targets": 0.0, "receptions": 0.0,
}
WR_ROW = {
    "position": "WR", "targets": 148.6, "receptions": 101.9, "receiving_yards": 1241.3,
    "receiving_tds": 7.5, "carries": 3.1, "rushing_yards": 19.4, "rushing_tds": 0.04,
}


def test_one_position_gets_its_own_line_and_a_mix_gets_the_union() -> None:
    assert ui.stat_columns("QB") == list(ui.STAT_LINE["QB"])
    assert ui.stat_columns(["WR"]) == list(ui.STAT_LINE["WR"])
    both = ui.stat_columns(["QB", "WR"])
    assert "attempts" in both and "targets" in both
    # the union is in box-score order, not in the order the positions were named
    assert both.index("passing_yards") < both.index("receiving_yards")
    # an unknown or empty filter falls back to every position rather than to nothing
    assert ui.stat_columns([]) == ui.stat_columns(None) == ui.stat_columns(["K"])


def test_the_stat_line_reads_as_a_box_score_and_drops_what_a_player_does_not_do() -> None:
    qb = ui.stat_line(QB_ROW)
    assert "362/546" in qb and "4,180 yds" in qb and "31.2 TD" in qb and "9.4 INT" in qb
    assert "42 car" in qb                          # a running quarterback keeps his rushing line
    wr = ui.stat_line(WR_ROW)
    assert "102/149 rec" in wr and "1,241 rec yds" in wr
    assert "3 car" in wr
    # a receiver with no carries is not given a rushing line of zeroes
    assert "car" not in ui.stat_line({**WR_ROW, "carries": 0.2, "rushing_yards": 1.0})
    assert ui.stat_line({"position": "TE"}) == "no projected volume"


def test_yards_are_whole_and_touchdowns_are_not() -> None:
    assert ui.stat_digits("passing_yards") == 0
    assert ui.stat_digits("passing_tds") == 1
    cfg = ui.stat_config(["passing_yards", "passing_tds"])
    assert cfg["passing_yards"]["type_config"]["format"] == "%.0f"
    assert cfg["passing_tds"]["type_config"]["format"] == "%.1f"


# --------------------------------------------------------------------------- #
# the batched editor
# --------------------------------------------------------------------------- #
# The bench is one player's whole rating card with three empty columns on the end, and its failure modes
# are all silent: a factor of 1.0 written down as an override, a season row taking every week with it, or
# three chosen weeks sharing one week's base. None of those look wrong on screen -- they look like an edit
# list with rows in it -- so the translation from typed cell to override is pinned here, and that it
# reaches a page is `scripts/smoke_app.py`.
def typed_bench(**cells: dict) -> pl.DataFrame:
    """A bench sheet as `st.data_editor` hands it back: one row a knob, the three edit columns filled."""
    fields = list(cells)
    return pl.DataFrame(
        {
            "field": fields,
            "set to": [cells[f].get("set to") for f in fields],
            "x by": [cells[f].get("x by") for f in fields],
            "drop": [bool(cells[f].get("drop", False)) for f in fields],
        },
        schema={"field": pl.String, "set to": pl.Float64, "x by": pl.Float64, "drop": pl.Boolean},
    )


def bench_edits(after: pl.DataFrame, *, weeks: tuple[int, ...] = (), bases: dict | None = None):
    """`ui.bench_edits` with the player fixed, since every test here is about one man's sheet."""
    return ui.bench_edits(after, "p1", weeks, bases or {})


def test_a_typed_value_and_a_typed_factor_are_the_two_modes_of_an_override() -> None:
    after = typed_bench(target_share={"set to": 0.32}, yards_per_target={"x by": 1.1})
    writes, drops = bench_edits(after, bases={("target_share", None): 0.24,
                                             ("yards_per_target", None): 8.4})
    assert drops == []
    by_field = {o.field: o for o in writes}
    assert (by_field["target_share"].mode, by_field["target_share"].value) == ("set", 0.32)
    assert by_field["target_share"].base == 0.24        # the engine's own number, for the ↺
    # a multiplier stays a multiplier: baked into a number it would stop tracking the engine
    assert (by_field["yards_per_target"].mode, by_field["yards_per_target"].value) == ("multiply", 1.1)
    assert all(o.level == "player" and o.key == "p1" and o.week is None for o in writes)


def test_an_override_that_changes_nothing_is_not_written_down() -> None:
    """Otherwise the edit list fills with rows that look like decisions and are not.

    Both cases arise from reading rather than typing: the value already in the cell gets re-entered while
    scanning the sheet, and `x by` reads as a field wanting a number so it gets a 1.
    """
    after = typed_bench(target_share={"set to": 0.24}, catch_rate={"x by": 1.0})
    assert bench_edits(after, bases={("target_share", None): 0.24}) == ([], [])


def test_a_drop_beats_anything_else_typed_in_the_same_row() -> None:
    """Ticking drop and typing a value is a mind changed twice, and the tick is the later thought."""
    after = typed_bench(target_share={"set to": 0.32, "drop": True})
    writes, drops = bench_edits(after, bases={("target_share", None): 0.24})
    assert writes == []
    assert drops == [("target_share", None)]


def test_one_typed_row_over_three_weeks_is_three_edits_each_against_its_own_week() -> None:
    """The reason the scope control is worth having: "he is on a pitch count in December" in one pass.

    Each week's base has to be that week's own number -- a bye-shortened week 12 and a week 13 do not
    share a `p_play` -- or dropping one of the three would put him on another week's answer.
    """
    after = typed_bench(p_play={"set to": 0.5})
    writes, drops = bench_edits(after, weeks=(12, 13, 14),
                                bases={("p_play", 12): 0.9, ("p_play", 13): 0.8, ("p_play", 14): 0.7})
    assert drops == []
    assert sorted((o.week, o.base) for o in writes) == [(12, 0.9), (13, 0.8), (14, 0.7)]
    assert {o.value for o in writes} == {0.5} and len({o.id for o in writes}) == 3


def test_a_week_the_engine_has_no_number_for_is_still_written_against_no_base() -> None:
    """A week override the engine cannot place is reported unapplied, which is visible; a missing base
    silently recorded as another week's would not be."""
    after = typed_bench(p_play={"set to": 0.5})
    writes, _ = bench_edits(after, weeks=(12, 13), bases={("p_play", 12): 0.9})
    assert sorted((o.week, o.base) for o in writes) == [(12, 0.9), (13, None)]


def test_a_bench_scoped_to_weeks_only_offers_knobs_a_week_can_carry() -> None:
    """The slot and the games count are season facts, and the sheet must not pretend otherwise."""
    for column_set in ui.BENCH_SETS:
        season = ui.bench_field_set(column_set, "WR")
        week = ui.bench_field_set(column_set, "WR", per_week=True)
        assert week, column_set
        assert not set(week) - set(overrides.GAME_ALL_FIELDS), column_set
        assert overrides.DEPTH_FIELD not in week and ui.GAMES_METRIC not in week
        assert not set(season) - (set(overrides.PLAYER_FIELDS) | {overrides.DEPTH_FIELD}), column_set


def test_dropping_a_season_edit_leaves_the_week_edits_on_the_same_knob_alone() -> None:
    """`Scenario.clear` reads a missing week as a wildcard, which is right for "clear this player" and
    wrong for one row of a sheet: the two edits are separate claims and are dropped separately."""
    ui.edit(Override("player", "p1", "target_share", "set", 0.30),
            Override("player", "p1", "target_share", "set", 0.05, week=12),
            Override("player", "p1", "catch_rate", "set", 0.7))
    ui.drop_edit("player", "p1", "target_share")
    assert {(o.field, o.week) for o in ui.live().items} == {("target_share", 12), ("catch_rate", None)}
    # and the week edit goes on its own too, without taking the season one it was made beside
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    ui.drop_edit("player", "p1", "target_share", week=12)
    assert {(o.field, o.week) for o in ui.live().items} == {("target_share", None),
                                                           ("catch_rate", None)}


def test_releasing_a_man_is_one_edit_and_reinstating_him_drops_it() -> None:
    """The page's two verbs, which have to be exact inverses or a scenario keeps a hole in it."""
    ui.release("p1", "p2")
    assert ui.released_ids() == ["p1", "p2"]
    assert {(o.field, o.mode, o.value, o.base) for o in ui.live().items} == {
        (overrides.ROSTER_FIELD, "set", 0.0, 1.0)}
    ui.reinstate("p1")
    assert ui.released_ids() == ["p2"]
    ui.reinstate("p2")
    assert ui.live().is_baseline


def test_releasing_the_same_man_twice_is_still_one_edit() -> None:
    ui.release("p1")
    ui.release("p1")
    assert len(ui.live().items) == 1


def test_a_zeroed_availability_is_not_a_release() -> None:
    """The distinction the control exists for: no games still leaves him holding his slot."""
    ui.edit(Override("player", "p1", ui.GAMES_METRIC, "set", 0.0, base=16.0))
    assert ui.released_ids() == []


def test_the_release_is_not_a_cell_in_any_sheet() -> None:
    """It takes the row a sheet would have shown him in, so `everything editable` must not offer it."""
    assert overrides.ROSTER_FIELD not in ui.sheet_fields(ui.SHEET_EVERYTHING)
    assert overrides.ROSTER_FIELD not in ui.BULK_FIELDS
    for column_set in ui.BENCH_SETS:
        assert overrides.ROSTER_FIELD not in ui.bench_field_set(column_set, "WR")


# --------------------------------------------------------------------------- #
# where the overrides are
# --------------------------------------------------------------------------- #
def test_a_frame_says_which_rows_carry_an_override_and_what_it_says(before) -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.32, base=0.24))
    got = ui.with_edits(before, "player", "player_id")
    marked = got.filter(pl.col("edited"))
    assert marked["player_id"].to_list() == ["p1"]
    assert marked["edits"][0] == "target_share =0.32"
    assert got.filter(~pl.col("edited"))["edits"].to_list() == [""]


def test_a_team_edit_does_not_mark_a_player_row_and_a_weekly_edit_names_its_week(before) -> None:
    ui.edit(Override("team", "ARI", "targets", "multiply", 1.25))
    assert not ui.with_edits(before, "player", "player_id")["edited"].any()
    teams = pl.DataFrame({"team": ["ARI", "BUF"]})
    got = ui.with_edits(teams, "team", "team")
    assert got.filter(pl.col("team") == "ARI")["edits"][0] == "targets x1.25"

    ui.edit(Override("team", "ARI", "carries", "set", 30.0, week=4))
    got = ui.with_edits(teams, "team", "team")
    assert "carries wk4 =30" in got.filter(pl.col("team") == "ARI")["edits"][0]


def test_a_frame_with_no_edits_still_has_the_two_columns(before) -> None:
    """So a page can select them unconditionally rather than branching on the scenario."""
    got = ui.with_edits(before, "player", "player_id")
    assert got["edited"].to_list() == [False, False]
    assert got["edits"].to_list() == ["", ""]


# --------------------------------------------------------------------------- #
# the evidence beside an override
# --------------------------------------------------------------------------- #
# An override is supposed to be made on the record, so the shaping of that record is worth pinning: a
# season split across two teams that averages instead of adding, or a quick-set button offering the
# number it is already set to, are both failures that render perfectly well.
def long_record() -> pl.DataFrame:
    """One player, five seasons, with 2023 split across two teams -- the case that averages wrongly."""
    return pl.DataFrame({
        "season": [2021, 2022, 2023, 2023, 2024, 2025],
        "player_id": ["p1"] * 6,
        "team": ["ARI", "ARI", "ARI", "BUF", "BUF", "BUF"],
        "metric": ["target_share"] * 6,
        "num": [40.0, 90.0, 12.0, 60.0, 130.0, 120.0],
        "n": [400.0, 450.0, 100.0, 300.0, 520.0, 500.0],
        "value": [0.10, 0.20, 0.12, 0.20, 0.25, 0.24],
    })


def test_a_split_season_combines_by_the_totals_rather_than_by_the_mean() -> None:
    got = ui.collapse_seasons(long_record())
    assert got.height == 5
    row = got.filter(pl.col("season") == 2023).row(0, named=True)
    assert row["n"] == 400.0 and row["num"] == 72.0
    assert row["value"] == pytest.approx(0.18)          # not (0.12 + 0.20) / 2
    # and the seasons nobody split are left exactly as they were
    assert got.filter(pl.col("season") == 2025)["value"][0] == 0.24


def test_a_record_with_nothing_duplicated_is_returned_untouched() -> None:
    one = long_record().filter(pl.col("team") == "BUF")
    assert ui.collapse_seasons(one).equals(one.sort(["player_id", "season"]))


def test_the_record_pivots_to_one_row_with_a_column_per_season_and_a_trend() -> None:
    got = ui.season_wide(long_record(), seasons=(2021, 2022, 2023, 2024, 2025))
    assert got.height == 1
    assert got["2023"][0] == pytest.approx(0.18)
    assert got["record"][0].to_list() == pytest.approx([0.10, 0.20, 0.18, 0.25, 0.24])


def test_a_season_the_player_missed_is_zero_in_the_trend_rather_than_a_gap() -> None:
    """`LineChartColumn` draws a list, and a null inside one is not a line."""
    got = ui.season_wide(long_record().filter(pl.col("season") != 2022),
                         seasons=(2021, 2022, 2023, 2024, 2025))
    assert got["record"][0].to_list()[1] == 0.0
    assert got["2022"][0] is None                       # the column itself still tells the truth


def test_the_percentile_is_the_share_at_or_below_and_survives_a_missing_number() -> None:
    col = pl.Series([0.05, 0.10, 0.20, 0.30, None])
    assert ui.percentile_of(col, 0.20) == pytest.approx(0.75)   # at or below, so his own row counts
    assert ui.percentile_of(col, 0.31) == 1.0
    assert ui.percentile_of(col, 0.01) == 0.0
    assert ui.percentile_of(col, None) is None
    assert ui.percentile_of(pl.Series([None, None], dtype=pl.Float64), 0.2) is None


def test_the_quick_choices_for_a_share_are_the_numbers_already_on_screen() -> None:
    row = {"estimate": 0.20, "obs": 0.26, "n": 400.0, "prior": 0.17}
    got = dict(ui.quick_choices("target_share", row, {"median": 0.12, "p90": 0.24}))
    assert got == {"his own record": 0.26, "his job's prior": 0.17,
                   "position median": 0.12, "top of position": 0.24}


def test_a_player_with_no_record_of_his_own_is_not_offered_one() -> None:
    got = dict(ui.quick_choices("target_share", {"estimate": 0.17, "obs": None, "n": 0.0,
                                                 "prior": 0.17}))
    assert "his own record" not in got
    assert got == {}                                    # his prior *is* the estimate, so no button


def test_availability_offers_the_full_slate_first_because_that_is_the_argument() -> None:
    got = ui.quick_choices(ui.GAMES_METRIC, {"estimate": 14.2, "obs": 15.5, "n": 4.0})
    assert [name for name, _ in got][0] == "every game"
    assert dict(got)["every game"] == 17.0
    assert dict(got)["his own average"] == 15.5


def test_a_button_is_never_offered_for_the_number_already_in_use() -> None:
    """A quick set that changes nothing is furniture, and furniture in a row of buttons is a trap."""
    got = dict(ui.quick_choices(ui.GAMES_METRIC, {"estimate": 17.0, "obs": 17.0, "n": 5.0}))
    assert "every game" not in got and "his own average" not in got


# --------------------------------------------------------------------------- #
# a whole population at once
# --------------------------------------------------------------------------- #
# A bulk edit is one override per player rather than a bulk object, so what a test can pin is the
# translation: the base has to be the *engine's* number and not what the man is currently running on,
# because an override recorded against a mid-session value cannot be dropped back to the engine.
def preview() -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": ["p1", "p2"],
        "player": ["A", "B"],
        "engine": [14.2, 12.0],
        "now": [15.0, 12.0],
        "after": [16.5, 16.5],
        "change": [1.5, 4.5],
    })


def test_a_bulk_edit_is_one_override_per_player_against_the_engines_own_number() -> None:
    got = ui.bulk_edits(preview(), ui.GAMES_METRIC, "set", 16.5, why="healthy starters")
    assert [o.key for o in got] == ["p1", "p2"]
    assert {o.level for o in got} == {"player"}
    assert all(o.field == ui.GAMES_METRIC and o.mode == "set" and o.value == 16.5 for o in got)
    assert [o.base for o in got] == [14.2, 12.0]          # the engine's, not `now`
    assert {o.note for o in got} == {"healthy starters"}


def test_a_bulk_multiply_carries_the_multiplier_rather_than_the_result() -> None:
    """`multiply` has to stay a multiplier: baked into a number it would stop tracking the engine."""
    got = ui.bulk_edits(preview(), "target_share", "multiply", 0.9)
    assert all(o.mode == "multiply" and o.value == 0.9 for o in got)
    assert ui.bulk_edits(pl.DataFrame(), "target_share", "multiply", 0.9) == []


def test_every_roster_status_the_engine_prices_has_a_plain_english_meaning() -> None:
    """The availability factors are asserted, so the page has to be able to say what each code is."""
    from src.config import Settings

    missing = [s for s in Settings().status_availability if s not in ui.STATUS_MEANING]
    assert not missing


# --------------------------------------------------------------------------- #
# how a number is printed
# --------------------------------------------------------------------------- #
# One place decides decimals and headings for every table in the app, which is the only way two pages
# cannot show the same number to different precision. The failure is silent -- a share printed to two
# decimals reads as 0.24 for everybody between 0.235 and 0.245 -- so the defaults are pinned here.
def test_the_decimals_a_column_is_worth_come_from_one_place() -> None:
    assert ui.digits_for("fantasy_points") == 1          # the second decimal is noise
    assert ui.digits_for("points_per_game") == 2
    assert ui.digits_for("passing_yards") == 0           # a yard is a whole number to read
    assert ui.digits_for("target_share") == 4            # shown as a percent, so kept to four
    assert ui.digits_for("expected_games") == 1
    assert ui.digits_for("nothing_in_particular") == 2
    assert ui.digits_for("nothing_in_particular", 3) == 3


def test_a_share_is_printed_as_a_percentage_and_a_ratio_around_one_is_not() -> None:
    df = pl.DataFrame({"target_share": [0.24, 0.10], "context_factor": [1.05, 0.94],
                       "pace_factor": [2.0, 1.0]})
    cfg = ui.auto_config(df)
    assert cfg["target_share"]["type_config"]["format"] == "percent"
    # a factor is a multiplier, and 105% is a different number from x1.05
    assert cfg["context_factor"]["type_config"]["format"] == "%.2f"
    # and a percent-ish name whose values leave 0-1 is not a share at all
    assert cfg["pace_factor"]["type_config"]["format"] == "%.2f"


def test_every_column_gets_a_heading_a_reader_would_say() -> None:
    df = pl.DataFrame({"own_weight": [0.4], "expected_games": [16.0], "player_id": ["p1"],
                       "startable": [True], "player": ["A"]})
    cfg = ui.auto_config(df)
    assert cfg["own_weight"]["label"] == "weight on his own record"
    assert cfg["expected_games"]["label"] == "projected games"
    assert cfg["startable"]["type_config"]["type"] == "checkbox"
    assert "player_id" not in cfg                        # an id is not a number anybody reads
    assert ui.label("p_play") == "chance he plays"
    assert ui.label("receptions") == "receptions"        # no dictionary entry needed


def test_rounding_for_display_follows_the_column_and_leaves_ids_alone() -> None:
    """A column asking for *more* decimals than the table's default has to keep them.

    The two-pass version of this rounded everything to the default first, so a share inside a table
    defaulting to two decimals came out at 0.24 -- and 0.2431 and 0.2449 both read as 24.00%.
    """
    df = pl.DataFrame({"player_id": ["p1"], "fantasy_points": [283.456], "target_share": [0.24312],
                       "games": [16.44]})
    got = ui.rounded(df, 2, {c: ui.digits_for(c) for c in df.columns})
    assert got["fantasy_points"][0] == 283.5
    assert got["target_share"][0] == 0.2431
    assert got["games"][0] == 16.4
    assert got["player_id"][0] == "p1"
    # and with nothing said per column, the default applies to every float
    assert ui.rounded(df, 1)["target_share"][0] == 0.2


# --------------------------------------------------------------------------- #
# what a knob moves, before anything is run
# --------------------------------------------------------------------------- #
# The What-if tab tells the user what a number does *before* it does it -- whether raising it takes
# something from a teammate or from nobody. That claim comes from the pool table, not from prose, so it
# is pinned here: a share of an exclusive pool is zero-sum, a participation term is not, and getting the
# two the wrong way round would put a confident sentence under a wrong number.
def test_a_knob_knows_whether_it_is_zero_sum() -> None:
    assert ui.pool_for_field("target_share") == "targets"
    assert ui.pool_for_field("dropback_share") == "dropbacks"     # filled by queue, but still divided
    assert ui.pool_for_field("snap_share") is None                # nothing is taken from anybody
    assert ui.shared_pool_for_field("snap_share") == "offense_snaps"
    assert ui.shared_pool_for_field("route_participation") == "routes"
    assert ui.shared_pool_for_field("target_share") is None

    assert ui.knob_kind("expected_games") == "availability"
    assert ui.knob_kind("target_share") == "pool"
    assert ui.knob_kind("dropback_share") == "pool"               # not presence: a QB2 is not a co-starter
    assert ui.knob_kind("yards_per_target") == "rate"
    assert ui.knob_kind("nothing_the_engine_has") == "rate"       # an unknown number is nobody else's

    # playing time beats the shape test, which is the whole point of the exception: both snap shares
    # divide a pool that is not exclusive, so `pool_for_field` is None and the shared pool is the snap
    # pool -- read off shape alone they would be `presence`, "nobody else moves", which is what the
    # engine stopped doing when it started spending them on every other claim the man makes.
    assert ui.knob_kind("snap_share") == "playing time"
    assert ui.knob_kind("qb_snap_share") == "playing time"
    assert ui.shared_pool_for_field("qb_snap_share") == "offense_snaps"
    assert ui.pool_for_field("qb_snap_share") is None
    # and a participation term that is still evidence keeps the old reading
    assert ui.knob_kind("route_participation") == "presence"

    for kind in ("availability", "playing time", "pool", "presence", "rate"):
        assert ui.KIND_NOTE[kind] and ui.KIND_ICON[kind]


def test_every_pool_the_engine_divides_can_be_looked_at() -> None:
    """A pool the engine splits and the page cannot show is a room nobody can audit."""
    from src.model import opportunity

    offered = dict(ui.CONTRIBUTION_POOLS)
    assert set(offered) == {p.name for p in opportunity.POOLS}
    assert all(name for name in offered.values())


def test_every_knob_offered_is_one_the_engine_takes_and_is_explained() -> None:
    from src.model import overrides as ov

    for position, fields in ui.KNOBS.items():
        assert len(set(fields)) == len(fields), position
        for field in fields:
            assert field in ov.PLAYER_FIELDS, f"{position}: {field} is not an override the engine takes"
            assert ui.KNOB_HELP.get(field), f"{position}: nothing is said about {field}"
            assert ui.knob_option(field).endswith(ui.label(field))
        assert ui.knobs_for(position) == fields
    # an unlisted position still gets a full set rather than an empty selectbox
    assert ui.knobs_for("K") == tuple(ov.PLAYER_FIELDS)
    assert ui.knobs_for(None) == tuple(ov.PLAYER_FIELDS)


def test_the_reading_names_the_transfer_and_does_not_claim_one_that_is_not_there() -> None:
    """The sentence under the four cards, which is the only part of the tab that interprets."""
    transfer = ui._reading("target_share", "A", 23.0, -20.4, 22, normalising=True)
    assert "**A +23.0 points**" in transfer and "22 teammates" in transfer
    assert "the offence **+2.6**" in transfer
    assert "transfer inside the room" in transfer

    # the same knob when the room barely pays for it: a claim of new production, and said as one
    assert "zero-sum" in ui._reading("target_share", "A", 23.0, -2.0, 3, normalising=True)
    # and with normalisation off nothing is taken back at all
    assert "normalisation off" in ui._reading("target_share", "A", 23.0, 0.0, 0, normalising=False)

    assert "availability scales" in ui._reading("expected_games", "A", -75.6, 75.2, 30,
                                               normalising=True)
    assert "takes nothing from anybody" in ui._reading("route_participation", "A", 4.0, 0.0, 0,
                                                       normalising=True)
    # playing time does take it from somebody, and the sentence has to say so rather than reassure
    moved = ui._reading("snap_share", "A", 15.5, -11.0, 12, normalising=True)
    assert "scales every claim he makes at once" in moved
    assert "on the field instead of" in moved
    assert "takes nothing from anybody" not in moved
    assert "check the room" in ui._reading("qb_snap_share", "A", 21.3, 0.0, 0, normalising=True)
    alone = ui._reading("yards_per_target", "A", 10.0, 0.0, 0, normalising=True)
    assert "nobody else's line rides on it" in alone
    assert "his room" not in alone                       # no room panel, so no sentence about one
    assert "through a pool it feeds" in ui._reading("yards_per_target", "A", 10.0, 1.0, 1,
                                                    normalising=True)
    assert "across 1 teammate ·" in ui._reading("yards_per_target", "A", 10.0, 1.0, 1,
                                                normalising=True)   # one man, said in the singular

    nothing = ui._reading("target_share", "A", 0.0, 0.0, 0, normalising=True)
    assert nothing.startswith("Nothing moved")
    assert ui.label("target_share") in nothing           # and it names what was tried


# --------------------------------------------------------------------------- #
# what a number looks like when it is drawn rather than tabulated
# --------------------------------------------------------------------------- #
# The tiles and the meters are the redesign's answer to a forty-four-column table, and they are HTML
# built in Python, so they can be wrong in ways a rendering test never sees: a fill past the end of its
# track, a tick drawn off the bar, a missing number printed as 0.0 instead of nothing. The markup is
# checked here; that it reaches a page is `scripts/smoke_app.py`.
def test_a_number_nobody_has_is_a_dash_rather_than_a_zero() -> None:
    assert ui.fmt_num(None) == "—"
    assert ui.fmt_num(float("nan")) == "—"
    assert ui.fmt_num(0.0) == "0.0"                      # a real zero is a number and is printed
    assert ui.fmt_num(0.2431, 3) == "0.243"
    # the digits are of the number itself, so a share asked for to three decimals reads to one as a
    # percentage -- 0.243 and 24.3% are the same claim about the same estimate
    assert ui.fmt_num(0.2431, 3, percent=True) == "24.3%"
    assert ui.fmt_num(0.2431, 1, percent=True) == "24%"
    assert ui.fmt_num(283.4, 0, suffix=" pts") == "283 pts"
    assert ui.fmt_num("out") == "out"                    # a status is not a number to format


def test_a_tile_carries_its_own_name_and_survives_a_missing_value() -> None:
    got = ui.tile_html("fantasy points", 283.46, "17.4 per game", digits=1, highlight=True)
    assert "fantasy points" in got and "283.5" in got and "17.4 per game" in got
    assert "nf-tile" in got
    assert "—" in ui.tile_html("vs last season", None)
    # a difference is only readable with its sign on it
    assert "+12.4" in ui.tile_html("vs avg starter", 12.4, signed=True)
    assert "-12.4" in ui.tile_html("vs avg starter", -12.4, signed=True)
    # and a name a player supplied cannot close the tag it is printed inside
    assert "<script>" not in ui.tile_html("<script>x</script>", 1.0)


def test_a_meters_fill_and_ticks_stay_on_the_track() -> None:
    got = ui.meter_html("target share", 0.24, maximum=0.4, digits=3,
                        marks=(("his record", 0.30), ("his job", 0.12)))
    assert "60.0%" in got                                # 0.24 of 0.4 of the track
    assert "0.240" in got and "target share" in got
    assert got.count("nf-mark") == 2                     # both ticks drawn
    assert "his record" in got and "his job" in got
    # a number over the axis is clamped to the end rather than drawn past it
    assert "100.0%" in ui.meter_html("m", 0.9, maximum=0.4)
    # a tick nobody has is not drawn at all, and neither is a fill
    lonely = ui.meter_html("m", None, maximum=1.0, marks=(("his record", None),))
    assert "nf-mark" not in lonely and "—" in lonely


def test_a_meters_axis_is_the_share_scale_for_a_share_and_headroom_for_a_rate() -> None:
    assert ui.rating_axis({"obs": 0.31, "prior": 0.12, "used": 0.24, "applied": 0.24}) == 1.0
    # a rate has no natural top, so the track is the biggest number on the row plus room to see it move
    assert ui.rating_axis({"obs": 4.6, "prior": 4.2, "used": 4.4}) == pytest.approx(5.29)
    assert ui.rating_axis({"obs": None, "prior": None, "used": None}) is None


def test_one_estimate_is_one_row_however_many_frames_carry_it() -> None:
    """The same metric from two frames is one meter, and the share is the heading kept.

    `dropback_share` is both a participation metric and a pool share. Two rows means two meters for one
    number, two ✎ writing the same override, and -- because the widget keys are built from the metric --
    a duplicate-key error the moment a quarterback is opened.
    """
    both = pl.DataFrame({
        "kind": ["availability", "share", "rate"],
        "metric": ["dropback_share", "dropback_share", "yards_per_attempt"],
        "used": [0.99, 0.99, 7.4],
    })
    got = ui.one_row_per_metric(both)
    assert got.height == 2
    assert got.filter(pl.col("metric") == "dropback_share")["kind"].to_list() == ["share"]
    assert ui.one_row_per_metric(pl.DataFrame()).is_empty()
    # and the overlap this exists for is real rather than assumed
    assert set(roster.PARTICIPATION_METRICS) & set(opportunity.SHARE_METRICS)


# --------------------------------------------------------------------------- #
# keeping the work
# --------------------------------------------------------------------------- #
# Session state does not survive a browser reload or a server restart, and a pass over the league is
# hours of typing. So every edit funnels through `set_live`, which writes, and every new session opens on
# what was written. These tests are the ones standing between the user and losing an afternoon.
def test_an_edit_is_on_disk_the_moment_it_is_made() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    assert overrides.names() == [overrides.WORKING]
    back = overrides.load(overrides.WORKING)
    assert [o.label for o in back.items] == ["p1 target_share = 0.3"]


def test_the_baseline_is_not_written_because_there_is_nothing_to_lose() -> None:
    ui.set_live(Scenario())
    assert overrides.names() == []


def test_a_named_scenario_keeps_its_own_name_rather_than_becoming_the_working_one() -> None:
    ui.set_live(Scenario(name="my draft board").set(
        Override("team", "BUF", "plays", "multiply", 1.05)))
    assert overrides.names() == ["my draft board"]
    assert ui.live().name == "my draft board"


def test_a_new_session_opens_on_the_work_that_was_left() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    ui._state().pop(ui.LIVE, None)                     # what a browser reload amounts to
    assert [o.label for o in ui.live().items] == ["p1 target_share = 0.3"]
    assert ui._state()[ui.RESTORED] == overrides.WORKING


def test_a_session_with_nothing_saved_opens_on_the_baseline() -> None:
    ui._state().pop(ui.LIVE, None)
    ui._state().pop(ui.RESTORED, None)
    assert ui.live().is_baseline


def test_the_sidebar_says_when_it_last_wrote() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    stamp = ui._state()[ui.SAVED_AT]
    assert ui._clock(stamp) == stamp[11:16] and len(ui._clock(stamp)) == 5


# --------------------------------------------------------------------------- #
# how far through the league
# --------------------------------------------------------------------------- #
# The coverage table is the one thing in the app about the user rather than the season, and it is derived
# from the edits so that there is nothing extra to keep in step. What it must never do is call a team
# reviewed because a *different* team was edited.
COVER_BOARD = pl.DataFrame({
    "player_id": ["p1", "p2", "p3"],
    "team": ["BUF", "BUF", "KC"],
})


def _coverage(items: tuple[Override, ...]) -> pl.DataFrame:
    v = ui.View(payload=Scenario().set(*items).content_json())
    board = ui.board
    ui.board = lambda _v: COVER_BOARD                  # the frame, not the engine: this is bookkeeping
    try:
        return ui.coverage(v)
    finally:
        ui.board = board


def test_every_team_is_listed_and_none_is_reviewed_before_anything_is_edited() -> None:
    got = _coverage(())
    assert got.height == 32
    assert not bool(got["reviewed"].any())
    assert got["edits"].sum() == 0


def test_a_players_edit_counts_against_the_team_he_is_on() -> None:
    got = _coverage((Override("player", "p1", "target_share", "set", 0.3),
                     Override("player", "p2", "target_share", "set", 0.2)))
    buf = got.filter(pl.col("team") == "BUF").row(0, named=True)
    assert buf["reviewed"] and buf["edits"] == 2 and buf["players_edited"] == 2
    assert not bool(got.filter(pl.col("team") == "KC").row(0, named=True)["reviewed"])


def test_a_team_edit_and_a_week_edit_are_counted_as_what_they_are() -> None:
    got = _coverage((Override("team", "KC", "plays", "multiply", 1.1),
                     Override("player", "p3", "target_share", "set", 0.3, week=4)))
    kc = got.filter(pl.col("team") == "KC").row(0, named=True)
    assert kc["edits"] == 2 and kc["team_edits"] == 1 and kc["week_edits"] == 1
    assert kc["players_edited"] == 1


def test_an_edit_on_a_player_nobody_has_is_not_counted_anywhere() -> None:
    got = _coverage((Override("player", "nobody", "target_share", "set", 0.3),))
    assert got["edits"].sum() == 0


# --------------------------------------------------------------------------- #
# the data under the projection, and whether the server has it
# --------------------------------------------------------------------------- #
# `st.cache_data` lives in the server process, so the nightly update -- a different process -- cannot
# reach it: an app that was already open keeps serving the frames it read at startup. The sidebar says
# so by comparing the stamp the update wrote against when this process began, and the one thing that
# must not happen is a bad or absent stamp taking the sidebar down with it.
def test_no_update_stamp_yet_is_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ui, "UPDATE_STAMP", tmp_path / "never_written.json")
    assert ui.last_update() == {}


def test_an_unreadable_stamp_is_not_an_error(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "last_update.json"
    bad.write_text("half a file", encoding="utf-8")
    monkeypatch.setattr(ui, "UPDATE_STAMP", bad)
    assert ui.last_update() == {}


def test_the_stamp_is_read_fresh_every_time_rather_than_cached(tmp_path, monkeypatch) -> None:
    """The whole point is noticing a file that changed *after* the caches were filled, so a cached
    reader here would report exactly the staleness it exists to detect."""
    stamp = tmp_path / "last_update.json"
    monkeypatch.setattr(ui, "UPDATE_STAMP", stamp)
    stamp.write_text('{"finished": "2026-09-02T06:30:01", "mode": "light"}', encoding="utf-8")
    assert ui.last_update()["mode"] == "light"
    stamp.write_text('{"finished": "2026-09-09T04:12:00", "mode": "full"}', encoding="utf-8")
    assert ui.last_update()["mode"] == "full"


# --------------------------------------------------------------------------- #
# taking an override back off
# --------------------------------------------------------------------------- #
# `drop_all` is one line over `Scenario.clear_keys`, and the line is the part worth pinning: it goes
# through `set_live`, so the drop is autosaved exactly like the edit was. A drop that only lived in
# session state would come back on the next browser reload, which is the one failure mode a user would
# read as the tool ignoring them.
def test_dropping_a_player_takes_every_edit_on_him_and_reports_how_many() -> None:
    ui.edit(
        Override("player", "p1", "target_share", "set", 0.30),
        Override("player", "p1", "target_share", "set", 0.25, week=4),
        Override("player", "p1", "catch_rate", "set", 0.70),
        Override("player", "p2", "target_share", "set", 0.10),
    )
    assert ui.drop_all("player", "p1") == 3
    assert {o.key for o in ui.live().items} == {"p2"}


def test_dropping_several_at_once_is_one_write() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30),
            Override("player", "p2", "target_share", "set", 0.10),
            Override("team", "PHI", "targets", "multiply", 1.1))
    assert ui.drop_all("player", "p1", "p2") == 2
    assert [(o.level, o.key) for o in ui.live().items] == [("team", "PHI")]


def test_dropping_nobody_writes_nothing() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    was = ui.live()
    assert ui.drop_all("player", "p9") == 0
    assert ui.live() is was


def test_a_dropped_player_is_gone_from_the_file_and_not_only_from_the_session() -> None:
    """Autosave is the funnel, so the scenario on disk has to agree with the one on screen."""
    ui.edit(Override("player", "p1", "target_share", "set", 0.30),
            Override("player", "p2", "target_share", "set", 0.10))
    ui.drop_all("player", "p1")
    assert {o.key for o in overrides.load(ui.live().name).items} == {"p2"}
