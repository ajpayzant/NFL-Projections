"""Run every page in a real Streamlit session and fail on any exception.

`streamlit run` is not scriptable, but `st.testing.v1.AppTest` is: it runs a page with a genuine
session, so the widgets hold state, `st.rerun()` reruns, and the scenario in `st.session_state`
behaves the way it does in a browser. That is the difference from executing a page as a bare script --
bare mode gives every widget its default and never reruns, so it cannot test an edit at all.

Sixteen passes:

1. **Baseline** -- every page with no edits. Catches the ordinary breakage: a column that does not
   exist, a bad `column_config`, two frames joined the wrong way round.
2. **Under a scenario** -- the same pages with a team edit, a player edit and `k_scale` live, so the
   provenance table, the diff against the baseline and the "what the edits did" panels all render.
3. **Interactions** -- a sidebar toggle and a per-knob reset actually clicked, asserting the scenario
   changed. This is the part that proves the override layer is wired to the widgets rather than merely
   importable.
4. **Ranges** -- the Monte Carlo switch clicked on Home and on the player page, asserting the range
   columns actually arrive. The simulation is on demand, so a page that never had the switch flipped is
   a page whose range code has never run.
5. **Interface** -- the reading surfaces: the board switched to a football line, the depth chart drawn
   and its depth changed, and the knob beside a player's own ratings turned. A column set nobody
   switched to and a chart nobody drew are things that break without a page failing to render.
6. **List and panel** -- the reading surface itself: a narrow ranked list, the panel of the man beside
   it, and every estimate drawn as a meter with its two ticks. All three break without an exception --
   a list that quietly grows back to forty-four columns, a panel that draws nothing, a ✎ that opens
   on an empty popover -- so the ✎ is clicked and followed into the scenario.
7. **Evidence** -- the record beside every override point, and a one-click set driven from it. An
   override made on nothing is the failure this pass exists to catch: the panel has to name the
   engine's number, the player's own record and his job's average, and a named button has to write the
   value it printed.
8. **Depth** -- somebody moved on a chart, from the team's own editor and from the one man's page. A slot
   is the only override that is an ordering rather than a number, so it is the only one that can look
   applied while changing nothing: the pass drives the move, then checks the scenario holds it, the page
   reads him back at the slot he was sent to, and the ledger reports it as a move in slots.
9. **Off the roster** -- a man removed from the team he is listed on, driven from the depth chart. It is
   the one override that takes a *row* out of the projection rather than changing a number in one, so
   both of its failures are quiet: a release that does nothing leaves the chart reading a man the team
   has cut, and a release nothing reports leaves a scenario with a hole in it. The pass checks the room
   closes up behind him, the ledger still names a player no run has a row for, and the reset puts him
   back.
10. **Population** -- the two edits that are not one number at one point: a filtered bulk edit written
   across a whole position, and a return date turned into a games count on the availability page. Both
   write many overrides or convert a control into a different unit, so both can look right on screen
   while writing nothing.
11. **Each room's sheet** -- all four of a team's rooms as editable tables, and the evidence join switched
   both on and off. A room is a different frame built from a different list of fields with a different
   `column_config`, so a list naming a column the frame does not carry is an exception in a room the
   baseline pass never opened -- it only ever visits the default one.
12. **The batched editor** -- one player's whole rating card, drawn for every column set, scoped to two
   chosen weeks, and opened over another page from the session flag every ✎ in the app raises. Nothing is
   typed, because `AppTest` cannot type into a `data_editor` and `tests/test_app_ui.py` pins what a typed
   cell becomes; what is caught here is a field list a frame does not carry and an editor that renders
   only on the page it was written for.
13. **What if** -- one candidate edit run through the engine before it is written down, on the team page.
   The pass drives the control and then asserts the sign structure a share edit must have -- he gains,
   his room loses, the offence barely moves -- because a preview that quietly did arithmetic on the page
   instead of running the season again would render perfectly and be wrong about the only thing it is
   there to say.
14. **One game** -- the Week page picked through to a fixture, both box scores read off the screen, and
   the per-game editing surfaces rendered for every column set. A per-game edit is the one edit whose
   grain can be wrong without anything failing -- a page that wrote a season-wide override from a
   week's sheet would look identical -- so the pass also checks that the fields offered are per-game
   fields, and that the box score reconciles to the offence it was divided out of.
15. **The league** -- the projected records, the arithmetic that makes them readable, and the coverage
   table. Expected wins have to add to exactly half the games played: a filtered or double-counted frame
   would look perfectly plausible on screen, since one team's 10.5 reads the same whether the league adds
   to 272 or 300. The same pass edits one team and checks that the record moved and that exactly that
   team reads as reviewed.
16. **The room sheet** -- the surface a team is actually worked through: the record joined onto the
   editable sheet, what the room's shares add up to against the pool that has to hold them, and the
   buttons that walk the league. The sums are the part worth a gate -- they are arithmetic about the
   engine's arithmetic, and a version summed over the wrong player set reads as a plausible number rather
   than as a failure -- so they are tied to the pool audit the same page draws.

    python scripts/smoke_app.py
"""

from __future__ import annotations

import gc
import re
import sys
import tempfile
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import polars as pl                                   # noqa: E402
from streamlit.testing.v1 import AppTest              # noqa: E402

import ui                                            # noqa: E402
from src.model import opportunity, overrides          # noqa: E402
from src.model.overrides import Override, Scenario     # noqa: E402

# Named rather than spelled out at every call site, because the sidebar order is a numbered filename and
# inserting a page renames several of them.
HOME = "app/Home.py"
LEAGUE = "app/pages/1_League.py"
TEAM = "app/pages/2_Team.py"
PLAYER = "app/pages/3_Player.py"
WEEKLY = "app/pages/4_Week.py"
AVAILABILITY = "app/pages/5_Availability.py"
HISTORY = "app/pages/6_History.py"
EDITS = "app/pages/7_Edits.py"
EXPORTS = "app/pages/8_Exports.py"

PAGES = [HOME, LEAGUE, TEAM, PLAYER, WEEKLY, AVAILABILITY, HISTORY, EDITS, EXPORTS]
LIVE = "scenario"

# Every edit is written to disk as it is made, and a new session opens on what was left -- which is the
# point of the feature in a browser and poison in a harness. Without these two lines the sixth pass opens
# on the fifth pass's overrides, every check that asserts "one edit was made" fails on six of them, and
# the run leaves its scrap in the user's own scenario folder. So the folder is a throwaway, and every page
# the harness opens is *handed* a scenario rather than restoring one.
overrides.SCENARIOS = Path(tempfile.mkdtemp(prefix="smoke-scenarios-"))


def run_page(page: str, scenario: Scenario | None = None) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    at.session_state[LIVE] = scenario if scenario is not None else Scenario()
    return at.run()


def check(what: str, scenario: Scenario | None) -> int:
    bad = 0
    for page in PAGES:
        at = run_page(page, scenario)
        if at.exception:
            bad += 1
            print(f"  FAIL  {page}")
            for e in at.exception:
                print(f"        {e.value}")
        else:
            print(f"  ok    {page}  {len(at.dataframe)} tables, {len(at.metric)} metrics")
    print(f"{what}: {len(PAGES) - bad}/{len(PAGES)} pages ran")
    return bad


def demo() -> tuple[Scenario, dict]:
    """A scenario the pages will actually show: the top receiver on the team they open on."""
    board = overrides.run(Scenario()).board
    first = sorted(board["team"].unique().to_list())[0]
    top = board.filter((board["position"] == "WR") & (board["team"] == first)).row(0, named=True)
    sc = overrides.demo_scenario(top["player_id"], top["team"]).patch_league(k_scale=2.0)
    return sc, top


def interactions(sc: Scenario) -> int:
    bad = 0

    at = run_page(HOME)
    at.sidebar.toggle[0].set_value(False).run()
    got = at.session_state[LIVE]
    if got.league.get("normalize_pools") is not False:
        bad += 1
        print(f"  FAIL  the sidebar toggle did not reach the scenario: {got.league}")
    else:
        print(f"  ok    normalisation off is recorded, digest {got.digest}")

    at = run_page(EDITS, sc)
    drops = [b for b in at.button if (b.key or "").startswith("drop:")]
    if not drops:
        bad += 1
        print("  FAIL  no per-knob reset button rendered")
    else:
        drops[0].click().run()
        left = at.session_state[LIVE]
        if len(left.items) != len(sc.items) - 1:
            bad += 1
            print(f"  FAIL  reset took {len(sc.items) - len(left.items)} edits, not 1")
        else:
            print(f"  ok    one knob reset, {len(left.items)} edits left, digest {left.digest}")

    # Both scopes of a team edit are on the Team page now, and the season-wide one is behind a scope
    # radio -- so the radio is driven rather than assumed, because a knob nobody can reach is the same
    # failure as a knob that does not write.
    at = run_page(TEAM)
    scopes = [r for r in at.radio if (r.key or "").startswith("env:scope:")]
    if not scopes:
        bad += 1
        print("  FAIL  Team: no scope control on the volume tab")
    else:
        at = scopes[0].set_value("every week at once").run()
        boxes = [n for n in at.number_input if "allweeks" in (n.key or "") and ":targets@" in n.key]
        if not boxes:
            bad += 1
            print("  FAIL  no all-weeks team knob rendered")
        else:
            boxes[0].set_value(1.25).run()
            got = [o for o in at.session_state[LIVE].items if o.field == "targets"]
            if len(got) != 1 or got[0].mode != "multiply" or got[0].value != 1.25:
                bad += 1
                print(f"  FAIL  typing a multiplier produced {[o.label for o in got]}")
            else:
                print(f"  ok    a typed multiplier becomes an edit: {got[0].label}")

    # starting again is the sidebar's, on every page, rather than a button on one of them
    at = run_page(EDITS, sc)
    clears = [b for b in at.button if (b.key or "") == "scenario:new"]
    if not clears:
        bad += 1
        print("  FAIL  the sidebar has no way back to the engine's own answer")
    else:
        clears[0].click().run()
        if not at.session_state[LIVE].is_baseline:
            bad += 1
            print("  FAIL  starting a new scenario left edits behind")
        else:
            print("  ok    a new scenario goes back to the engine's own answer")
    return bad


def screen(at: AppTest) -> str:
    """Everything readable on the page as one string.

    The redesign moved most numbers out of `st.metric` and into drawn tiles and meters, which reach the
    page as `st.html`. A check that only reads metric labels would pass on a page that had stopped
    showing the number, so it reads the text, the captions and the drawn markup together.
    """
    parts: list[str] = []
    for m in at.metric:
        parts += [str(m.label), str(m.value)]
    parts += [str(e.value) for e in at.markdown]
    parts += [str(e.value) for e in at.caption]
    # `st.html` arrives as an element the test tree has no wrapper for, so the markup comes off the proto:
    # `body` on this version, and the loop keeps a rename from silently emptying every check that reads it
    for e in at.get("html"):
        parts += [str(getattr(e.proto, f)) for f in ("body", "value", "html")
                  if hasattr(e.proto, f)]
    for d in at.dataframe:
        parts += list(d.proto.column_order)
        if hasattr(d.value, "columns"):
            parts += [str(c) for c in d.value.columns]
    return "\n".join(parts)


TILE = re.compile(
    r'<div class="nf-tile[^"]*">'
    r'<span class="k">(?P<name>.*?)</span>'
    r'<span class="v[^"]*">(?P<value>.*?)</span>'
    r'(?:<span class=.s.>(?P<sub>.*?)</span>)?'
    r"</div>"
)


def drawn_tiles(at: AppTest) -> list[dict]:
    """The tiles on screen as `{name, value, sub}`, in the order they were drawn.

    A headline number used to be an `st.metric` the test tree hands back with a label and a value. It is
    now markup, so the checks that assert *what a number did* have to read it out of the markup — and
    reading it with the same regex everywhere means a change to `tile_html` breaks one helper rather
    than five checks.
    """
    out: list[dict] = []
    for chunk in screen(at).split("\n"):
        for m in TILE.finditer(chunk):
            out.append({k: unescape(v or "").strip() for k, v in m.groupdict().items()})
    return out


def tile_named(items: list[dict], prefix: str) -> dict | None:
    """The first tile whose name starts with this, matched loosely on case."""
    low = prefix.lower()
    return next((t for t in items if t["name"].lower().startswith(low)), None)


def switch(at: AppTest, label: str):
    """The one control that has been a radio, a set of pills and a segmented control across releases."""
    for group in (at.radio, at.segmented_control, at.pills, at.selectbox):
        got = [w for w in group if w.label == label]
        if got:
            return got[0]
    return None


def ranges() -> int:
    """Flip the on-demand Monte Carlo switch and check the ranges reach the screen.

    The draws are wound down to the smallest option first: this is checking that the wiring produces a
    floor and a ceiling, and ten thousand seasons would prove exactly the same thing far more slowly.
    """
    bad = 0
    for page in (HOME, PLAYER):
        at = run_page(page)
        toggles = [t for t in at.toggle if (t.key or "").endswith(":on")]
        if not toggles:
            bad += 1
            print(f"  FAIL  {page}: no ranges switch rendered")
            continue
        sliders = [s for s in at.select_slider if (s.key or "").endswith(":draws")]
        if sliders:
            sliders[0].set_value(2_000)
        at = toggles[0].set_value(True).run()
        if at.exception:
            bad += 1
            print(f"  FAIL  {page} with ranges on")
            for e in at.exception:
                print(f"        {e.value}")
            continue
        # the floor is a drawn tile on the player page and a column on the board, so both are read out
        # of everything on screen rather than out of `st.metric` alone
        want = ("floor · P5" if "Player" in page else "volatility")
        if want not in screen(at):
            bad += 1
            print(f"  FAIL  {page}: ranges on but no {want!r} on screen")
        else:
            print(f"  ok    {page}  ranges on, {len(at.dataframe)} tables, {len(at.metric)} metrics")
    return bad


def columns_of(at: AppTest) -> list[str]:
    """Every column on screen, in the order the tables ask for them."""
    got: list[str] = []
    for d in at.dataframe:
        got.extend(d.proto.column_order)
        if hasattr(d.value, "columns"):
            got.extend(str(c) for c in d.value.columns)
    return got


def interface() -> int:
    """The reading surfaces, driven rather than merely rendered.

    Three things here can break without a page failing: a column set nobody switched to, a depth chart
    nobody drew, and an override knob nobody turned. So the board is switched to the football line, the
    depth chart is read back off the screen and its depth changed, and the knob beside a player's own
    ratings is turned -- with the scenario checked for what it wrote, and the ratings table checked for
    the number the projection then ran on.
    """
    bad = 0

    # the board as a box score
    at = run_page(HOME)
    pick = switch(at, "Columns")
    if pick is None:
        bad += 1
        print("  FAIL  Home: no column-mode switch rendered")
    else:
        at = pick.set_value("Stat line").run()
        order = columns_of(at)
        missing = [c for c in ("passing_yards", "targets", "receiving_yards", "carries",
                              "fantasy_points") if c not in order]
        if at.exception or missing:
            bad += 1
            print(f"  FAIL  Home stat line: missing {missing}"
                  + "".join(f"\n        {e.value}" for e in at.exception))
        else:
            print("  ok    the board switches to a football line, points beside it")

        leaders = [s for s in at.selectbox if s.label == "Stat"]
        if not leaders:
            bad += 1
            print("  FAIL  Home: no stat-leaders picker rendered")
        else:
            at = leaders[0].set_value("receptions").run()
            if at.exception or "receptions" not in columns_of(at):
                bad += 1
                print("  FAIL  Home: stat leaders did not draw receptions"
                      + "".join(f"\n        {e.value}" for e in at.exception))
            else:
                print("  ok    stat leaders ranks the board by one stat at a time")

    # the depth chart: cards on screen, and a slider that changes how many
    at = run_page(TEAM)
    cards = [c.value for c in at.caption if "ranked" in c.value and "slot" in c.value]
    tables = [d for d in at.dataframe
              if hasattr(d.value, "columns") and "pool_share" in d.value.columns]
    # the availability grid is the editable one carrying the attendance record beside the knob; the
    # per-position input grids also have `expected_games` in them, so `record` is what distinguishes it
    grids = [d for d in at.dataframe
             if d.proto.editing_mode and hasattr(d.value, "columns")
             and "expected_games" in d.value.columns and "record" in d.value.columns]
    deep = [s for s in at.slider if s.label.startswith("Players shown")]
    if not cards or not tables or not grids or not deep:
        bad += 1
        print(f"  FAIL  Team depth chart: {len(cards)} cards, {len(tables)} chart tables, "
              f"{len(grids)} availability grids, {len(deep)} depth sliders")
    else:
        after = deep[0].set_value(12).run()
        deeper = [c.value for c in after.caption if "ranked" in c.value and "slot" in c.value]
        if after.exception or len(deeper) <= len(cards):
            bad += 1
            print(f"  FAIL  Team: depth {len(cards)} -> {len(deeper)} cards"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    the depth chart draws {len(cards)} men, {len(deeper)} when opened up, "
                  "with a pool share and a stat line each")

    # a player's own ratings, overridden where they are shown
    at = run_page(PLAYER)
    boxes = [n for n in at.number_input if ":knob:" in (n.key or "") and n.key.endswith(":value")]
    if not boxes:
        bad += 1
        print("  FAIL  Player: no ratings knob rendered")
    else:
        was = float(boxes[0].value)
        at = boxes[0].set_value(round(was * 0.5, 4)).run()
        got = at.session_state[LIVE].items
        if len(got) != 1 or got[0].level != "player" or got[0].mode != "set":
            bad += 1
            print(f"  FAIL  the ratings knob wrote {[o.label for o in got]}")
        else:
            o = got[0]
            if o.field not in overrides.PLAYER_FIELDS or o.base != was:
                bad += 1
                print(f"  FAIL  the knob recorded {o.field} on base {o.base} rather than {was}")
            else:
                print(f"  ok    a rating overridden where it is shown: {o.label}, base {o.base:.4g}")
            # and the ratings table says what the projection then ran on
            rated = [d.value for d in at.dataframe if hasattr(d.value, "columns")
                     and {"used", "applied", "override"} <= set(d.value.columns)]
            if not rated:
                bad += 1
                print("  FAIL  Player: no ratings table with used against applied")
            else:
                said = [str(s) for s in rated[0]["override"] if str(s).strip()]
                if not said:
                    bad += 1
                    print("  FAIL  Player: the override is live but no row admits to it")
                else:
                    print(f"  ok    the ratings table names the override: {said}")
        # `edit_list` scopes its ↺ to whoever drew it, so the key is `<caller>:edits:player:...` in a
        # panel and a bare `edit:...` where a page still calls it directly
        left = [b for b in at.button
                if (b.key or "").startswith("edit:") or ":edits:" in (b.key or "")]
        if not left:
            bad += 1
            print("  FAIL  Player: no reset beside the override in the header list")
        else:
            left[0].click().run()
            if not at.session_state[LIVE].is_baseline:
                bad += 1
                print(f"  FAIL  the header reset left {len(at.session_state[LIVE].items)} edits")
            else:
                print("  ok    the reset beside it drops the override again")
    return bad


# What a ranked list may carry before it stops being readable at a glance. Raised from fourteen when the
# board's default list took the projected football line in front of the scoring: the line is the point of
# the list now, and six stat columns is what it costs. The cap is still here because the failure it guards
# against is real -- the board was once a forty-four-column table nobody scrolled.
LIST_CAP = 16


def panels() -> int:
    """The list-and-panel reading surface, and the ✎ that sits beside a drawn number.

    The redesign's claim is that a projection is read one player at a time: a narrow ranked list, a panel
    of the man beside it, and every estimate drawn as a meter with its two ticks rather than tabulated.
    All three can break without an exception -- a list that quietly grows back to forty-four columns, a
    panel that draws nothing because no row is selected, a ✎ that opens on an empty popover -- so each is
    checked for what it puts on screen, and the ✎ is clicked and followed into the scenario.
    """
    bad = 0

    at = run_page(HOME)
    lists = [d for d in at.dataframe if list(d.proto.selection_mode)]
    if not lists:
        bad += 1
        print("  FAIL  Home: no selectable ranked list")
    else:
        cols = list(lists[0].proto.column_order) or list(lists[0].value.columns)
        if len(cols) > LIST_CAP:
            bad += 1
            print(f"  FAIL  Home: the ranked list carries {len(cols)} columns, cap is {LIST_CAP}")
        else:
            print(f"  ok    the board is a {len(cols)}-column list, selectable a row at a time")

        # and it leads with the football rather than with the scoring, which is the whole point of the
        # default: the points are a conversion of the projection and not the projection
        head = cols[:cols.index("fantasy_points")] if "fantasy_points" in cols else cols
        if "line" not in head:
            bad += 1
            print(f"  FAIL  Home: the list opens on {head}, with no projected line in front of the "
                  "points")
        else:
            print("  ok    the projected line leads the list, the scoring follows it")

        # narrowed to one position it has to become numbers, because that is the moment they compare
        pos = [m for m in at.multiselect if (m.key or "") == "pos"]
        if not pos:
            bad += 1
            print("  FAIL  Home: no position filter to narrow the list with")
        else:
            after = pos[0].set_value(["RB"]).run()
            narrowed = [d for d in after.dataframe if list(d.proto.selection_mode)]
            got = ((list(narrowed[0].proto.column_order) or list(narrowed[0].value.columns))
                   if narrowed else [])
            missing = [c for c in ("carries", "rushing_yards", "rushing_tds", "targets") if c not in got]
            if after.exception or missing or len(got) > LIST_CAP:
                bad += 1
                print(f"  FAIL  Home filtered to the backs: a {len(got)}-column list missing {missing}"
                      + "".join(f"\n        {e.value}" for e in after.exception))
            else:
                print(f"  ok    one position turns the line into {len(got)} sortable columns")

    drawn = screen(at)
    tiles = drawn.count("nf-tile")
    meters = drawn.count("nf-meter")
    if not tiles or not meters:
        bad += 1
        print(f"  FAIL  Home: the panel drew {tiles} tiles and {meters} meters")
    elif "fantasy points" not in drawn or "his record" not in drawn:
        bad += 1
        print("  FAIL  Home: the panel is drawn but names neither the points nor the record")
    else:
        print(f"  ok    a player panel beside it: {tiles} tiles, {meters} meters, his record on each")

    # the ✎ beside a meter, clicked
    at = run_page(PLAYER)
    quick = [b for b in at.button if ":meter:" in (b.key or "")]
    if not quick:
        bad += 1
        print("  FAIL  Player: no ✎ beside the drawn ratings")
    else:
        wanted = float(quick[0].label.rsplit("·", 1)[1].strip())
        after = quick[0].click().run()
        got = after.session_state[LIVE].items
        if after.exception or len(got) != 1:
            bad += 1
            print(f"  FAIL  the ✎ wrote {[o.label for o in got]}"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            o = got[0]
            if o.level != "player" or o.mode != "set" or abs(o.value - wanted) > 5e-4:
                bad += 1
                print(f"  FAIL  the ✎ wrote {o.label}, wanted a player set at {wanted}")
            elif o.base is None:
                bad += 1
                print(f"  FAIL  the ✎ wrote {o.label} with no engine number to compare against")
            else:
                print(f"  ok    a rating changed from the meter beside it: {o.label}, "
                      f"engine had {o.base:.4g}")
            if "✏️" not in screen(after):
                bad += 1
                print("  FAIL  the edit is live but nothing on the page is marked as edited")
            else:
                print("  ok    the page marks the number it is no longer taking from the engine")

    # The complaint the rooms tab exists for: one man adjusted while his whole room is on screen. So the
    # room has to be a list somebody can pick out of, and the pick has to arrive as a panel with a ✎ on
    # it -- a room that renders as a thirty-nine-column sheet and nothing else is the regression.
    at = run_page(TEAM)
    rooms = [d for d in at.dataframe if list(d.proto.selection_mode)]
    picker = [w for w in at.segmented_control if (w.key or "").startswith("room:pos:")]
    drawn = screen(at)
    pens = [b for b in at.button if ":meter:" in (b.key or "")]
    if not (rooms and picker):
        bad += 1
        print(f"  FAIL  Team: {len(rooms)} selectable room list(s), {len(picker)} room picker(s)")
    elif not pens or "nf-meter" not in drawn:
        bad += 1
        print("  FAIL  Team: the room is a list but the man beside it has no drawn ratings to change")
    else:
        cols = list(rooms[0].proto.column_order) or list(rooms[0].value.columns)
        rooms_named = list(picker[0].options)
        print(f"  ok    a {len(cols)}-column room list over {len(rooms_named)} rooms, the man beside "
              f"it drawn as {drawn.count('nf-meter')} meters, each with a ✎ on it")
        if len(cols) > LIST_CAP:
            bad += 1
            print(f"  FAIL  Team: the room list carries {len(cols)} columns, cap is {LIST_CAP}")
    return bad


def evidence() -> int:
    """The evidence beside every override point, and a one-click set that uses it.

    An override made on nothing is the failure this pass is against, so it checks the things that make
    one made on something: the four cards naming the engine's number, the projection's, the player's own
    record and his job's average; the per-season record; the league spread; and a named button that
    writes exactly the value it printed, against the engine's number as its base.
    """
    bad = 0

    at = run_page(PLAYER)
    labels = [m.label for m in at.metric]
    want = ["engine says", "projection is using", "his own record", "his job is worth"]
    missing = [w for w in want if w not in labels]
    if missing:
        bad += 1
        print(f"  FAIL  Player: the override panel is missing {missing}")
    else:
        print("  ok    the override panel names the engine, the projection, the man and the job")

    quick = [b for b in at.button if ":quick:" in (b.key or "")]
    if not quick:
        bad += 1
        print("  FAIL  Player: no one-click set beside the knob")
    else:
        # the label is the value rounded for reading; the override carries the number itself, so the
        # tolerance here is the printing rather than a licence for the button to write something else
        wanted = float(quick[0].label.rsplit("·", 1)[1].strip())
        after = quick[0].click().run()
        got = after.session_state[LIVE].items
        if after.exception or len(got) != 1 or abs(got[0].value - wanted) > 5e-4:
            bad += 1
            print(f"  FAIL  a quick set wrote {[o.label for o in got]}, wanted {wanted}"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            o = got[0]
            print(f"  ok    a named value set in one click: {o.label}, "
                  f"base {'—' if o.base is None else f'{o.base:.4g}'}")

    # the record itself, and the league it is read against
    frames = [d.value for d in at.dataframe if hasattr(d.value, "columns")]
    seasons = [f for f in frames if "season" in f.columns and "value" in f.columns]
    spread = [f for f in frames if {"median", "p90", "best"} <= set(f.columns)]
    if not seasons or not spread:
        bad += 1
        print(f"  FAIL  Player: {len(seasons)} per-season tables, {len(spread)} league spreads")
    else:
        print(f"  ok    the record is on screen season by season, against {len(spread)} league spread(s)")

    # and the same evidence on the team page, where a whole room is edited at once
    at = run_page(TEAM)
    frames = [d.value for d in at.dataframe if hasattr(d.value, "columns")]
    with_engine = [f for f in frames if {"engine", "obs", "own_weight"} <= set(f.columns)]
    rooms = [f for f in frames if {"estimate", "applied", "own_weight"} <= set(f.columns)]
    if not with_engine or not rooms:
        bad += 1
        print(f"  FAIL  Team: {len(with_engine)} grids with the record, {len(rooms)} room tables")
    else:
        print(f"  ok    the team grids carry the record ({len(with_engine)}) and the room "
              f"comparison ({len(rooms)})")
    return bad


def depth() -> int:
    """A man moved on a chart, which is the one edit that re-prices rather than re-labels.

    Every other override sets a number and the number is what changes. A slot sets a *rank*, so a broken
    version of it is not an exception -- it is a chart that still reads 1, 2, 3 while the projection
    underneath is unchanged. So this drives the move from the page and then insists on all three: the
    edit in the scenario, a `depth` row in the provenance carrying two slots, and the page reading him
    back where he was sent.
    """
    bad = 0

    at = run_page(TEAM)
    charts = [d for d in at.dataframe
              if d.proto.editing_mode and hasattr(d.value, "columns")
              and "depth_slot" in d.value.columns]
    resets = [b for b in at.button if "resetchart" in (b.key or "")]
    editable = [d for d in charts if "depth slot" in d.proto.columns]   # the label the editor gives it
    if not charts or not resets or not editable:
        bad += 1
        print(f"  FAIL  Team: {len(charts)} charts with a slot column, {len(editable)} of them "
              f"editable, {len(resets)} chart resets")
    else:
        print(f"  ok    the chart is editable in place ({len(editable[0].value)} men) with a reset")

    at = run_page(PLAYER)
    boxes = [n for n in at.number_input if (n.key or "").startswith("player:slot:")]
    if not boxes:
        bad += 1
        print("  FAIL  Player: nowhere to move a man on the chart")
        return bad

    was = int(boxes[0].value)
    want = 2 if was == 1 else 1
    after = boxes[0].set_value(want).run()
    got = [o for o in after.session_state[LIVE].items if o.field == overrides.DEPTH_FIELD]
    if after.exception or len(got) != 1 or int(got[0].value) != want:
        bad += 1
        print(f"  FAIL  moving a man wrote {[o.label for o in got]}, wanted slot {want}"
              + "".join(f"\n        {e.value}" for e in after.exception))
        return bad
    print(f"  ok    slot {was} -> {want} is an edit: {got[0].label}, base {got[0].base}")

    # the provenance is the ledger's, on the page that keeps it: the move has to arrive there as a
    # `depth` stage carrying two slots rather than as a value nobody can read back
    led = run_page(EDITS, after.session_state[LIVE])
    frames = [d.value for d in led.dataframe if hasattr(d.value, "columns")]
    prov = [f for f in frames if "stage" in f.columns and "depth" in set(f["stage"])]
    if not prov:
        bad += 1
        print("  FAIL  the move is in the scenario but the ledger does not report a depth stage")
    else:
        table = prov[0]
        row = table[table["stage"] == "depth"].iloc[0]           # a pandas frame off the screen
        if not bool(row["applied"]) or int(row["used_now"]) != want:
            bad += 1
            print(f"  FAIL  the depth stage reports base {row['base_now']} -> used {row['used_now']}, "
                  f"applied {row['applied']}")
        else:
            print(f"  ok    reported in slots: {row['base_now']:.0f} -> {row['used_now']:.0f}, applied")

    # his slot is a drawn tile now: the value is the slot, and the line under it says how big the room is
    slot_tiles = [t for t in drawn_tiles(after) if "in the room" in t["sub"]]
    if not any(t["value"] == str(want) for t in slot_tiles):
        bad += 1
        print(f"  FAIL  the page does not read him back at slot {want}: "
              f"{[(t['name'], t['value']) for t in slot_tiles][:6]}")
    else:
        print(f"  ok    the page reads him back at slot {want}, priced off the slot he moved to")
    return bad


def off_roster() -> int:
    """A man taken off a roster: the one edit whose subject is missing from the run it produced.

    It fails quietly in both directions. Broken one way the release does nothing and the chart still
    reads a man the team has cut; broken the other the row is gone and nothing on screen says so -- a
    scenario with a hole in it, a ledger row nobody can read a name off, and no way to put him back. So
    the pass drives it from the depth chart and insists on all four: the edit, the room closing up behind
    him, the ledger naming him and reporting a `roster` stage, and the reset putting him back.
    """
    bad = 0
    b = overrides.run(Scenario())
    team = sorted(b.board["team"].unique().to_list())[0]        # the team the page opens on
    rooms = (b.part.filter(pl.col("team") == team).group_by("position").len()
             .filter(pl.col("len") >= 4).sort("len", descending=True))
    if rooms.is_empty():
        print(f"  ok    no room on {team} is deep enough to cut from — nothing to drive")
        return bad
    pos = rooms["position"][0]
    room = b.part.filter((pl.col("team") == team) & (pl.col("position") == pos)).sort("depth_slot")
    pid = room["player_id"][1]                                  # the second man: men on both sides of him
    name = b.board.filter(pl.col("player_id") == pid)["player"][0]

    at = run_page(TEAM)
    boxes = [m for m in at.multiselect if (m.key or "").startswith(f"release:{team}")]
    if not boxes:
        bad += 1
        print(f"  FAIL  Team: no way to take a man off {team}'s roster")
        return bad

    after = boxes[0].set_value([pid]).run()
    got = [o for o in after.session_state[LIVE].items if o.field == overrides.ROSTER_FIELD]
    if after.exception or len(got) != 1 or got[0].key != pid or float(got[0].value) != 0.0:
        bad += 1
        print(f"  FAIL  releasing {name} wrote {[o.label for o in got]}"
              + "".join(f"\n        {e.value}" for e in after.exception))
        return bad
    print(f"  ok    {name} ({pos}{int(room['depth_slot'][1])} of {team}) is off the roster: "
          f"{got[0].label}")

    charts = [d.value for d in after.dataframe
              if d.proto.editing_mode and hasattr(d.value, "columns")
              and "depth_slot" in d.value.columns and "depth slot" in d.proto.columns]
    if not charts:
        bad += 1
        print("  FAIL  the chart did not redraw after the release")
    else:
        mine = charts[0][charts[0]["position"] == pos]
        slots = sorted(int(s) for s in mine["depth_slot"])
        if name in list(charts[0]["player"]) or slots != list(range(1, len(slots) + 1)):
            bad += 1
            print(f"  FAIL  the chart still reads {name}" if name in list(charts[0]["player"])
                  else f"  FAIL  the room did not close up: slots {slots}")
        else:
            print(f"  ok    the {pos} room closed up behind him — {len(slots)} men, slots 1 to "
                  f"{len(slots)}")

    # the ledger is where a release has to survive being invisible: he has no row in the run, so the
    # page has to reach for the baseline to name him rather than printing an id
    led = run_page(EDITS, after.session_state[LIVE])
    frames = [d.value for d in led.dataframe if hasattr(d.value, "columns")]
    prov = [f for f in frames if "stage" in f.columns and "roster" in set(f["stage"])]
    named = [f for f in frames if "who" in f.columns and any(name in str(w) for w in f["who"])]
    if not prov:
        bad += 1
        print("  FAIL  the release is in the scenario but the ledger reports no roster stage")
    else:
        row = prov[0][prov[0]["stage"] == "roster"].iloc[0]
        if not bool(row["applied"]) or float(row["used_now"]) != 0.0:
            bad += 1
            print(f"  FAIL  the roster stage reports used {row['used_now']}, "
                  f"applied {row['applied']}")
        elif not named:
            bad += 1
            print(f"  FAIL  the ledger has the release but does not name {name}")
        else:
            print(f"  ok    the ledger names him and reports it: on_roster "
                  f"{row['base_now']:.0f} -> {row['used_now']:.0f}, applied")

    resets = [x for x in after.button if "resetroster" in (x.key or "")]
    if not resets:
        bad += 1
        print("  FAIL  nothing to put him back with")
        return bad
    back = resets[0].click().run()
    left = [o for o in back.session_state[LIVE].items if o.field == overrides.ROSTER_FIELD]
    again = [d.value for d in back.dataframe
             if d.proto.editing_mode and hasattr(d.value, "columns")
             and "depth_slot" in d.value.columns and "depth slot" in d.proto.columns]
    if back.exception or left or not again or name not in list(again[0]["player"]):
        bad += 1
        print(f"  FAIL  putting him back left {len(left)} releases and "
              f"{'no chart' if not again else ('him off it' if not again or name not in list(again[0]['player']) else 'him on it')}"
              + "".join(f"\n        {e.value}" for e in back.exception))
    else:
        print(f"  ok    ↺ roster puts {name} back on the chart and drops the edit")
    return bad


def population() -> int:
    """The two edits that are not one number at one point.

    A bulk edit writes an override per player, so the failure mode is not an exception: it is a filter
    that catches nobody, a preview that renders and an Apply button that writes one row or none. And a
    return date is a *unit conversion* -- week 8 becomes "ten games left" off that team's own schedule --
    so it can print a sensible number and record something else entirely.
    """
    bad = 0

    # a whole position at once
    at = run_page(EDITS)
    fields = [s for s in at.selectbox if (s.key or "") == "bulk:field"]
    if not fields:
        bad += 1
        print("  FAIL  Edits: no bulk-edit field picker rendered")
    else:
        at = fields[0].set_value("expected_games").run()
        boxes = [n for n in at.number_input if (n.key or "").startswith("bulk:value:")]
        if not boxes:
            bad += 1
            print("  FAIL  Edits: no bulk value box")
        else:
            at = boxes[0].set_value(16.5).run()
            applies = [b for b in at.button if b.label.startswith("Apply to ")]
            if not applies:
                bad += 1
                print(f"  FAIL  Edits: no Apply button; buttons were "
                      f"{[b.label for b in at.button][:8]}")
            else:
                wanted = int(applies[0].label.split()[2])
                after = applies[0].click().run()
                got = [o for o in after.session_state[LIVE].items if o.field == "expected_games"]
                if after.exception or len(got) != wanted or wanted < 10:
                    bad += 1
                    print(f"  FAIL  a bulk edit on {wanted} players wrote {len(got)} overrides"
                          + "".join(f"\n        {e.value}" for e in after.exception))
                elif any(o.mode != "set" or o.value != 16.5 for o in got):
                    bad += 1
                    print(f"  FAIL  the bulk edit wrote {got[0].mode} {got[0].value}")
                else:
                    based = sum(1 for o in got if o.base is not None)
                    print(f"  ok    one filter, {len(got)} overrides at 16.5 games, {based} of them "
                          "against the engine's own number")

    # a return date, as the games it is worth
    at = run_page(AVAILABILITY)
    backs = [s for s in at.select_slider if "avail:back:" in (s.key or "")]
    sets = [b for b in at.button if b.label.startswith("Set to ")]
    if not backs or not sets:
        bad += 1
        print(f"  FAIL  Availability: {len(backs)} return sliders, {len(sets)} set buttons")
    else:
        weeks = list(backs[0].options)
        at = backs[0].set_value(weeks[min(7, len(weeks) - 1)]).run()
        sets = [b for b in at.button if b.label.startswith("Set to ")]
        wanted = float(sets[0].label.split()[2])
        after = sets[0].click().run()
        got = [o for o in after.session_state[LIVE].items if o.field == "expected_games"]
        if after.exception or len(got) != 1 or got[0].value != wanted:
            bad += 1
            print(f"  FAIL  'back in week' wrote {[o.label for o in got]}, wanted {wanted} games"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    a return date becomes a games count: {got[0].label}"
                  + (f" · {got[0].note}" if got[0].note else ""))

    # the comparison, which is a table of estimates rather than of totals. It is a tab of the Player page
    # now, and it opens on the man nearest him at his own position -- so a page that ran is not enough:
    # the two-column `n` is the whole point of the layer and it is what is checked.
    at = run_page(PLAYER)
    frames = [d.value for d in at.dataframe if hasattr(d.value, "columns")]
    rated = [f for f in frames if {"kind", "metric", "units"} <= set(f.columns)
             and any(str(c).startswith("n · ") for c in f.columns)]
    if at.exception or not rated:
        bad += 1
        print("  FAIL  Player: no side-by-side ratings table"
              + "".join(f"\n        {e.value}" for e in at.exception))
    else:
        people = [c for c in rated[0].columns if str(c).startswith("n · ")]
        if len(people) < 2:
            bad += 1
            print(f"  FAIL  Player: {len(people)} players' sample sizes side by side, wanted 2")
        else:
            print(f"  ok    two men compared on {len(rated[0])} shared estimates, sample size each")

    # one week of a season that is computed per week
    at = run_page(WEEKLY)
    picks = [s for s in at.select_slider if (s.key or "") == "weekly:week"]
    if not picks:
        bad += 1
        print("  FAIL  Weekly: no week picker")
    else:
        weeks = list(picks[0].options)
        after = picks[0].set_value(weeks[min(5, len(weeks) - 1)]).run()
        order = columns_of(after)
        if after.exception or "week_rank" not in order:
            bad += 1
            print("  FAIL  Weekly: a week picked but no per-week rank on screen"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    week {weeks[min(5, len(weeks) - 1)]} ranked at position, "
                  f"{len(after.dataframe)} tables")
    return bad


# The one number each room is really argued about, which is the column its sheet has to carry.
ROOM_HEADLINE = {"QB": "dropback_share", "RB": "carry_share", "WR": "target_share",
                 "TE": "target_share"}


def typeable(at: AppTest) -> list:
    """Every table on screen somebody can type into."""
    return [d.value for d in at.dataframe
            if d.proto.editing_mode and hasattr(d.value, "columns")]


def sheet() -> int:
    """Each of a team's four rooms as a sheet, every rating in it editable.

    `AppTest` cannot type into a `st.data_editor`, so the translation from a typed cell to an override is
    pinned in `tests/test_app_ui.py` instead. What can only be caught here is the rendering: each room is
    a different frame built from a different list of fields with a different `column_config`, and a list
    that names a column the frame does not carry -- or hands an integer slot a percent format -- is an
    exception in a room nobody opened during the baseline pass, which only visits the default one. The
    evidence join is driven for both of its branches for the same reason: *nothing* is the setting no
    other check ever sees.
    """
    bad = 0
    at = run_page(TEAM)
    team = at.session_state[ui.FOCUS_TEAM]
    for pos, metric in ROOM_HEADLINE.items():
        # a segmented control cannot be clicked from a test, so the room is chosen by writing its key
        at.session_state[f"room:pos:{team}"] = pos
        after = at.run()
        sheets = [f for f in typeable(after) if metric in f.columns]
        if after.exception or not sheets:
            bad += 1
            print(f"  FAIL  the {team} {pos} room does not render as a sheet carrying `{metric}`"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    {pos}: {len(sheets[0])} men, {len(sheets[0].columns)} columns")
        at = after

    last, metric = "TE", ROOM_HEADLINE["TE"]
    for choice in ("nothing", metric):
        picks = [s for s in at.selectbox if (s.key or "") == f"inputs:evidence:{team}:{last}"]
        if not picks:
            bad += 1
            print(f"  FAIL  Team: the {last} sheet has no evidence picker")
            break
        after = picks[0].set_value(choice).run()
        joined = [f for f in typeable(after) if metric in f.columns and "engine" in f.columns]
        if after.exception or bool(joined) != (choice != "nothing"):
            bad += 1
            print(f"  FAIL  evidence set to '{choice}' produced {len(joined)} sheets with the record "
                  "joined on" + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    evidence '{choice}': "
                  + (f"{len(joined[0].columns)} columns with the record beside the knob"
                     if joined else "the knobs on their own"))
        at = after
    return bad


BENCH_IDLE = "Type a number into"


def bench_at(at: AppTest) -> str:
    """The batched editor's own idle caption, or empty if it never drew."""
    return next((str(c.value) for c in at.caption if BENCH_IDLE in str(c.value)), "")


def bench() -> int:
    """One player's whole rating card, typed over in a batch.

    `AppTest` cannot type into a `st.data_editor`, so what a typed cell becomes is pinned in
    `tests/test_app_ui.py`; what can only be caught here is the shape. Each `Show` set is a different
    list of knobs against a different frame, the week scope swaps the whole field list for the ones a week
    can carry, and the editor is reached from a session flag rather than a page -- three things that can
    each be an exception on a surface the baseline pass never opened. The scope controls are
    `segmented_control`, which `AppTest` cannot click, so they are driven by seeding their widget state,
    which is what Streamlit reads as the default.
    """
    bad = 0
    for column_set in ui.BENCH_SETS:
        at = AppTest.from_file(str(ROOT / PLAYER), default_timeout=300)
        at.session_state[LIVE] = Scenario()
        at.session_state["player:bench:set"] = column_set
        at.run()
        if at.exception or not bench_at(at):
            bad += 1
            print(f"  FAIL  the editor does not render for '{column_set}'"
                  + "".join(f"\n        {e.value}" for e in at.exception))
        else:
            print(f"  ok    {column_set}: his card drawn, nothing written until it is applied")

    # the same rows scoped to two weeks: a different field list, and a base per week rather than one
    at = AppTest.from_file(str(ROOT / PLAYER), default_timeout=300)
    at.session_state[LIVE] = Scenario()
    at.session_state["player:bench:span"] = "some weeks"
    at.session_state["player:bench:weeks"] = [3, 4]
    at.run()
    said = next((str(c.value) for c in at.caption if "own numbers" in str(c.value)), "")
    if at.exception or not bench_at(at) or "week 3" not in said:
        bad += 1
        print("  FAIL  the editor does not scope to chosen weeks"
              + (f"\n        caption: {said!r}" if not at.exception else "")
              + "".join(f"\n        {e.value}" for e in at.exception))
    else:
        print(f"  ok    scoped to weeks 3 and 4: {said.split('.')[0]}")

    # and the way every other page reaches it: a flag, raised as a dialog by `ui.controls`
    board = overrides.run(Scenario()).board
    who = board.row(0, named=True)
    at = AppTest.from_file(str(ROOT / TEAM), default_timeout=300)
    at.session_state[LIVE] = Scenario()
    at.session_state[ui.BENCH_OPEN] = who["player_id"]
    at.run()
    on_screen = screen(at)
    if at.exception or not bench_at(at) or who["player"] not in on_screen:
        bad += 1
        print(f"  FAIL  the editor does not open on {who['player']} from another page"
              + "".join(f"\n        {e.value}" for e in at.exception))
    else:
        print(f"  ok    opened over the team page on {who['player']}, "
              f"{len([b for b in at.button if 'bench' in (b.key or '')])} ways in on screen")
    return bad


def points(value: str) -> float:
    """`'-25.0 pts'`, `'+3.4 points'` and `'234.6'` back into numbers, so a sign can be asserted."""
    text = str(value).replace(",", "").replace("+", "")
    for word in ("points", "pts", "pt"):
        text = text.replace(word, "")
    return float(text.strip())


def whatif() -> int:
    """One edit previewed before it is written, and the property that makes the preview worth reading.

    The tab answers *what does this cost his teammates*, and the answer is only trustworthy if it comes
    from a second engine run rather than from arithmetic on the page. So this pass drives the control and
    then insists on the sign structure a pool edit must have: he gains, the room loses, and the offence
    barely moves -- a preview that showed all three going the same way would be a page adding numbers up
    instead of a projection being recomputed. Then it applies the candidate and checks exactly one
    override was written, against the engine's own number.
    """
    bad = 0

    at = run_page(TEAM)
    pools = [s for s in at.selectbox if (s.key or "").startswith("whatif:pool:")]
    who = [s for s in at.selectbox if (s.key or "").startswith("whatif:who:")]
    knobs = [s for s in at.selectbox if (s.key or "").startswith("whatif:field:")]
    dials = [s for s in at.slider if "whatif:pct:" in (s.key or "")]
    if not (pools and who and knobs and dials):
        bad += 1
        print(f"  FAIL  Team/what if: {len(pools)} pool pickers, {len(who)} player pickers, "
              f"{len(knobs)} number pickers, {len(dials)} controls")
        return bad
    field = knobs[0].value
    print(f"  ok    who produces the offence, over {len(pools[0].options)} pools; "
          f"{pools[0].value} opens on {field}")

    before_charts = len(at.get("vega_lite_chart"))
    after = dials[0].set_value(120).run()
    if after.exception:
        bad += 1
        print("  FAIL  the ripple raised"
              + "".join(f"\n        {e.value}" for e in after.exception))
        return bad

    cards = drawn_tiles(after)
    labels = [t["name"] for t in cards]
    room = tile_named(cards, "his teammates (")
    whole = tile_named(cards, "the whole offence")
    him = next((t for t in cards if t["sub"].endswith("points")), None)
    said = [i.value for i in after.info if "points" in str(i.value)]
    if room is None or whole is None or him is None or not said:
        bad += 1
        print(f"  FAIL  no ripple panels: tiles were {labels[:8]}")
        return bad

    his = points(him["sub"])                     # his own tile carries the change under the total
    theirs, net = points(room["value"]), points(whole["value"])
    movers = int(room["name"].split("(")[1].rstrip(")"))
    if not (his > 0 > theirs and movers > 0 and abs(net) < abs(his)):
        bad += 1
        print(f"  FAIL  {field} +20% reads him {his:+.1f}, {movers} teammates {theirs:+.1f}, "
              f"offence {net:+.1f} — a share is zero-sum and this is not")
    else:
        print(f"  ok    +20% of {field}: him {his:+.1f}, {movers} teammates {theirs:+.1f}, "
              f"the offence {net:+.1f} — a transfer, not new production")

    charts = len(after.get("vega_lite_chart"))
    if charts <= before_charts:
        bad += 1
        print(f"  FAIL  the pool is not drawn before and after: {charts} charts, was {before_charts}")
    else:
        print(f"  ok    the same pool divided two ways is drawn ({charts} charts, was {before_charts})")

    applies = [b for b in after.button if b.label.startswith("✔ Apply this to ")]
    if not applies:
        bad += 1
        print(f"  FAIL  nothing to apply the candidate with: {[b.label for b in after.button][:6]}")
        return bad
    done = applies[0].click().run()
    got = [o for o in done.session_state[LIVE].items if o.field == field]
    if done.exception or len(got) != 1 or got[0].mode != "multiply":
        bad += 1
        print(f"  FAIL  applying the candidate wrote {[o.label for o in got]}"
              + "".join(f"\n        {e.value}" for e in done.exception))
    elif got[0].base is None:
        bad += 1
        print(f"  FAIL  {got[0].label} was written with no base, so no diff and no reset to the engine")
    else:
        print(f"  ok    the candidate becomes one override: {got[0].label}, base {got[0].base:.4f}")
    return bad


def one_game_edit(week: int = 5) -> tuple[Scenario, dict]:
    """A scenario that rules the busiest man in one game out of that game alone.

    The edit the game tabs exist for, and the one the season has to follow on its own: nothing about
    his season is touched, only the availability in one fixture, so his total falls by exactly the game
    and his teammates rise by what he was taking.
    """
    wk = overrides.run(Scenario()).weekly
    row = (wk.filter(pl.col("week") == week).sort("fantasy_points", descending=True)
           .row(0, named=True))
    sc = Scenario().set(Override("player", row["player_id"], "p_play", "set", 0.0, week=week))
    return sc, row


def games() -> int:
    """One game picked through to a fixture, both box scores read back, and the editors rendered.

    Two things here break without a page failing. The reconciliation is one: an edit lands before the
    pool is divided, so a team's counted work must still add up to the offence it was divided out of
    whatever has been typed -- and a version that applied the edit after the division would render
    perfectly with the targets no longer summing to the targets. The other is the grain: a sheet of one
    week that quietly wrote season-wide overrides would look identical on screen, so the season effect
    is driven from a per-game edit and the man's own week is checked for the zero.
    """
    bad = 0

    at = run_page(WEEKLY)
    picks = [s for s in at.select_slider if (s.key or "") == "weekly:week"]
    fixtures = [s for s in at.selectbox if (s.key or "") == "games:game"]
    if not picks or not fixtures:
        print(f"  FAIL  One game: {len(picks)} week pickers, {len(fixtures)} fixture pickers")
        return bad + 1

    weeks = list(picks[0].options)
    at = picks[0].set_value(weeks[min(4, len(weeks) - 1)]).run()
    fixtures = [s for s in at.selectbox if (s.key or "") == "games:game"]
    slate = list(fixtures[0].options)
    at = fixtures[0].set_value(slate[-1]).run()
    if at.exception:
        bad += 1
        print("  FAIL  One game: picking a fixture raised"
              + "".join(f"\n        {e.value}" for e in at.exception))
        return bad
    print(f"  ok    week {weeks[min(4, len(weeks) - 1)]}: {len(slate)} games, {slate[-1]} opened")

    frames = [d.value for d in at.dataframe if hasattr(d.value, "columns")]
    box = [f for f in frames if {"team", "carries", "targets", "fantasy_points"} <= set(f.columns)]
    rec = [f for f in frames
           if {"stat", "from_the_players", "from_the_offence", "gap"} <= set(f.columns)]
    if not box or not rec:
        bad += 1
        print(f"  FAIL  One game: {len(box)} box scores, {len(rec)} reconciliations on screen")
    else:
        pooled = rec[0][rec[0]["from_a_pool"]]
        worst = float(pooled["gap"].abs().max()) if len(pooled) else 0.0
        if worst > 0.05:
            bad += 1
            print(f"  FAIL  a counted pool is off by {worst:.3f} in one game — the players do not add "
                  "up to the offence they were divided out of")
        else:
            print(f"  ok    both box scores drawn, {len(pooled)} counted pools reconcile "
                  f"(worst gap {worst:.3f})")

    editors = [d for d in at.dataframe if d.proto.editing_mode and hasattr(d.value, "columns")]
    team_grid = [d for d in editors if {"team", "week"} <= set(d.value.columns)
                 and "dropbacks" in d.value.columns]
    player_grid = [d for d in editors if "p_play" in d.value.columns]
    if not team_grid or not player_grid:
        bad += 1
        print(f"  FAIL  One game: {len(team_grid)} team-volume grids, {len(player_grid)} player grids")
    else:
        print(f"  ok    both editors rendered: {len(team_grid[0].value)} offences, "
              f"{len(player_grid[0].value)} men in this game")

    sets = [s for s in at.selectbox if (s.key or "") == "games:sheet:cols"]
    if not sets:
        bad += 1
        print("  FAIL  One game: the player sheet has no column-set picker")
    else:
        for column_set in list(sets[0].options):
            after = [s for s in at.selectbox
                     if (s.key or "") == "games:sheet:cols"][0].set_value(column_set).run()
            grids = [d.value for d in after.dataframe
                     if d.proto.editing_mode and hasattr(d.value, "columns")
                     and "p_play" in d.value.columns]
            if after.exception or not grids:
                bad += 1
                print(f"  FAIL  the game sheet does not render for '{column_set}'"
                      + "".join(f"\n        {e.value}" for e in after.exception))
                continue
            offered = [c for c in grids[0].columns if c in overrides.GAME_ALL_FIELDS]
            season_only = [c for c in grids[0].columns
                           if c in overrides.PLAYER_FIELDS and c not in overrides.GAME_ALL_FIELDS]
            if not offered or season_only:
                bad += 1
                print(f"  FAIL  '{column_set}': {len(offered)} per-game fields offered, and "
                      f"{season_only} which a single week cannot carry")
            else:
                print(f"  ok    {column_set}: {len(offered)} per-game fields, no season-only column")
            at = after

    # a per-game edit, and the season following it
    sc, row = one_game_edit()
    at = run_page(WEEKLY, sc)
    picks = [s for s in at.select_slider if (s.key or "") == "weekly:week"]
    at = picks[0].set_value(int(row["week"])).run()
    fixtures = [s for s in at.selectbox if (s.key or "") == "games:game"]
    his = [g for g in fixtures[0].options if row["team"] in g]
    if not his:
        bad += 1
        print(f"  FAIL  One game: {row['team']} is not in any week {row['week']} fixture")
        return bad
    at = fixtures[0].set_value(his[0]).run()
    # a man ruled out projects nothing, and the box score hides nothing men by default -- so the
    # everybody switch is what puts the zero he is now on screen to be read
    show_all = [t for t in at.toggle if (t.key or "") == "games:box:all"]
    if show_all:
        at = show_all[0].set_value(True).run()
    if at.exception:
        bad += 1
        print("  FAIL  One game: the edited game raised"
              + "".join(f"\n        {e.value}" for e in at.exception))
        return bad

    frames = [d.value for d in at.dataframe if hasattr(d.value, "columns")]
    moved = [f for f in frames if "d_fantasy_points" in f.columns and "player" in f.columns]
    # the box score rather than the editing sheet: only the box carries the yardage columns
    lines = [f for f in frames
             if {"player", "p_play", "receiving_yards", "rushing_yards"} <= set(f.columns)]
    if not moved or not lines:
        bad += 1
        print(f"  FAIL  One game: {len(moved)} season-effect tables, {len(lines)} player box scores")
        return bad

    him = moved[0][moved[0]["player"] == row["player"]]
    if not len(him) or float(him["d_fantasy_points"].iloc[0]) >= 0.0:
        bad += 1
        print(f"  FAIL  {row['player']} out of one game did not lower his season: "
              f"{list(moved[0]['player'])[:5]}")
    else:
        others = moved[0][moved[0]["player"] != row["player"]]
        rose = int((others["d_fantasy_points"] > 0).sum())
        print(f"  ok    out of week {row['week']} only: his season "
              f"{float(him['d_fantasy_points'].iloc[0]):+.1f} pts, {rose} teammates up — "
              f"{len(moved[0])} men moved in all")
    # he is on one of the two sides, so both are looked in rather than assuming which
    hits = [f[f["player"] == row["player"]] for f in lines]
    hits = [h for h in hits if len(h)]
    if not hits:
        bad += 1
        print(f"  FAIL  {row['player']} is not in either box score for his own game")
    elif float(hits[0]["p_play"].iloc[0]) != 0.0:
        bad += 1
        print(f"  FAIL  the box score still has him at {float(hits[0]['p_play'].iloc[0]):.2f} to play")
    else:
        print("  ok    the box score for that game has him out, and only that game")
    return bad


def frames_with(at: AppTest, columns: set[str], height: int | None = None) -> list:
    """Every table on screen carrying all of these columns, optionally of a given height."""
    out = []
    for d in at.dataframe:
        got = d.value
        if not hasattr(got, "columns") or not columns <= set(got.columns):
            continue
        if height is None or len(got) == height:
            out.append(got)
    return out


def league() -> int:
    """The projected records, and the arithmetic on screen that makes them readable.

    A record is the one number on the board a football person checks against instinct before checking
    anything else, and the invariant it lives on is that somebody wins every game: the thirty-two
    expected-win totals have to add to exactly half the games played. That cannot be asserted in the
    model alone, because the page is free to show a filtered or double-counted frame and would look
    perfectly plausible doing it -- BUF at 10.5 reads the same whether the league adds to 272 or 300.

    The coverage table is checked in the same pass and for the same reason: it is the surface that says
    which of thirty-two sittings are done, so a version that called every team reviewed because one was
    edited would be worse than not having it.
    """
    bad = 0

    at = run_page(LEAGUE)
    if at.exception:
        print("  FAIL  League: the page raised"
              + "".join(f"\n        {e.value}" for e in at.exception))
        return 1

    table = frames_with(at, {"team", "expected_wins"}, height=32)
    if not table:
        bad += 1
        print("  FAIL  League: no thirty-two-row table of expected wins on screen")
    else:
        total = float(table[0]["expected_wins"].sum())
        if abs(total - 272.0) > 0.5:
            bad += 1
            print(f"  FAIL  League: the league projects {total:.1f} wins out of 272 games")
        else:
            top = table[0].sort_values("expected_wins", ascending=False).iloc[0]
            print(f"  ok    32 records summing to {total:.1f} — best {top['team']} "
                  f"{float(top['expected_wins']):.2f}")

    tiles = drawn_tiles(at)
    for name in ("best record", "worst record", "win model"):
        if tile_named(tiles, name) is None:
            bad += 1
            print(f"  FAIL  League: no '{name}' tile on screen")
    scored = tile_named(tiles, "win model")
    if scored is not None:
        print(f"  ok    the fit is stated on the page: {scored['value']} {scored['sub']}")

    # and the coverage table, under a scenario that touches exactly one team
    board = overrides.run(Scenario()).board
    team = sorted(board["team"].unique().to_list())[0]
    man = (board.filter(pl.col("team") == team).sort("fantasy_points", descending=True)
           .row(0, named=True))
    sc = Scenario().set(Override("player", man["player_id"], "target_share", "multiply", 1.2),
                        Override("team", team, "plays", "multiply", 1.05))
    at = run_page(LEAGUE, sc)
    if at.exception:
        bad += 1
        print("  FAIL  League: the edited league page raised"
              + "".join(f"\n        {e.value}" for e in at.exception))
        return bad

    cover = frames_with(at, {"team", "reviewed", "edits"}, height=32)
    if not cover:
        bad += 1
        print("  FAIL  League: no coverage table on screen")
    else:
        done = cover[0][cover[0]["reviewed"]]
        if list(done["team"]) != [team] or int(done["edits"].iloc[0]) != 2:
            bad += 1
            print(f"  FAIL  League: {list(done['team'])} read as reviewed after editing {team} alone")
        else:
            print(f"  ok    {team} reads as reviewed on 2 edits, the other 31 do not")

    # An edit has to reach the record, or the page is a second model rather than a reading of this one.
    # It is `implied_points` that is edited and not the play count on purpose: a record is derived from
    # the scoring level alone, and the volume edit above deliberately does *not* move it -- more plays is
    # not more points in this chain, which is exactly the kind of thing the page has to be honest about.
    scored = Scenario().set(*[Override("team", team, "implied_points", "multiply", 1.15, week=w)
                              for w in range(1, 19)])
    bare = frames_with(run_page(LEAGUE), {"team", "expected_wins"}, height=32)
    now = frames_with(run_page(LEAGUE, scored), {"team", "expected_wins"}, height=32)
    if not bare or not now:
        bad += 1
        print("  FAIL  League: no standings to compare with and without the edit")
        return bad
    was = float(bare[0].set_index("team").loc[team, "expected_wins"])
    after = float(now[0].set_index("team").loc[team, "expected_wins"])
    if after - was < 0.25:
        bad += 1
        print(f"  FAIL  League: {team} projects {after:.3f} wins on 15% more points, was {was:.3f}")
    else:
        print(f"  ok    {team} moves {after - was:+.3f} wins on 15% more points")
    total = float(now[0]["expected_wins"].sum())
    if abs(total - 272.0) > 0.5:
        bad += 1
        print(f"  FAIL  League: the edited league projects {total:.1f} wins out of 272 games")
    else:
        print(f"  ok    and the league still projects {total:.1f} — somebody else paid for it")
    return bad


def room() -> int:
    """The room sheet: the record beside each knob, what the room adds up to, and the way to the next team.

    All three can be wrong while the page renders perfectly. The pool sums are arithmetic about the
    engine's own arithmetic -- summed over the wrong player set, or left unweighted where the engine
    weights by availability -- and a wrong answer reads as a plausible number rather than as a failure, so
    `as run` is tied to the pool audit the same page draws. The navigation writes the Team page's own
    selectbox key from a callback, which either moves the dropdown or silently does nothing, and *next
    untouched* has to agree with the league page about which teams are done.
    """
    bad = 0
    at = run_page(TEAM)
    opened = at.session_state[ui.FOCUS_TEAM]

    wanted = {"target_share", "engine", "obs", "n", "own_weight", "prior"}
    sheets = [d.value for d in at.dataframe
              if d.proto.editing_mode and hasattr(d.value, "columns")
              and wanted <= set(d.value.columns)]
    if not sheets:
        bad += 1
        print("  FAIL  Team: no editable room sheet carries the record beside the knob")
    else:
        print(f"  ok    the room sheet is editable with the evidence joined on: {len(sheets[0])} men, "
              f"{len(sheets[0].columns)} columns")

    sums = frames_with(at, {"pool", "shares", "as run", "there to divide"})
    if not sums:
        bad += 1
        print("  FAIL  Team: the room's shares are not summed against their pools anywhere")
    else:
        v = ui.View(payload=Scenario().content_json())
        audit = ui.team_pools(v).filter(pl.col("team") == opened)
        keys = {ui.POOL_NAME.get(p.name, ui.label(p.name)): p.name for p in opportunity.POOLS}
        checked = 0
        for r in sums[0].to_dict(orient="records"):
            name = keys.get(str(r["pool"]))
            seen = audit.filter(pl.col("pool") == name) if name else pl.DataFrame()
            if name is None or seen.is_empty():
                bad += 1
                print(f"  FAIL  '{r['pool']}' is not a pool the audit knows about")
            # the screen is rounded to three decimals, which is the tolerance here and nothing more: a
            # sum over the wrong player set, or one left unweighted, is out by tenths rather than by a
            # rounding step
            elif abs(float(r["as run"]) - float(seen["raw_sum"][0])) > 1e-3:
                bad += 1
                print(f"  FAIL  {r['pool']}: the sheet sums to {float(r['as run']):.4f} as run, the "
                      f"audit says {float(seen['raw_sum'][0]):.4f}")
            elif float(r["room"]) > float(r["roster"]) + 1e-9:
                bad += 1
                print(f"  FAIL  {r['pool']}: the room claims {float(r['room']):.3f} of a roster that "
                      f"only claims {float(r['roster']):.3f}")
            else:
                checked += 1
        if checked:
            print(f"  ok    {checked} pool sums tie to {opened}'s own audit, each room inside its roster")

    # the way to the next team, which is the difference between one pass over the league and thirty-two
    # trips back to a dropdown
    order = sorted(overrides.run(Scenario()).board["team"].unique().to_list())
    nxt = [b for b in at.button if (b.key or "") == "team:nav:next"]
    if not nxt:
        bad += 1
        print("  FAIL  Team: no next-team button")
    else:
        after = nxt[0].click().run()
        moved = after.session_state[ui.FOCUS_TEAM]
        want = order[(order.index(opened) + 1) % len(order)]
        if after.exception or moved != want:
            bad += 1
            print(f"  FAIL  next moved {opened} to {moved}, wanted {want}"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    next moved {opened} to {moved} and the page redrew on it")

    # and it skips what is already done, on the same definition of done the league page uses
    sc = Scenario().set(Override("team", order[1], "plays", "multiply", 1.05))
    at = run_page(TEAM, sc)
    fresh = [b for b in at.button if (b.key or "") == "team:nav:fresh"]
    if not fresh:
        bad += 1
        print("  FAIL  Team: no next-untouched button")
    else:
        after = fresh[0].click().run()
        landed = after.session_state[ui.FOCUS_TEAM]
        if after.exception or landed != order[2]:
            bad += 1
            print(f"  FAIL  next untouched landed on {landed} with {order[1]} already edited, "
                  f"wanted {order[2]}"
                  + "".join(f"\n        {e.value}" for e in after.exception))
        else:
            print(f"  ok    next untouched skipped the edited {order[1]} and opened {landed}")
    return bad


def flush() -> None:
    """Drop the cached runs between passes, because fourteen passes will not fit in memory at once.

    One composed season pickles to about 40 MB and every pass makes several scenarios; the app bounds
    each cache at a handful of entries, which is right for a server that holds one user's recent work and
    still adds up to gigabytes over a whole harness run. The symptom without this is an allocation failure
    partway down the run -- on a machine with a browser open, a pass that has nothing wrong with it.
    """
    ui.st.cache_data.clear()
    gc.collect()


def main() -> int:
    passes: list[tuple[str, object]] = []

    print("baseline")
    bad = check("baseline", None)
    flush()

    sc, top = demo()
    print(f"\nwith a scenario: {top['player']} of {top['team']}, k_scale 2")
    bad += check("scenario", sc)
    flush()

    passes = [
        ("interactions", lambda: interactions(sc)),
        ("ranges", ranges),
        ("interface", interface),
        ("list and panel", panels),
        ("evidence", evidence),
        ("depth", depth),
        ("off the roster", off_roster),
        ("population", population),
        ("each room's sheet", sheet),
        ("the batched editor", bench),
        ("what if", whatif),
        ("one game", games),
        ("the league", league),
        ("the room sheet", room),
    ]
    for name, run in passes:
        print(f"\n{name}")
        bad += run()
        flush()
    print("\nOK" if not bad else f"\n{bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
