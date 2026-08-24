"""Run every page in a real Streamlit session and fail on any exception.

`streamlit run` is not scriptable, but `st.testing.v1.AppTest` is: it runs a page with a genuine
session, so the widgets hold state, `st.rerun()` reruns, and the scenario in `st.session_state`
behaves the way it does in a browser. That is the difference from executing a page as a bare script --
bare mode gives every widget its default and never reruns, so it cannot test an edit at all.

Four passes:

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

    python scripts/smoke_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from streamlit.testing.v1 import AppTest              # noqa: E402

from src.model import overrides                       # noqa: E402
from src.model.overrides import Override, Scenario     # noqa: E402

PAGES = ["app/Home.py", "app/pages/1_Player.py", "app/pages/2_Team.py", "app/pages/3_Matchups.py",
         "app/pages/4_Adjustments.py", "app/pages/5_Exports.py", "app/pages/6_Diagnostics.py"]
LIVE = "scenario"


def run_page(page: str, scenario: Scenario | None = None) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=300)
    if scenario is not None:
        at.session_state[LIVE] = scenario
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

    at = run_page("app/Home.py")
    at.sidebar.toggle[0].set_value(False).run()
    got = at.session_state[LIVE]
    if got.league.get("normalize_pools") is not False:
        bad += 1
        print(f"  FAIL  the sidebar toggle did not reach the scenario: {got.league}")
    else:
        print(f"  ok    normalisation off is recorded, digest {got.digest}")

    at = run_page("app/pages/4_Adjustments.py", sc)
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

    at = run_page("app/pages/4_Adjustments.py")
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

    at = run_page("app/pages/4_Adjustments.py", sc)
    clears = [b for b in at.button if b.label == "Clear all"]
    if clears:
        clears[0].click().run()
        if not at.session_state[LIVE].is_baseline:
            bad += 1
            print("  FAIL  clear all left edits behind")
        else:
            print("  ok    clear all goes back to the engine's own answer")
    return bad


def ranges() -> int:
    """Flip the on-demand Monte Carlo switch and check the ranges reach the screen.

    The draws are wound down to the smallest option first: this is checking that the wiring produces a
    floor and a ceiling, and ten thousand seasons would prove exactly the same thing far more slowly.
    """
    bad = 0
    for page in ("app/Home.py", "app/pages/1_Player.py"):
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
        shown = "\n".join(str(m.value) for m in at.metric) + "".join(
            " ".join(d.value.columns) for d in at.dataframe if hasattr(d.value, "columns")
        )
        want = ("Floor (P5)" if "Player" in page else "volatility")
        labels = " ".join(m.label for m in at.metric) + shown
        if want not in labels:
            bad += 1
            print(f"  FAIL  {page}: ranges on but no {want!r} on screen")
        else:
            print(f"  ok    {page}  ranges on, {len(at.dataframe)} tables, {len(at.metric)} metrics")
    return bad


def main() -> int:
    print("baseline")
    bad = check("baseline", None)

    sc, top = demo()
    print(f"\nwith a scenario: {top['player']} of {top['team']}, k_scale 2")
    bad += check("scenario", sc)

    print("\ninteractions")
    bad += interactions(sc)

    print("\nranges")
    bad += ranges()
    print("\nOK" if not bad else f"\n{bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
