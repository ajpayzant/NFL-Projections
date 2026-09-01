"""Chrome, cached engine access and formatting. No arithmetic lives here.

The rule this module exists to keep: **a page never computes a projected number.** Every value on
screen comes from `src.model`, so the app cannot drift from the engine the backtest scored, and
anything a user disputes can be reproduced from the command line without Streamlit running.

Since the override layer went in there is a second rule of the same kind: **a page never reads an
unedited frame.** `overrides.run` builds the whole projection under one scenario and hands back every
frame together, and `view()` is the only way a page gets one. A page that read the share frame from
the scenario and the team environment from the baseline would be showing a projection nobody ran.

What is here is the three things a page does need:

- **One control surface.** `controls()` renders the sidebar and returns a `View`: the live scenario's
  content as canonical JSON, plus the season. Content, not the object, because it is the cache key --
  and content only, so renaming a scenario does not throw away a projection.
- **Cached loaders.** Composing a season takes a few seconds and Streamlit reruns the script on every
  widget change, so the run is `st.cache_data` keyed on the `View`. Switching pages recomputes
  nothing; changing a knob recomputes once.
- **Formatting.** Rounding, percentages and the table call in one place, so no page invents its own
  float format.
- **A vocabulary to draw with.** `tiles`, `meter`, `chips`, `section`, `pick_from` and `focus_table` are
  the shapes a page is built from, so a headline number, a number-inside-a-range and a fact that is
  either true or absent each look the same everywhere and different from each other. The app was
  ninety-three `st.dataframe` calls and nothing else, which reads as "every number here is equally
  important" -- the opposite of what a projection needs.
- **The evidence behind an override.** Everywhere a number can be changed, what is known about it is on
  the same screen: the engine's estimate against what actually ran, the player's own record season by
  season, what his job is worth to everyone who has held it, his league percentile, and his room on the
  same metric. `metric_values`, `metric_spread`, `season_wide`, `with_evidence`, `room_panel` and
  `override_panel` are that layer, and it is not a fourth thing so much as the first rule applied to
  editing: an override is a claim about a number, and a claim needs its evidence beside it.

`streamlit run app/Home.py`
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

# Streamlit puts the main script's directory on `sys.path`, not the repo root, so `src` has to be
# found deliberately. Doing it here means a page only ever has to `import ui`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    LAST_COMPLETE_SEASON,
    POSITIONS,
    PROJ_SEASON,
    REG_WEEKS,
    Settings,
)
from src.data import history, lake  # noqa: E402
from src.model import compose, efficiency, opportunity, overrides, roster, simulate  # noqa: E402
from src.model.overrides import Override, Scenario  # noqa: E402

SCORINGS = ("ppr", "half_ppr", "standard", "ppr_6td")
UPDATE_STAMP = ROOT / "build" / "review" / "last_update.json"  # written by scripts/update_data.py
HISTORY_VIEW = tuple(range(2021, LAST_COMPLETE_SEASON + 1))   # the seasons a player page shows
LIVE = "scenario"                                             # the session-state key
RESTORED = "scenario:restored_from"     # the saved name this session opened on, if it opened on one
SAVED_AT = "scenario:saved_at"          # when autosave last wrote, so the sidebar can say so
SAVE_ERROR = "scenario:save_error"      # why it could not, if it could not
UPLOADED = "scenario:uploaded"           # the file_id of the last uploaded scenario, so it lands once


# --------------------------------------------------------------------------- #
# the live scenario
# --------------------------------------------------------------------------- #
_fallback: dict = {}


def _state() -> dict:
    """Session state, or a module dict when there is no session -- which is bare mode, i.e. a test."""
    try:
        st.session_state[LIVE]  # noqa: B018 -- the access is the probe
    except KeyError:
        return st.session_state
    except Exception:
        return _fallback
    return st.session_state


def live() -> Scenario:
    """The scenario being edited. One object, held in session state, replaced on every edit.

    On a session's first call there is nothing in state, and what goes in is *the work that was left*:
    session state does not survive a browser reload or a server restart, so a scenario that only ever
    lived here would be lost by either. `overrides.last_used` names the file autosave wrote to, and it
    is read back once per session rather than polled, so nothing on disk can move under the page.
    """
    state = _state()
    if LIVE not in state:
        state[LIVE] = _restored()
    return state[LIVE]


def _restored() -> Scenario:
    """The scenario a new session should open on: the last one autosave wrote, or the baseline."""
    try:
        name = overrides.last_used()
        if name:
            was = overrides.load(name)
            _state()[RESTORED] = name
            return was
    except Exception:                     # noqa: BLE001 -- a bad file must not stop the app opening
        pass
    return Scenario()


def set_live(scenario: Scenario) -> None:
    """Replace the live scenario and put it on disk, in that order.

    Autosave is here rather than in `edit` because this is the one funnel: every bulk write, every
    drop, every league toggle and every page that loads a saved scenario comes through it. A scenario
    with no edits is not written -- there is nothing to lose, and a file called `working` that says
    "no edits" would be a worse thing to restore than nothing at all.
    """
    _state()[LIVE] = scenario
    if scenario.is_baseline:
        return
    named = scenario if scenario.name != overrides.BASELINE else scenario.rename(overrides.WORKING)
    if named.name != scenario.name:
        _state()[LIVE] = named
    try:
        overrides.save(named)
        _state()[SAVED_AT] = named.updated or ""
    except OSError as exc:                # a full or read-only disk is worth saying out loud, once
        _state()[SAVE_ERROR] = str(exc)


def edit(*items: Override) -> None:
    """Record edits against the live scenario. Every page that changes a number comes through here."""
    set_live(live().set(*items))


def reset(**where) -> None:
    set_live(live().clear(**where))


def drop_edit(level: str, key_value: str, field_name: str, week: int | None = None) -> None:
    """Drop exactly one edit, without taking the week-level edits on the same field with it.

    `Scenario.clear` reads a missing argument as a wildcard, which is right for "clear this player" and
    wrong for "drop this one row": clearing a season-wide `target_share` with `week=None` matches every
    week-level `target_share` too. Since a season edit and a week edit on one field are now routinely
    made side by side, the week ones are lifted out and put back.
    """
    if week is not None:
        reset(level=level, key=key_value, field_name=field_name, week=week)
        return
    keep = [o for o in live().items
            if o.level == level and o.key == key_value and o.field == field_name
            and o.week is not None]
    reset(level=level, key=key_value, field_name=field_name)
    if keep:
        edit(*keep)


def released_ids() -> list[str]:
    """Everybody the live scenario has taken off a roster, in the order the edits were made.

    Kept as one reader because a release is the only edit whose subject is *absent* from the run it
    produced: nothing downstream has a row for him, so the scenario is the only place the app can find
    out he was ever there, and every surface that wants to say so has to ask the same question.
    """
    return [o.key for o in live().items
            if o.level == "player" and o.field == overrides.ROSTER_FIELD and o.week is None
            and o.mode == "set" and float(o.value) == overrides.OFF_ROSTER]


def release(*player_ids: str) -> None:
    """Take players off the roster they are listed on, for the whole season.

    `base=1.0` because what is being overridden is the roster itself, and the roster says he is on it.
    """
    edit(*[Override("player", pid, overrides.ROSTER_FIELD, "set", overrides.OFF_ROSTER, base=1.0,
                    note="off the roster")
           for pid in player_ids])


def reinstate(*player_ids: str) -> None:
    """Put released players back, by dropping the release rather than writing anything against it."""
    for pid in player_ids:
        drop_edit("player", pid, overrides.ROSTER_FIELD)


@dataclass(frozen=True)
class View:
    """What a projection depends on, as two hashable primitives: the scenario's content, and when.

    `payload` is `Scenario.content_json` -- the edits and the league patch, canonically ordered, with
    the name and the timestamps stripped out. That makes it exactly the cache key it should be.
    """

    payload: str
    season: int = PROJ_SEASON

    @property
    def scenario(self) -> Scenario:
        return Scenario.from_json(self.payload)

    @property
    def settings(self) -> Settings:
        return overrides.settings_for(self.scenario)

    @property
    def scoring(self) -> str:
        return self.scenario.scoring or "ppr"


def view() -> View:
    return View(payload=live().content_json(), season=PROJ_SEASON)


def controls(title: str, icon: str = "🏈") -> View:
    """Page chrome and the sidebar. Every page starts with this and gets the same knobs."""
    st.set_page_config(page_title=f"{title} — NFL {PROJ_SEASON}", page_icon=icon, layout="wide")
    style()
    st.title(title)
    sc = live()
    was = sc.league

    with st.sidebar:
        st.caption(f"{PROJ_SEASON} season projections")
        scoring = st.selectbox(
            "Scoring", SCORINGS, index=SCORINGS.index(sc.scoring or overrides.DEFAULT_SCORING),
            format_func=scoring_label,
        )
        normalize = st.toggle(
            "Normalise pools", value=sc.league.get("normalize_pools", True),
            help="Rescale each exclusive team pool (targets, carries, red-zone work, touchdowns) to "
                 "what it was measured to sum to. Off shows the raw estimated shares.",
        )
        context = st.toggle(
            "Per-game context", value=sc.league.get("use_context_factors", True),
            help="Apply the venue / rest / script / defence chain per game. Off projects every game "
                 "at the team's season rate — which is all the workbook could do.",
        )
        market = st.toggle(
            "Use posted lines", value=sc.league.get("market_weight") != 0.0,
            help="Blend the Vegas line into a game's scoring level where one is posted. Off uses the "
                 "model's own team estimate everywhere.",
        )
        # written back only when something moved: patching on every rerun would restamp `updated`
        # and, worse, make the widgets fight a scenario the Edits page had just loaded.
        edited = sc.patch_league(scoring=scoring, normalize_pools=normalize,
                                 use_context_factors=context,
                                 market_weight=None if market else 0.0)
        if edited.league != was or edited.scoring != sc.scoring:
            sc = edited
            set_live(sc)

        st.divider()
        _scenario_panel(sc)
        st.divider()
        _quick_edit()
        st.divider()
        _freshness_panel()
    # outside the sidebar on purpose: a dialog is raised from the page body, and putting the call here
    # means every page in the app has the editor without a line of its own asking for it
    v = view()
    bench_dialog(v)
    return v


def _quick_edit() -> None:
    """Any player's numbers, from any page, without losing the page.

    The sidebar is the one part of the app that is on screen everywhere, so it is where "I want to change
    something about that man" belongs. Wrapped in try/except and drawn last because it is the only piece
    of chrome that needs the projection to have run: a page that cannot list players is still a page worth
    seeing, and History's own report card is one of them.
    """
    try:
        who = board(view()).select("player_id", "player", "position", "team").head(1200)
    except Exception:                     # noqa: BLE001 -- chrome must not take a page down
        return
    if who.is_empty():
        return
    names = {r["player_id"]: f"{r['player']} · {r['position']} · {r['team']}"
             for r in who.rows(named=True)}
    with st.expander("✎ Adjust a player", expanded=False):
        pick = st.selectbox("Player", list(names), format_func=lambda p: names[p], index=None,
                            placeholder="type a name", key="sidebar:bench:pick",
                            label_visibility="collapsed")
        st.caption("Every number behind his projection in one list — for the season or for chosen weeks.")
        if st.button("Open the editor", width="stretch", disabled=pick is None,
                     key="sidebar:bench:open"):
            open_bench(str(pick))


def _clock(stamp: str) -> str:
    """An ISO timestamp as the only part of it a person reading a sidebar wants: the time."""
    return stamp[11:16] if len(stamp) >= 16 else stamp


def _scenario_panel(sc: Scenario) -> None:
    """The scenario's identity, size and save state, on every page. An edit is never invisible.

    This is the sidebar rather than a page because it is the one piece of chrome on screen everywhere,
    and because naming, saving and loading a scenario used to live on one tab of one page -- which is
    how a session's work came to be lost. Autosave means Save is not a thing anybody has to remember;
    what is still worth a control is *which* scenario is being written, and getting an old one back.
    """
    st.markdown(f"**Scenario** `{sc.name}`")
    state = _state()
    if sc.is_baseline:
        st.caption("no edits — this is the engine's own answer")
    else:
        applied = len(sc.items)
        bits = [f"{applied} edit{'s' if applied != 1 else ''}"]
        if sc.k_scale != 1.0:
            bits.append(f"k×{sc.k_scale:g}")
        st.caption(" · ".join(bits) + f" · `{sc.digest}`")
        if state.get(SAVE_ERROR):
            st.error(f"not saved: {state[SAVE_ERROR]}", icon="⚠️")
        else:
            st.caption(f"✓ saved to `{sc.name}` at {_clock(str(state.get(SAVED_AT) or sc.updated))}")
    if state.get(RESTORED) and not sc.is_baseline:
        st.caption(f"picked up from where `{state[RESTORED]}` was left")

    with st.expander("Scenarios", expanded=False):
        if not sc.is_baseline:
            renamed = st.text_input(
                "Name", value=sc.name, key="scenario:name",
                help="What this set of edits is filed under. Renaming writes a new file and leaves the "
                     "old one alone, which is how a variant is made.")
            if renamed and renamed != sc.name:
                set_live(sc.rename(renamed))
                st.rerun()
        saved = [n for n in overrides.names() if n != sc.name]
        pick = st.selectbox("Open a saved one", ["—", *saved], index=0, key="scenario:open",
                            help="Loads it as the live scenario. The current one is already on disk "
                                 "under its own name, so nothing is lost by switching.")
        if pick != "—":
            set_live(overrides.load(str(pick)))
            _state()[RESTORED] = str(pick)
            st.rerun()
        if st.button("Start a new one", width="stretch", key="scenario:new",
                     help="Back to the engine's own answer. What is on disk stays there."):
            _state()[LIVE] = Scenario()
            _state().pop(RESTORED, None)
            st.rerun()
        _transfer_panel(sc)
    try:
        st.page_link("pages/7_Edits.py", label="Every edit →")
    except Exception:                     # noqa: BLE001
        # A page link resolves against the app's entrypoint. Running one page on its own -- which is
        # what the smoke harness does -- means there is no page set to resolve against, and the link
        # is not worth failing a page over.
        st.caption("Edits")


def _transfer_panel(sc: Scenario) -> None:
    """A scenario as a file you hold, either direction.

    Autosave already means no edit is lost to a closed tab, but everything it writes is under
    `data/scenarios`, which is deliberately not in git -- so on this machine and nowhere else. A
    downloaded scenario is the copy that survives a rebuilt laptop, goes in a backup, or is handed to
    somebody else to open; an uploaded one is that in reverse. It is the same JSON `overrides.save`
    writes, so a file from here can also simply be dropped into `data/scenarios`.

    Overrides travel and projections do not, on purpose: an override is "this man plays 12 games", which
    is still true against next week's numbers, while a projection is an answer to data that has since
    moved. Opening a file re-runs it against whatever the lake now holds.
    """
    st.divider()
    st.download_button(
        "⬇ Download this scenario", data=sc.to_json(), mime="application/json",
        file_name=f"{overrides.path(sc.name).stem}.json", width="stretch", key="scenario:download",
        help="Every edit and league setting as one JSON file. Keep it as a backup, or open it on "
             "another machine.")
    got = st.file_uploader("Upload a scenario file", type="json", key="scenario:upload",
                           help="A file downloaded from here. It becomes the live scenario and is "
                                "saved under its own name, so nothing already on disk is overwritten "
                                "unless it shares that name.")
    if got is None:
        return
    # The uploader holds its file across reruns, so without a "seen this one" marker the first edit
    # made after an upload would be immediately overwritten by the upload again, on the very next
    # rerun. `file_id` is per selection, so re-choosing the same file deliberately still works.
    seen = _state().get(UPLOADED)
    if seen == getattr(got, "file_id", None):
        return
    try:
        loaded = Scenario.from_json(got.getvalue().decode("utf-8"))
    except Exception as exc:              # noqa: BLE001 -- any malformed file, named as such
        st.error(f"not a scenario file: {type(exc).__name__}", icon="⚠️")
        return
    _state()[UPLOADED] = getattr(got, "file_id", None)
    set_live(loaded)
    _state()[RESTORED] = loaded.name
    st.rerun()


def scoring_label(name: str) -> str:
    return {"ppr": "Full PPR", "half_ppr": "Half PPR", "standard": "Standard",
            "ppr_6td": "PPR, 6-pt passing TD"}.get(name, name)


def _freshness_panel() -> None:
    status = freshness()
    stale = status.filter(pl.col("missing_proj_season"))
    oldest = status["age_days"].max()
    st.caption(f"lake read from `{lake.LAKE}`")
    st.caption(f"oldest table {oldest:.0f} days old" if oldest is not None else "no tables found")
    if not stale.is_empty():
        st.warning(f"missing {PROJ_SEASON}: {', '.join(stale['table'].to_list())}", icon="⚠️")
    _update_panel()


# When this process started, which is what makes a scheduled update visible or not: the projection is
# `st.cache_data`, the cache is in memory, and `lake.clear_cache()` in a *different* process cannot
# reach it. So a server that was up when the nightly refresh ran keeps serving yesterday's rosters
# until it is restarted, silently, and the one thing worse than stale data is stale data that looks
# current.
_STARTED = datetime.now().astimezone()


def last_update() -> dict:
    """What `scripts/update_data.py` wrote when it last finished. Never cached: the point is to notice
    a file that changed *after* the caches were filled."""
    try:
        return json.loads(UPDATE_STAMP.read_text(encoding="utf-8"))
    except Exception:                     # noqa: BLE001 -- no stamp yet is the normal first-run case
        return {}


def _update_panel() -> None:
    stamp = last_update()
    when = str(stamp.get("finished") or "")
    if not when:
        return
    try:
        finished = datetime.fromisoformat(when)
    except ValueError:
        return
    mode = "full rebuild" if stamp.get("mode") == "full" else "refresh"
    if finished > _STARTED:
        st.warning(f"data {mode} finished {_clock(when)}, after this server started — restart it to "
                   "use the new numbers", icon="🔄")
    else:
        st.caption(f"last data {mode} {when[:10]} {_clock(when)}")


# --------------------------------------------------------------------------- #
# the projection, under one scenario
# --------------------------------------------------------------------------- #
# How many *scenarios* worth of anything to keep. `st.cache_data` never evicts unless it is told a
# bound, and a session spent making overrides is a session producing one new scenario digest per edit
# -- so an unbounded cache on anything keyed by a `View` grows for as long as the tab is open, and the
# process dies rather than the cache. One run pickles to ~40 MB and the derived frames are on top of
# that, so this is deliberately small: past the last few scenarios the run they were cut from has been
# evicted anyway, and recomputing is ~0.7s.
SCENARIO_ENTRIES = 8


@st.cache_data(show_spinner="projecting the season", max_entries=SCENARIO_ENTRIES)
def _run(payload: str, season: int) -> overrides.Run:
    sc = Scenario.from_json(payload)
    prev = actuals((LAST_COMPLETE_SEASON,), sc.scoring or "ppr")
    return overrides.run(sc, season, prev)


def projection(v: View) -> overrides.Run:
    """Every frame, produced together under one scenario. The only door into the engine."""
    return _run(v.payload, v.season)


def projection_of(scenario: Scenario, season: int = PROJ_SEASON) -> overrides.Run:
    """Any scenario's run, not only the live one -- comparing two needs both at once."""
    return _run(scenario.content_json(), season)


def baseline(v: View) -> overrides.Run:
    """The same season with no edits, for the diff. Cached separately and shared by every page."""
    bare = Scenario(scoring=v.scoring)
    return _run(bare.content_json(), v.season)


# thin accessors, so a page reads a frame by name rather than reaching into the run
def board(v: View) -> pl.DataFrame:
    return projection(v).board


def weekly(v: View) -> pl.DataFrame:
    return projection(v).weekly


def season_frame(v: View) -> pl.DataFrame:
    return projection(v).season_frame


def environment(v: View) -> pl.DataFrame:
    return projection(v).env


def opportunity_frame(v: View) -> pl.DataFrame:
    return projection(v).opp


def participation(v: View) -> pl.DataFrame:
    return projection(v).part


def shares_wide(v: View) -> pl.DataFrame:
    return projection(v).shares


def rates_wide(v: View) -> pl.DataFrame:
    return projection(v).rates


def roster_frame(v: View) -> pl.DataFrame:
    return projection(v).roster


def provenance(v: View) -> pl.DataFrame:
    return projection(v).provenance


# --------------------------------------------------------------------------- #
# everything else the engine can be asked for
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="reading the lake")
def freshness() -> pl.DataFrame:
    return lake.status()


@st.cache_data(show_spinner=False)
def staleness() -> dict:
    """Whether `processed/` has kept up with the season, not merely when it was last touched."""
    return lake.staleness(PROJ_SEASON)


# --------------------------------------------------------------------------- #
# the backtest, read from disk
# --------------------------------------------------------------------------- #
# Read rather than run: a full backtest is minutes of composition per variant, and it is a command-line
# job whose output is an artifact. The page reports what the last run found and says how old it is; a
# page that re-ran it would be a page nobody waited for.
@st.cache_data(show_spinner="reading the backtest")
def backtest_summary() -> pl.DataFrame:
    from src.model import backtest

    return (pl.read_parquet(backtest.SUMMARY_PATH) if backtest.SUMMARY_PATH.is_file()
            else pl.DataFrame())


@st.cache_data(show_spinner="reading the backtest")
def backtest_players() -> pl.DataFrame:
    from src.model import backtest

    return (pl.read_parquet(backtest.PLAYER_PATH) if backtest.PLAYER_PATH.is_file()
            else pl.DataFrame())


@st.cache_data(show_spinner="measuring calibration")
def backtest_calibration() -> dict[str, pl.DataFrame]:
    from src.model import backtest

    players = backtest_players()
    if players.is_empty():
        return {}
    return backtest.calibration(players)


def backtest_age() -> float | None:
    """Days since the backtest artifacts were written, so a stale number is read as one."""
    from datetime import UTC, datetime

    from src.model import backtest

    if not backtest.SUMMARY_PATH.is_file():
        return None
    when = datetime.fromtimestamp(backtest.SUMMARY_PATH.stat().st_mtime, UTC)
    return (datetime.now(UTC) - when).total_seconds() / 86400.0


@st.cache_data(show_spinner="projecting team season shape", max_entries=SCENARIO_ENTRIES)
def shape(v: View) -> pl.DataFrame:
    from src.model import team

    return team.shape_wide(v.season, v.settings)


@st.cache_data(show_spinner="reading the shares behind each answer", max_entries=SCENARIO_ENTRIES)
def share_detail(v: View) -> pl.DataFrame:
    from src.model import estimate

    sc = v.scenario
    return estimate.estimate(opportunity.SHARE_METRICS, roster_frame(v), v.season, v.settings,
                             overrides.fitted_for(sc))


@st.cache_data(show_spinner="reading the rates behind each answer", max_entries=SCENARIO_ENTRIES)
def rate_detail(v: View) -> pl.DataFrame:
    return efficiency.rate_detail(v.season, v.settings, ros=roster_frame(v),
                                  fitted=overrides.fitted_for(v.scenario))


@st.cache_data(show_spinner="reading the participation behind each answer",
               max_entries=SCENARIO_ENTRIES)
def participation_detail(v: View) -> pl.DataFrame:
    return roster.participation_detail(v.season, v.settings, ros=roster_frame(v),
                                       fitted=overrides.fitted_for(v.scenario))


@st.cache_data(show_spinner="auditing the pools", max_entries=SCENARIO_ENTRIES)
def pool_report(v: View) -> pl.DataFrame:
    return opportunity.pool_report(v.season, v.settings, opportunity_frame(v))


@st.cache_data(show_spinner="auditing each team's pools", max_entries=SCENARIO_ENTRIES)
def team_pools(v: View) -> pl.DataFrame:
    return opportunity.team_pool_sums(v.season, v.settings, shares=shares_wide(v),
                                      part=participation(v))


@st.cache_data(show_spinner="reconciling players against their teams", max_entries=SCENARIO_ENTRIES)
def reconciliation(v: View) -> pl.DataFrame:
    return compose.team_check(weekly(v), v.season, v.settings, env=environment(v))


# --------------------------------------------------------------------------- #
# the league: records, team totals, and how much of it has been looked at
# --------------------------------------------------------------------------- #
# All four of these are the same rule as everywhere else -- `src.model.standings` does the arithmetic and
# the page does none of it. What is new is the direction of reading: every other surface starts from a
# player and works up, and a projection is checked from the top down. A roster that looks like a
# contender and comes out 5-12 is wrong somewhere, and no per-player table says so.
@st.cache_data(show_spinner="projecting records", max_entries=SCENARIO_ENTRIES)
def standings(v: View) -> pl.DataFrame:
    from src.model import standings as model

    return model.standings(environment(v))


@st.cache_data(show_spinner="projecting every game", max_entries=SCENARIO_ENTRIES)
def game_grid(v: View) -> pl.DataFrame:
    from src.model import standings as model

    return model.game_margins(environment(v))


@st.cache_data(show_spinner="scoring the win model")
def standings_accuracy() -> dict[str, float]:
    """How the win probability did on the seasons it was fitted from. Not scenario-dependent."""
    from src.model import standings as model

    return model.accuracy()


# What a team is projected to *do*, summed off the same board the player pages read, so a total on this
# page and the players behind it can never disagree. `points` comes from the environment instead,
# because scoring is a team-level estimate the pool chain divides rather than something players add up
# to -- the reconciliation page is where those two are compared.
TEAM_STAT_TOTALS = ("attempts", "completions", "passing_yards", "passing_tds", "interceptions",
                    "sacks", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
                    "receiving_yards", "receiving_tds", "fumbles_lost", "fantasy_points")


@st.cache_data(show_spinner="totalling all 32 teams", max_entries=SCENARIO_ENTRIES)
def team_projections(v: View) -> pl.DataFrame:
    """One row per team: its season totals, its scoring level, and its projected record."""
    have = [c for c in TEAM_STAT_TOTALS if c in board(v).columns]
    totals = board(v).group_by("team").agg(
        pl.len().alias("players"),
        *[pl.col(c).sum().alias(c) for c in have],
    )
    env_cols = [c for c in ("plays", "dropbacks", "pass_attempts", "carries", "dropback_rate",
                            "yards_per_attempt", "yards_per_carry", "success_rate") if
                c in environment(v).columns]
    team_env = environment(v).group_by("team").agg(
        pl.col("implied_points").sum().alias("points_for"),
        *[(pl.col(c).mean() if c.endswith(("_rate", "_attempt", "_carry")) else pl.col(c).sum())
          .alias(f"team_{c}") for c in env_cols],
    )
    record = standings(v).select("team", "division", "conference", "expected_wins", "expected_losses",
                                "points_against", "point_margin", "points_per_game",
                                "allowed_per_game")
    out = record.join(team_env, on="team", how="left").join(
        totals, on="team", how="left", suffix="_players")
    return out.sort("expected_wins", descending=True)


# The one thing on the league page that is about the *user* rather than the season: going team by team is
# thirty-two sittings, and the question between sittings is which ones are done. Derived from the
# scenario's own edits rather than from a checkbox, so there is nothing extra to keep in step and
# nothing to lose -- a team with edits on it has been looked at, and a team with none has not.
def coverage(v: View) -> pl.DataFrame:
    """Per team: how many edits it carries, on how many players, and when it was last touched."""
    from src.model import standings as model

    sc = v.scenario
    of_player = dict(board(v).select("player_id", "team").iter_rows())
    rows: dict[str, dict] = {t: {"team": t, "division": model.DIVISION_OF.get(t, "—"), "edits": 0,
                                 "players": set(), "team_edits": 0, "week_edits": 0, "last": ""}
                             for t in sorted(model.DIVISION_OF)}
    for o in sc.items:
        team = o.key if o.level == "team" else of_player.get(o.key)
        if team not in rows:
            continue
        r = rows[team]
        r["edits"] += 1
        if o.level == "team":
            r["team_edits"] += 1
        else:
            r["players"].add(o.key)
        if o.week is not None:
            r["week_edits"] += 1
        r["last"] = max(r["last"], o.at or "")
    return pl.DataFrame([
        {"team": r["team"], "division": r["division"], "reviewed": r["edits"] > 0,
         "edits": r["edits"], "players_edited": len(r["players"]), "team_edits": r["team_edits"],
         "week_edits": r["week_edits"], "last_touched": r["last"]}
        for r in rows.values()
    ]).sort(["reviewed", "division", "team"], descending=[True, False, False])


@st.cache_data(show_spinner="reading what actually happened")
def actuals(seasons: tuple[int, ...], scoring: str) -> pl.DataFrame:
    """Played seasons, scored the same way the projection is. Keyed on scoring, not on the whole
    `View`: an override cannot change what a player did in 2023."""
    return history.player_seasons(seasons, Settings().with_scoring(scoring).scoring)


@st.cache_data(show_spinner="reading defensive history")
def defence_by_position(season: int = LAST_COMPLETE_SEASON) -> pl.DataFrame:
    return history.defense_by_position((season,))


@st.cache_data(show_spinner="comparing against the baseline", max_entries=SCENARIO_ENTRIES)
def board_diff(v: View) -> pl.DataFrame:
    return overrides.diff(baseline(v).board, board(v))


@st.cache_data(show_spinner="comparing against the baseline", max_entries=SCENARIO_ENTRIES)
def board_diff_summary(v: View) -> pl.DataFrame:
    return overrides.summary(baseline(v).board, board(v))


# --------------------------------------------------------------------------- #
# the simulation, on demand
# --------------------------------------------------------------------------- #
SIM_DRAWS = (2_000, 5_000, 10_000, 25_000)
SIM_STATE = "sim_on"                       # the session-state key for "the user asked for ranges"


@st.cache_data(show_spinner="simulating the season", max_entries=4)
def _simulate(payload: str, season: int, draws: int, detail: tuple[str, ...]) -> simulate.Sim:
    """Ten thousand seasons of the *current* weekly frame, edits and all.

    Keyed on the scenario's content exactly as the projection is, which is what makes a floor move when
    a share moves: an edit is a different payload, so it is a different simulation rather than a cached
    range describing a frame nobody is looking at any more.

    `detail` is in the key because the per-week and per-stat draws are only kept for the players who
    were asked for -- a page wanting one player's distribution should not pay to store every player's.
    """
    v = View(payload=payload, season=season)
    return simulate.run(weekly(v), v.settings, draws=draws, detail=detail)


def simulation(v: View, draws: int = 10_000, detail: tuple[str, ...] = ()) -> simulate.Sim:
    """The simulation for this view. Call it only behind a control -- it is seconds, not milliseconds."""
    return _simulate(v.payload, v.season, int(draws), tuple(detail))


def dispersion() -> simulate.Dispersion:
    return _dispersion()


@st.cache_data(show_spinner=False)
def _dispersion() -> simulate.Dispersion:
    return simulate.load()


def sim_controls(v: View, detail: tuple[str, ...] = (), key: str = "sim") -> simulate.Sim | None:
    """The on-demand switch, and the state of the fit, in one row. `None` until the user asks.

    The plan is explicit that the simulation runs on demand rather than on load, and the reason shows up
    here rather than in the engine: a page that simulated on every rerun would spend ten seconds
    redrawing after a filter change that cannot move a range.
    """
    left, mid, right = st.columns([2, 2, 5])
    with left:
        on = st.toggle(
            "Ranges", value=bool(st.session_state.get(SIM_STATE, False)), key=f"{key}:on",
            help="Simulate the season to get floors, ceilings and volatility. Takes a few seconds.",
        )
        st.session_state[SIM_STATE] = on
    with mid:
        draws = st.select_slider("Draws", SIM_DRAWS, value=10_000, key=f"{key}:draws",
                                 disabled=not on)
    with right:
        st.caption(sim_provenance())
    if not on:
        return None
    return simulation(v, draws=int(draws), detail=detail)


def sim_on() -> bool:
    """Has the reader asked for ranges anywhere on the page?

    A page with tabs wants the simulated frame in more than one place -- the weekly band and the range
    tab are the same run -- but drawing a second `sim_controls` for it would be a second switch to keep
    in step. So the switch stays in one tab and everything else asks this.
    """
    return bool(st.session_state.get(SIM_STATE, False))


def sim_draws(key: str = "sim") -> int:
    """The draw count that switch is set to, so a second reader of the run hits the same cache entry."""
    return int(st.session_state.get(f"{key}:draws") or 10_000)


def sim_provenance() -> str:
    """One line saying where the sigmas came from, because an uncalibrated range is a different claim."""
    d = dispersion()
    if not d.meta.get("fitted"):
        note = ("dispersion **not fitted** — the ranges are the pooled defaults. Run "
                "`python -m src.model.simulate --fit`.")
        return note
    seasons = d.meta.get("seasons") or []
    span = f"{seasons[0]}–{seasons[1]}" if len(seasons) == 2 else "history"
    if not d.meta.get("calibrated"):
        return f"dispersion measured on {span}, **not calibrated** — intervals are likely narrow."
    scale = ", ".join(f"{k} ×{v:g}" for k, v in sorted(d.scale.items()))
    cover = d.meta.get("coverage") or {}
    # named rather than left as "coverage", because the number is only meaningful with its population:
    # it is measured over projected-startable seasons the model had not seen, not over the whole board
    hit = (f" · realised P5–P95 coverage {cover['cover_90']:.2f} against 0.90 on held-out startable "
           f"seasons" if "cover_90" in cover else "")
    return f"dispersion measured on {span}, calibrated ({scale}){hit}"


def range_config(*, per_game: bool = False) -> dict:
    """One column config for every range table, so a floor is formatted the same everywhere."""
    digits = 2 if per_game else 1
    return {
        **fixed(digits, "p5", "p25", "p50", "p75", "p95", "floor", "ceiling", "range",
                "mean", "sim_mean", "sim_sd", "sim_vs_projected", "projected"),
        **percent("boom_rate", "bust_rate", "volatility"),
        **fixed(1, "games_mean", "games_p5", "games_p50", "games_p95"),
    }


RANGE_COLUMNS = ["player", "position", "team", "projected", "p50", "floor", "ceiling", "range",
                 "volatility", "boom_rate", "bust_rate"]


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def rounded(df: pl.DataFrame, digits: int = 2, per_column: dict[str, int] | None = None) -> pl.DataFrame:
    """Round for display. Floats only, so an id or a count is never touched."""
    # One pass, not two: rounding everything to `digits` first and then applying the per-column
    # numbers would silently cap a column that wants *more* decimals than the default -- a share asked
    # for at four decimals inside a table defaulting to two came out at 0.24, and 0.2431 and 0.2449
    # then read as the same 24.00%.
    per = per_column or {}
    return df.with_columns([
        pl.col(c).round(int(per.get(c, digits)))
        for c, dt in zip(df.columns, df.dtypes, strict=True) if dt.is_float()
    ])


def table(
    df: pl.DataFrame,
    digits: int = 2,
    per_column: dict[str, int] | None = None,
    height: int | str = "auto",
    config: dict | None = None,
    order: list[str] | None = None,
    auto: bool = True,
) -> None:
    """One table call for the whole app: named, rounded, index hidden, full width.

    `auto` is the default because the alternative was every page deciding for itself how many decimals
    a share is worth, and getting a different answer -- so the digits and the column name both come from
    `digits_for` and `LABELS` unless the caller says otherwise. A config passed in still wins on the
    columns it names, which is how a bar, a trend line or a per-page label survives.
    """
    per = {c: digits_for(c, digits) for c in df.columns} if auto else {}
    per.update(per_column or {})
    st.dataframe(
        rounded(df, digits, per),
        hide_index=True,
        height=height,
        width="stretch",
        column_config={**(auto_config(df, digits) if auto else {}), **(config or {})},
        column_order=order,
    )


# The name a reader would use, where the column name is not already it. Only the opaque ones are here:
# renaming `targets` to "targets" is noise in a dictionary, and a page that shows `fantasy_points` under
# a heading that says points is not confusing anybody. Abbreviations, engine internals and anything
# whose units are not obvious from the name are.
LABELS = {
    "adot": "air yards per target", "tprr": "targets per route run",
    "proe": "pass rate over expected", "epa_per_play": "EPA per play",
    "epa_per_dropback": "EPA per dropback", "epa_per_rush": "EPA per rush",
    "p_play": "chance he plays", "active_weeks": "share of the season active",
    "own_weight": "weight on his own record", "n": "sample size", "obs": "his own record",
    "k": "shrinkage k", "prior_slot": "his slot's prior", "curve": "draft-slot curve",
    "slot_bucket": "slot he is priced off", "avail_slot": "slot for expected games",
    "depth_slot": "slot on the chart", "depth_tier": "published tier",
    "on_roster": "on this roster",
    "pool_share": "share of his pool", "presence": "chance the job exists",
    "status_factor": "assumed availability", "expected_games": "projected games",
    "games_if_available": "games if fully available", "obs_games": "games he has averaged",
    "prior_games": "his slot's games", "games_own_weight": "weight on his own attendance",
    "n_seasons": "seasons on record", "charted": "on the published chart",
    "team_disagreement": "sources disagree", "years_exp": "years in the league",
    "draft_pick": "draft pick", "is_rookie": "rookie", "alignment": "where he lines up",
    "vs_starter": "vs average starter", "drop_next": "drop to the next man",
    "points_per_game": "points per game", "position_rank": "rank at position",
    "overall_rank": "overall rank", "startable": "startable",
    "has_market": "line posted", "is_home": "at home", "implied_points": "implied points",
    "seconds_per_play": "seconds per play", "rest_days": "days rest", "div_game": "division game",
    "rz_targets": "red-zone targets", "rz_carries": "red-zone carries",
    "rz_target_share": "red-zone target share", "rz_carry_share": "red-zone carry share",
    "late_down_target_share": "third and fourth down target share",
    "short_yardage_carry_share": "short-yardage carry share",
    "inside_5_carry_share": "carry share inside the 5",
    "d_fantasy_points": "points moved", "rank_gain": "places gained",
    "new_fantasy_points": "points after", "base_recorded": "base when edited",
    "base_now": "base now", "used_now": "used now", "gap_pct": "gap %",
    "weeks_listed": "weeks on the report", "weeks_out": "weeks ruled out",
    "weeks_doubtful": "weeks doubtful", "weeks_questionable": "weeks questionable",
    "weeks_dnp": "weeks he did not practise", "weeks_limited": "weeks limited in practice",
    "main_injury": "usual injury", "distinct_injuries": "different injuries",
    "seasons_hurt": "seasons on the report", "attendance": "share of games played",
    "games_played": "games played", "of_possible": "of possible",
    "worst_season_out": "worst season, weeks out", "report": "weeks out by season",
    "review": "worth a look because", "week_rank": "rank at position that week",
    "path": "week by week", "season_points": "season points", "byes": "on a bye",
    "means": "what it means", "availability": "assumed availability", "players": "players",
    "now": "what it is now", "after": "what it becomes", "moves": "players it moves",
    "return_week": "back in week", "games_left": "games left",
    "of_team": "share of the team", "projected": "projected", "claimed": "claimed by the roster",
}

# How many decimals a number is worth. A season projection's second decimal is noise the model does not
# have, a yard is a whole number to read, and a share is kept to four because it is *shown* as a
# percentage and two decimals of a percent is two more of the share.
DIGITS = {
    "fantasy_points": 1, "new_fantasy_points": 1, "d_fantasy_points": 1, "points_per_game": 2,
    "games": 1, "weeks": 1, "expected_games": 1, "games_if_available": 1, "prior_games": 1,
    "obs_games": 1, "vs_starter": 1, "drop_next": 1, "delta_points": 1, "last_points": 1,
    "last_games": 1, "spread": 1, "total": 1, "implied_points": 1, "points": 1, "temp": 0, "wind": 0,
    "p5": 1, "p25": 1, "p50": 1, "p75": 1, "p95": 1, "floor": 1, "ceiling": 1, "range": 1,
    "mean": 1, "sim_mean": 1, "sim_sd": 1, "sim_vs_projected": 1, "projected": 1,
    "projected_games": 1, "games_mean": 1, "games_p5": 1, "games_p50": 1, "games_p95": 1,
    "age": 1, "years_exp": 0, "n": 0, "seconds_per_play": 2, "epa_per_play": 3, "proe": 3,
    "mae": 1, "rmse": 1, "bias": 1, "spearman": 3, "slope": 3, "gap_pct": 1,
    "yards_per_attempt": 2, "yards_per_carry": 2, "yards_per_target": 2, "yards_per_reception": 2,
    "adot": 2, "plays": 1, "dropbacks": 1, "attendance": 3,
    "season_points": 1, "games_left": 0, "return_week": 0,
}

# Shares and probabilities, which are read as percentages. Matched by name because the alternative is a
# per-page decision; guarded by value in `auto_config`, because a column called `_factor` is a ratio
# around one and printing 105% where the chain multiplies by 1.05 would be a different number.
PERCENT_COLUMNS = ("own_weight", "games_own_weight", "p_play", "active_weeks", "presence",
                   "status_factor", "volatility", "attendance", "rate", "pool_share", "share",
                   "cover_90", "cover_50", "below", "above", "never_played", "covered_pct",
                   "of_team")
PERCENT_PARTS = ("_share", "share_", "_rate", "_pct", "participation", "_pctile")


def _percentish(column: str) -> bool:
    return column in PERCENT_COLUMNS or any(p in column for p in PERCENT_PARTS)


def digits_for(column: str, default: int = 2) -> int:
    """The decimals this column is worth, before any page has an opinion about it."""
    if column in DIGITS:
        return DIGITS[column]
    if column in ALL_STATS:
        return stat_digits(column)
    if _percentish(column):
        return 4
    return default


def auto_config(df: pl.DataFrame, digits: int = 2) -> dict:
    """A column config for a whole frame: every column named, every number formatted.

    Non-numeric columns get a name and nothing else. Lists and dates are left alone -- a trend line or a
    timestamp is the caller's to configure, and guessing would overwrite it.
    """
    out: dict = {}
    for c, dt in zip(df.columns, df.dtypes, strict=True):
        if c.endswith("_id"):
            continue
        name = label(c)
        if dt == pl.Boolean:
            out[c] = st.column_config.CheckboxColumn(name)
        elif dt.is_numeric() and _percentish(c) and _unit_scale(df[c]):
            out[c] = st.column_config.NumberColumn(name, format="percent")
        elif dt.is_float():
            out[c] = st.column_config.NumberColumn(name, format=f"%.{digits_for(c, digits)}f")
        elif dt.is_integer():
            out[c] = st.column_config.NumberColumn(name)
        elif dt == pl.String:
            out[c] = st.column_config.TextColumn(name)
    return out


def _unit_scale(values: pl.Series) -> bool:
    """True where every value could be a share of one. What stops a ratio being printed as a percent."""
    if not values.dtype.is_numeric():
        return False
    top = values.drop_nulls().abs().max()
    return top is None or float(top) <= 1.5


def label(column: str) -> str:
    """A column name as a reader would say it."""
    return LABELS.get(column, column.replace("_", " "))


def percent(*columns: str) -> dict:
    """Column config showing a 0-1 share as a percentage, which is how a share is read."""
    return {c: st.column_config.NumberColumn(label(c), format="percent") for c in columns}


def numeric(columns: dict[str, str]) -> dict:
    """`{column: format}` as a Streamlit column config, e.g. `{"fantasy_points": "%.1f"}`."""
    return {c: st.column_config.NumberColumn(label(c), format=fmt) for c, fmt in columns.items()}


def fixed(digits: int, *columns: str) -> dict:
    """The common case of `numeric`: the same number of decimals for a group of columns."""
    return numeric({c: f"%.{digits}f" for c in columns})


def sparkline(column: str, title: str | None = None) -> dict:
    """A list-valued column rendered as a line, for a player's history in one cell."""
    return {column: st.column_config.LineChartColumn(title or label(column))}


def flag_row(**flags: bool) -> None:
    """Small badges for the true things about a player: rookie, moved, not active."""
    shown = [name.replace("_", " ") for name, on in flags.items() if on]
    if shown:
        st.markdown(" ".join(f"`{s}`" for s in shown))


def pick_player(v: View, label: str = "Player", key: str = "player", index: int = 0) -> dict | None:
    """A searchable player picker over the projected population, returning his board row."""
    b = board(v)
    options = b.select(
        "player_id",
        (pl.col("player") + "  ·  " + pl.col("position") + " " + pl.col("team")).alias("label"),
    )
    labels = options["label"].to_list()
    chosen = st.selectbox(label, labels, index=min(index, len(labels) - 1) if labels else None,
                          key=key)
    if chosen is None:
        return None
    pid = options.filter(pl.col("label") == chosen)["player_id"][0]
    return b.filter(pl.col("player_id") == pid).row(0, named=True)


def position_filter(key: str = "pos") -> list[str]:
    return st.multiselect("Position", POSITIONS, default=list(POSITIONS), key=key)


def team_filter(v: View, key: str = "team") -> list[str]:
    teams = sorted(board(v)["team"].unique().to_list())
    return st.multiselect("Team", teams, default=[], key=key)


def apply_filters(
    df: pl.DataFrame, positions: list[str], teams: list[str], search: str = ""
) -> pl.DataFrame:
    out = df
    if positions:
        out = out.filter(pl.col("position").is_in(positions))
    if teams:
        out = out.filter(pl.col("team").is_in(teams))
    if search:
        out = out.filter(pl.col("player").str.to_lowercase().str.contains(search.lower().strip()))
    return out


def note(text: str) -> None:
    """The explanation for what is on screen, folded behind an ⓘ beside it.

    It used to print as a caption, and with a hundred and thirty of them the app read as an essay with
    tables in it: the reasoning for a number took more vertical space than the number, and the data a
    page exists for started below the fold. The words are worth keeping -- a projection nobody can
    interrogate is a projection nobody should trust -- so none of them are cut, they are one click away
    instead of in the way.

    A tertiary popover is borderless, so an unopened note is a small grey ⓘ rather than a button
    demanding attention. `section()` puts it on the same line as the heading, which is where most of
    these end up as pages are converted.
    """
    if _POPOVER_DEPTH:
        # Streamlit refuses a popover inside a popover, and a note inside one is already in a place the
        # reader chose to open. Degrade to the caption rather than raising on a page.
        st.caption(text)
        return
    try:
        with popover("ⓘ", help="why this is here"):
            st.markdown(text)
    except Exception:                     # noqa: BLE001
        st.caption(text)


# --------------------------------------------------------------------------- #
# the look
# --------------------------------------------------------------------------- #
# Ninety-three tables and almost nothing else. A player's ratings, the draft board, a pool audit and a
# tier list all arrived as the same grey grid, which is a design that says every number on the page is
# equally worth reading -- so nothing tells the eye where to go, and the numbers that decide a projection
# sit in column nineteen of a frame nobody scrolls. Worse for the job the app is for: the one thing that
# matters about an estimate is *where it landed between his own record and his job's average, and how
# much sample is behind it*, and that is a gauge with two marks on it, not four numeric columns to diff
# in your head.
#
# So this section is the vocabulary: a tile for a headline number, a meter for a number that sits inside
# a range, a chip for a fact that is true or absent, and a picker that turns a wide table into a narrow
# list with a detail panel beside it. Everything here is presentation only -- no frame is built and no
# number computed, which keeps the rule that a page reads the engine through one door and this file is
# the door.
_POPOVER_DEPTH = 0

STYLE = """
<style>
.nf-row{display:flex;flex-wrap:wrap;gap:.45rem;margin:.15rem 0 .35rem}
.nf-tile{flex:1 1 0;min-width:6.6rem;border:1px solid rgba(128,128,128,.26);border-radius:.6rem;
         padding:.45rem .6rem;background:rgba(128,128,128,.05)}
.nf-tile .k{display:block;font-size:.66rem;text-transform:uppercase;letter-spacing:.045em;opacity:.62;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nf-tile .v{display:block;font-size:1.28rem;font-weight:650;line-height:1.25;
            font-variant-numeric:tabular-nums}
.nf-tile .s{display:block;font-size:.7rem;opacity:.62;font-variant-numeric:tabular-nums}
.nf-tile.hl{border-color:var(--primary-color,#ff4b4b);background:rgba(255,75,75,.07)}
.nf-up{color:#1c9e5f}.nf-dn{color:#d1495b}.nf-mu{opacity:.55}
.nf-meter{margin:.3rem 0 .55rem}
.nf-meter .hd{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;
              font-size:.78rem;margin-bottom:.2rem}
.nf-meter .nm{opacity:.78}
.nf-meter .vl{font-weight:650;font-variant-numeric:tabular-nums}
.nf-track{position:relative;height:.5rem;border-radius:.3rem;background:rgba(128,128,128,.2)}
.nf-fill{height:100%;border-radius:.3rem;background:var(--primary-color,#ff4b4b);opacity:.85}
.nf-fill.thin{background:#6f8fbf}
.nf-mark{position:absolute;top:-.16rem;width:2px;height:.82rem;background:currentColor;opacity:.6;
         border-radius:1px}
.nf-ft{display:flex;flex-wrap:wrap;gap:.55rem;font-size:.7rem;opacity:.66;margin-top:.22rem;
       font-variant-numeric:tabular-nums}
.nf-chips{display:flex;flex-wrap:wrap;gap:.3rem;margin:.1rem 0 .35rem}
.nf-chip{display:inline-block;padding:.06rem .44rem;border-radius:999px;font-size:.7rem;
         border:1px solid rgba(128,128,128,.4);opacity:.9;white-space:nowrap}
.nf-chip.good{border-color:rgba(28,158,95,.7);color:#1c9e5f}
.nf-chip.warn{border-color:rgba(214,158,46,.8);color:#c98a10}
.nf-chip.bad{border-color:rgba(209,73,91,.75);color:#d1495b}
.nf-chip.edit{border-color:var(--primary-color,#ff4b4b);color:var(--primary-color,#ff4b4b)}
.nf-head{display:flex;align-items:baseline;gap:.5rem;flex-wrap:wrap;margin:.1rem 0 .1rem}
.nf-head .t{font-size:1.05rem;font-weight:650}
.nf-head .sub{font-size:.78rem;opacity:.62}
.nf-kv{display:flex;flex-wrap:wrap;gap:.15rem 1.1rem;font-size:.78rem;margin:.1rem 0 .3rem}
.nf-kv span b{font-variant-numeric:tabular-nums}
/* The page frame. Every page is now a header, a row of tiles and five to seven tabs, so the tab strip
   is the page's navigation and is worth reading as one: a rule under it, and the live tab in weight
   rather than in colour alone. Selectors are BaseWeb's rather than Streamlit's own hashed classes,
   which is the pair of them least likely to move under a version bump -- and if one ever misses, the
   result is the default look rather than a broken page. */
div[data-baseweb="tab-list"]{gap:.1rem;border-bottom:1px solid rgba(128,128,128,.22)}
button[data-baseweb="tab"]{font-size:.88rem}
button[data-baseweb="tab"][aria-selected="true"]{font-weight:650}
hr{margin:.7rem 0}
</style>
"""


def style() -> None:
    """Inject the stylesheet. Called once per page run from `controls`."""
    st.html(STYLE)


@contextmanager
def popover(label_text: str, **kw):
    """`st.popover`, counting depth so a nested one degrades instead of raising."""
    global _POPOVER_DEPTH
    kw.setdefault("type", "tertiary")
    kw.setdefault("width", "content")
    with st.popover(label_text, **kw) as box:
        _POPOVER_DEPTH += 1
        try:
            yield box
        finally:
            _POPOVER_DEPTH -= 1


def _esc(text: object) -> str:
    return html_escape("" if text is None else str(text))


def fmt_num(value: object, digits: int = 1, suffix: str = "", percent: bool = False) -> str:
    """A number as it should read, and an em dash where there is nothing. `None` is not `0`."""
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    # a NaN is a number nobody has -- a rate over no attempts, a per-game over no games -- and printing
    # it as "nan" beside a tile's own heading reads as a bug in the projection rather than an empty cell
    if x != x or x in (float("inf"), float("-inf")):
        return "—"
    if percent:
        return f"{x:.{max(digits - 2, 0)}%}"
    return f"{x:,.{digits}f}{suffix}"


def tile_html(name: str, value: object, sub: str = "", *, digits: int = 1, percent: bool = False,
              tone: str = "", highlight: bool = False, signed: bool = False) -> str:
    """One headline number as a bordered tile: what it is, what it is, and what it is against."""
    text = fmt_num(value, digits, percent=percent)
    if signed and value is not None and text != "—":
        text = f"+{text}" if float(value) > 0 else text
        tone = tone or ("nf-up" if float(value) > 0 else "nf-dn" if float(value) < 0 else "nf-mu")
    klass = "nf-tile hl" if highlight else "nf-tile"
    return (f'<div class="{klass}"><span class="k">{_esc(name)}</span>'
            f'<span class="v {tone}">{_esc(text)}</span>'
            + (f'<span class="s">{_esc(sub)}</span>' if sub else "<span class='s'>&nbsp;</span>")
            + "</div>")


def tiles(items: Sequence[dict]) -> None:
    """A row of tiles from `{name, value, sub, digits, percent, tone, highlight, signed}` dicts.

    One `st.html` for the row rather than a `st.columns` of `st.metric`, because six metrics in six
    columns each reserve a column's width whatever they hold, and a row of tiles wraps.
    """
    body = "".join(tile_html(i.get("name", ""), i.get("value"), i.get("sub", ""),
                             digits=int(i.get("digits", 1)), percent=bool(i.get("percent")),
                             tone=str(i.get("tone", "")), highlight=bool(i.get("highlight")),
                             signed=bool(i.get("signed")))
                   for i in items)
    st.html(f'<div class="nf-row">{body}</div>')


def meter_html(name: str, value: float | None, *, maximum: float | None = None,
               marks: Sequence[tuple[str, float | None]] = (), foot: Sequence[str] = (),
               digits: int = 3, percent: bool = False, thin: bool = False) -> str:
    """A number drawn inside the range it lives in, with the numbers it should be read against marked.

    This is the component the redesign turns on. `value` is what the projection ran on, `marks` are the
    comparisons -- his own record, his job's prior, the position's median -- drawn as ticks on the same
    track, so *where the estimate landed between the man and the role* is a picture instead of a
    subtraction. `maximum` is the axis: pass the position's best where there is one, so two players'
    meters are comparable, and leave it out for a share, where 1.0 is the honest ceiling.
    """
    top = float(maximum) if maximum else 1.0
    vals = [abs(float(v)) for _, v in marks if v is not None]
    if value is not None:
        vals.append(abs(float(value)))
    if not maximum and vals:
        top = max(1.0, max(vals)) if percent or max(vals) <= 1.0 else max(vals) * 1.15
    top = top or 1.0

    def pos(x: float) -> float:
        return min(max(float(x) / top * 100.0, 0.0), 100.0)

    fill = "" if value is None else (f'<div class="nf-fill{" thin" if thin else ""}" '
                                     f'style="width:{pos(value):.1f}%"></div>')
    ticks = "".join(
        f'<i class="nf-mark" style="left:{pos(v):.1f}%" title="{_esc(nm)} {fmt_num(v, digits, percent=percent)}"></i>'
        for nm, v in marks if v is not None
    )
    feet = "".join(f"<span>{_esc(f)}</span>" for f in foot if f)
    return (f'<div class="nf-meter"><div class="hd"><span class="nm">{_esc(name)}</span>'
            f'<span class="vl">{_esc(fmt_num(value, digits, percent=percent))}</span></div>'
            f'<div class="nf-track">{fill}{ticks}</div>'
            + (f'<div class="nf-ft">{feet}</div>' if feet else "") + "</div>")


def meters(rows: Sequence[str]) -> None:
    """Render a group of `meter_html` strings as one block."""
    if rows:
        st.html("".join(rows))


def chips(*items: str, **flags: bool) -> None:
    """Small badges for the things that are true. A false flag is absent, not drawn grey.

    `chips("rookie", status="warn:questionable")` — a value of `tone:text` sets the colour.
    """
    out = [f'<span class="nf-chip">{_esc(i)}</span>' for i in items if i]
    for name, on in flags.items():
        if not on:
            continue
        tone, _, text = (str(on) if not isinstance(on, bool) else "").partition(":")
        out.append(f'<span class="nf-chip {_esc(tone)}">{_esc(text or name.replace("_", " "))}</span>')
    if out:
        st.html(f'<div class="nf-chips">{"".join(out)}</div>')


def flag_row(**flags: bool) -> None:
    """Small badges for the true things about a player: rookie, moved, not active."""
    chips(**{k: v for k, v in flags.items() if v})


def kv(**pairs: object) -> None:
    """A run of `name value` pairs on one line, for the facts too small to deserve a tile."""
    body = "".join(f"<span>{_esc(k.replace('_', ' '))} <b>{_esc(v)}</b></span>"
                   for k, v in pairs.items() if v is not None)
    if body:
        st.html(f'<div class="nf-kv">{body}</div>')


def section(title: str, note_text: str | None = None, sub: str = "", level: int = 3) -> None:
    """A heading with its explanation folded in beside it, rather than printed under the table."""
    if note_text is None:
        st.html(f'<div class="nf-head"><span class="t">{_esc(title)}</span>'
                + (f'<span class="sub">{_esc(sub)}</span>' if sub else "") + "</div>")
        return
    head, tail = st.columns([12, 1], vertical_alignment="center")
    with head:
        st.html(f'<div class="nf-head"><span class="t">{_esc(title)}</span>'
                + (f'<span class="sub">{_esc(sub)}</span>' if sub else "") + "</div>")
    with tail:
        note(note_text)


# --------------------------------------------------------------------------- #
# a narrow list, and a panel beside it
# --------------------------------------------------------------------------- #
# The replacement for a forty-four-column table: six columns you can actually read down, and everything
# else about the row you picked in a panel next to it. `st.dataframe` returns its selection, so the list
# stays a table -- sortable, filterable, familiar -- and stops being the place the detail has to fit.
def pick_from(df: pl.DataFrame, key: str, columns: Sequence[str] = (), height: int = 560,
              config: dict | None = None, default: int = 0) -> dict | None:
    """A selectable list. Returns the picked row as a dict, falling back to the first row.

    Falling back rather than returning `None` is deliberate: a page whose panel is empty until you click
    reads as broken, and the first row of a board sorted by projected points is the row somebody was
    going to click anyway.
    """
    if df.is_empty():
        return None
    order = [c for c in columns if c in df.columns] or list(df.columns)
    event = st.dataframe(
        rounded(df.select(order), 2, {c: digits_for(c, 2) for c in order}),
        hide_index=True, height=height, width="stretch", key=key,
        on_select="rerun", selection_mode="single-row",
        column_config={**auto_config(df.select(order), 2), **(config or {})},
    )
    rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    at = rows[0] if rows else min(default, df.height - 1)
    return df.row(int(at), named=True)


def focus_table(df: pl.DataFrame, columns: Sequence[str], key: str, *, height: int = 420,
                config: dict | None = None, digits: int = 2, note_text: str | None = None,
                label_text: str = "every column") -> None:
    """A table capped at the columns worth reading, with the rest of the frame behind a popover.

    Nothing is hidden -- `Everything` is still one click away, and the export writes the whole frame --
    but the default is the eight columns that answer the question the page is on, because a table wider
    than the screen is a table whose right-hand half nobody has ever read.
    """
    keep = [c for c in columns if c in df.columns] or list(df.columns)
    table(df.select(keep), digits=digits, height=height, config=config)
    rest = [c for c in df.columns if c not in keep]
    if not rest:
        if note_text:
            note(note_text)
        return
    bar_l, bar_r = st.columns([1, 6], vertical_alignment="center")
    with bar_l:
        with popover(f"⋯ {label_text} ({len(rest)} more)"):
            table(df, digits=digits, height=min(height, 520), config=config)
    with bar_r:
        if note_text:
            note(note_text)


# --------------------------------------------------------------------------- #
# the stat line
# --------------------------------------------------------------------------- #
# A projection is a football stat line first and a fantasy total second: 74 catches for 1,014 yards is
# the claim, and 214 points is only what a scoring system makes of it. Change the scoring and the points
# move while the line stands. So the columns are named once, here, and every page shows the same line
# for a position rather than inventing its own subset of the board's forty-four columns.
STAT_GROUPS: dict[str, tuple[str, ...]] = {
    "Passing": ("dropbacks", "attempts", "completions", "passing_yards", "passing_tds",
                "interceptions", "sacks", "passing_air_yards"),
    "Rushing": ("carries", "rushing_yards", "rushing_tds", "designed_rushes", "scrambles",
                "designed_rush_yards", "scramble_yards"),
    "Receiving": ("targets", "receptions", "receiving_yards", "receiving_tds",
                  "receiving_air_yards"),
    "Playing time": ("games", "offense_snaps", "routes", "rush_plays", "fumbles_lost"),
}

# The line for each position: what a box score would print for him, in box-score order.
STAT_LINE: dict[str, tuple[str, ...]] = {
    "QB": ("attempts", "completions", "passing_yards", "passing_tds", "interceptions", "sacks",
           "carries", "rushing_yards", "rushing_tds", "fumbles_lost"),
    "RB": ("carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards",
           "receiving_tds", "fumbles_lost"),
    "WR": ("targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
           "carries", "rushing_yards", "rushing_tds", "fumbles_lost"),
    "TE": ("targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
           "fumbles_lost"),
}

# Which box-score groups a position gets, most important first.
STAT_BLOCKS: dict[str, tuple[str, ...]] = {
    "QB": ("Passing", "Rushing", "Playing time"),
    "RB": ("Rushing", "Receiving", "Playing time"),
    "WR": ("Receiving", "Rushing", "Playing time"),
    "TE": ("Receiving", "Playing time"),
}

# Yards and snaps are whole numbers to read; a projected touchdown is 8.4 and rounding it to 8 would be
# a different claim, so anything countable in single digits keeps a decimal.
WHOLE_STATS = ("passing_yards", "rushing_yards", "receiving_yards", "passing_air_yards",
               "receiving_air_yards", "designed_rush_yards", "scramble_yards", "offense_snaps",
               "routes", "rush_plays", "dropbacks", "attempts", "completions")

ALL_STATS = tuple(dict.fromkeys(c for group in STAT_GROUPS.values() for c in group))


def stat_digits(field: str) -> int:
    return 0 if field in WHOLE_STATS else 1


def stat_columns(positions: str | list[str] | tuple[str, ...] | None = None) -> list[str]:
    """The stat line for these positions: one position's own line, or the union in box-score order."""
    want = [positions] if isinstance(positions, str) else list(positions or STAT_LINE)
    want = [p for p in want if p in STAT_LINE] or list(STAT_LINE)
    if len(want) == 1:
        return list(STAT_LINE[want[0]])
    picked = {c for p in want for c in STAT_LINE[p]}
    return [c for c in ALL_STATS if c in picked]


def stat_config(columns: list[str] | tuple[str, ...] | None = None) -> dict:
    """Column config for stat columns: yards whole, touchdowns to a decimal."""
    return numeric({c: f"%.{stat_digits(c)}f" for c in (columns or ALL_STATS)})


def stat_line(row: dict, position: str | None = None) -> str:
    """The projection as a football line — what a scout would say, before any scoring system."""
    pos = position or row.get("position") or ""

    def n(field: str, digits: int = 0) -> str:
        v = row.get(field)
        return "—" if v is None else f"{float(v):,.{digits}f}"

    def has(field: str) -> bool:
        return float(row.get(field) or 0.0) >= 1.0

    bits: list[str] = []
    if pos == "QB":
        bits += [f"{n('completions')}/{n('attempts')}", f"{n('passing_yards')} yds",
                 f"{n('passing_tds', 1)} TD", f"{n('interceptions', 1)} INT"]
        if has("carries"):
            bits += [f"{n('carries')} car", f"{n('rushing_yards')} yds", f"{n('rushing_tds', 1)} TD"]
    else:
        if has("carries"):
            bits += [f"{n('carries')} car", f"{n('rushing_yards')} rush yds",
                     f"{n('rushing_tds', 1)} TD"]
        if has("targets"):
            bits += [f"{n('receptions')}/{n('targets')} rec", f"{n('receiving_yards')} rec yds",
                     f"{n('receiving_tds', 1)} TD"]
    return "  ·  ".join(bits) if bits else "no projected volume"


def with_stat_line(df: pl.DataFrame) -> pl.DataFrame:
    """Add `line`: the projected football line as one string, correct for whatever position each man is.

    A board of four positions cannot show a box score as columns -- a passing column on a receiver's row
    is a blank cell pretending to be information, and there is no column set that is honest for a
    quarterback and a slot receiver at once. One string per row is, and it puts the football in front of
    the scoring where a review starts. The sortable numeric version comes back the moment the reader
    filters to a single position, which is also the moment those numbers can be compared.
    """
    if df.is_empty():
        return df.with_columns(pl.lit("", pl.String).alias("line"))
    return df.with_columns(
        pl.Series("line", [stat_line(r) for r in df.iter_rows(named=True)], dtype=pl.String)
    )


def stat_blocks(row: dict, position: str | None = None, per_game: dict | None = None) -> None:
    """The full stat line as metric cards, grouped the way a box score is.

    The point of the cards rather than a column of numbers: the season and the per-game figure sit
    together, and a lineup decision is made on the second one.
    """
    pos = position or row.get("position") or ""
    line = STAT_LINE.get(pos, ())
    for group in STAT_BLOCKS.get(pos, STAT_BLOCKS["WR"]):
        fields = [c for c in STAT_GROUPS[group]
                  if row.get(c) is not None
                  and (abs(float(row[c])) > 0.05 or c in line)]
        if not fields:
            continue
        st.caption(group)
        tiles([{"name": label(f), "value": row[f], "digits": stat_digits(f),
                "sub": ("" if per_game is None or per_game.get(f) is None else
                        f"{float(per_game[f]):,.{max(stat_digits(f), 1)}f} / g")}
               for f in fields])


# --------------------------------------------------------------------------- #
# the depth chart
# --------------------------------------------------------------------------- #
# The share that decides whether a man is the starter at his position. One per position, because "28% of
# the targets" is the sentence a receiver's projection turns on and "88% of the dropbacks" is the
# quarterback's; showing all sixteen shares at once answers neither question.
#
# Taken from the opportunity frame rather than from `shares`, and this is the whole reason `depth_frame`
# exists: `target_share` is the estimator's belief about a man on the field, while `share_targets` is
# what actually divided the team's pool -- availability-weighted, and after normalisation. A backup
# quarterback with a 0.79 `dropback_share` has 0.79 of the dropbacks *in the games he starts*, which as a
# depth chart reads like a co-starter. The pool share says 12%, which is what the projection did.
HEADLINE_SHARE = {"QB": "share_dropbacks", "RB": "share_carries", "WR": "share_targets",
                  "TE": "share_targets"}
SHARE_LABEL = {"share_dropbacks": "dropbacks", "share_carries": "carries",
               "share_targets": "targets"}

DEPTH_IDENTITY = ("player_id", "player", "team", "position", "depth_tier", "depth_slot",
                  "slot_bucket", "status", "is_rookie", "draft_pick", "years_exp", "age", "charted",
                  "team_disagreement", "alignment", "expected_games", "presence", "status_factor",
                  "snap_share", "route_participation", "rush_participation", "dropback_share")


def depth_frame(v: View, team: str | None = None) -> pl.DataFrame:
    """The depth chart with the projection beside each name.

    An assembly of frames the run already produced -- roster order and availability from
    `participation`, the headline pool share from `shares`, the stat line from `board` -- and nothing
    computed here, so the chart cannot disagree with the board about who the starter is.
    """
    men = participation(v).select([c for c in DEPTH_IDENTITY if c in participation(v).columns])
    if team:
        men = men.filter(pl.col("team") == team)

    opp = opportunity_frame(v)
    pools = [c for c in dict.fromkeys(HEADLINE_SHARE.values()) if c in opp.columns]
    if pools:
        if team:
            opp = opp.filter(pl.col("team") == team)
        men = men.join(
            opp.group_by("player_id").agg(pl.col(c).mean() for c in pools),
            on="player_id", how="left",
        )

    b = board(v)
    keep = [c for c in dict.fromkeys(
        ("fantasy_points", "points_per_game", "position_rank", "overall_rank", "tier", "startable",
         "vs_starter", "drop_next", *ALL_STATS))
        if c in b.columns and c not in men.columns]
    out = men.join(b.select("player_id", *keep), on="player_id", how="left")

    # one `pool_share` column: whichever pool decides this player's position
    picked = pl.lit(None, dtype=pl.Float64)
    for pos, col in HEADLINE_SHARE.items():
        if col in out.columns:
            picked = pl.when(pl.col("position") == pos).then(pl.col(col)).otherwise(picked)
    out = out.with_columns(picked.alias("pool_share"),
                           pl.col("position")
                           .replace_strict({p: i for i, p in enumerate(POSITIONS)}, default=99,
                                           return_dtype=pl.Int32).alias("_pos"))
    by = ["team", "_pos", "depth_slot"] if "team" in out.columns else ["_pos", "depth_slot"]
    return out.sort(by).drop("_pos")


def depth_board(chart: pl.DataFrame, limit: int = 7) -> None:
    """The depth chart as cards in slot order, one column per position.

    A depth chart is a shape, not a table: who is ahead of whom, how much of the pool each of them has,
    and where the drop is. Cards in slot order say that at a glance; a sortable grid of the same numbers
    invites sorting it into an order the depth chart is not in.
    """
    if chart.is_empty():
        st.info("No projected players on this roster.")
        return
    present = [p for p in POSITIONS if p in set(chart["position"].unique().to_list())]
    cols = st.columns(len(present))
    for col, pos in zip(cols, present, strict=True):
        room = chart.filter(pl.col("position") == pos)
        with col:
            st.markdown(f"##### {pos}  ·  {room.height}")
            for r in room.head(limit).rows(named=True):
                with st.container(border=True):
                    head = st.columns([3, 1])
                    mark = " ✏️" if live().touching("player", r["player_id"]) else ""
                    head[0].markdown(f"**{r['player']}**{mark}")
                    head[1].markdown(f"**{float(r.get('fantasy_points') or 0):.0f}**")
                    tags = [
                        f"{pos}{int(r['depth_slot'])}" if r.get("depth_slot") is not None else "",
                        f"slot {r['slot_bucket']}" if r.get("slot_bucket") is not None else "",
                        f"ranked {pos}{int(r['position_rank'])}"
                        if r.get("position_rank") is not None else "",
                        "" if r.get("status") in (None, "ACT") else str(r.get("status")),
                        "rookie" if r.get("is_rookie") else "",
                        "not charted" if r.get("charted") is False else "",
                    ]
                    st.caption(" · ".join(str(t) for t in tags if t))
                    share = r.get("pool_share")
                    if share is not None:
                        pool = SHARE_LABEL.get(HEADLINE_SHARE[pos], "pool")
                        st.progress(
                            min(max(float(share), 0.0), 1.0),
                            text=f"{float(share):.0%} of the {pool}  ·  "
                                 f"{float(r.get('games') or 0):.1f} games",
                        )
                    st.caption(stat_line(r, pos))


# The chart as something to change rather than to read. `depth_slot` is the only editable column and it
# is deliberately the leftmost number: everything to the right of it is a consequence of it, which is
# also the reason it is worth editing here rather than reaching for a share.
DEPTH_EDIT_COLUMNS = ("position", "depth_slot", "player", "status", "pool_share", "expected_games",
                      "games", "fantasy_points", "position_rank", "charted", "team_disagreement",
                      "is_rookie", "draft_pick", "years_exp")


def depth_editor(v: View, chart: pl.DataFrame, team: str, key: str = "depth") -> None:
    """Reorder a depth chart by typing slots into it, and re-price everybody the move touched.

    The one override that is an ordering rather than a quantity, so it behaves differently and the page
    says so: type 1 beside the third receiver and the room renumbers around him, then every prior keyed
    on a slot is read again -- his expected games, his survival rate, his share of the pool, and the same
    for the men he moved past. That is the difference between promoting a player and simply handing him a
    starter's target share: the second leaves him priced as a backup who happens to get the ball.
    """
    if chart.is_empty():
        st.info("No projected players on this roster.")
        return
    keep = ["player_id", *[c for c in DEPTH_EDIT_COLUMNS if c in chart.columns]]
    df = with_edits(chart.select(keep), "player", "player_id")
    moved = [o for o in live().items
             if o.level == "player" and o.field == overrides.DEPTH_FIELD
             and o.key in set(df["player_id"].to_list())]

    head = st.columns([5, 1])
    head[0].markdown(f"**Type a slot to move a man.** {len(moved)} move"
                     f"{'' if len(moved) == 1 else 's'} on this chart.")
    if head[1].button("↺ chart", key=_keyed(f"{key}:{team}:resetchart"), disabled=not moved,
                      help="drop every depth-slot move on this team and follow the published chart"):
        for o in moved:
            drop_edit("player", o.key, overrides.DEPTH_FIELD, o.week)
        st.rerun()

    grid(
        df, level="player", key_col="player_id", fields=(overrides.DEPTH_FIELD,),
        key=f"{key}:{team}", height=min(120 + 35 * df.height, 620), hide=("player_id",),
        config={
            overrides.DEPTH_FIELD: st.column_config.NumberColumn(
                "depth slot", min_value=1, max_value=30, step=1, format="%d",
                help="1 is the starter. Everybody else in the room renumbers around what you type.",
            ),
            **percent("pool_share"),
            **fixed(1, "expected_games", "games", "fantasy_points"),
        },
    )
    note(
        "The room is rebuilt around the move rather than swapped: the men you asked for go to the slots "
        "you asked for and everybody else falls into what is left, keeping the published order, so a "
        "chart always reads 1, 2, 3 with nobody sharing a number — send the starter to 2 and the man "
        "behind him takes 1. Two men sent to the same slot are "
        "ordered by the more recent edit. `charted` false is a player the published chart never listed "
        "and the estimator placed itself — the slots most worth moving are usually in those rows."
    )


# --------------------------------------------------------------------------- #
# off the roster entirely
# --------------------------------------------------------------------------- #
RELEASED_COLUMNS = ("player", "position", "team", "depth_slot", "games", "fantasy_points",
                    "position_rank", "status")


def released_frame(v: View, team: str | None = None) -> pl.DataFrame:
    """The men taken off a roster, named from the **baseline** run, and what they were projected for.

    It has to be the baseline: the edited run is the one they were removed from, so it has no row to
    read a name off. What that also gives is the honest cost of the cut -- the points the projection had
    on him before it was told he is not there -- which is the number a reader wants beside the name.
    """
    gone = released_ids()
    b = baseline(v).board
    if not gone or b.is_empty():
        return b.head(0).select(["player_id", *[c for c in RELEASED_COLUMNS if c in b.columns]])
    mine = b.filter(pl.col("player_id").is_in(gone))
    if team and "team" in mine.columns:
        mine = mine.filter(pl.col("team") == team)
    keep = ["player_id", *[c for c in RELEASED_COLUMNS if c in mine.columns]]
    by = [c for c in ("position", "depth_slot") if c in mine.columns]
    return mine.select(keep).sort(by) if by else mine.select(keep)


def release_label(row: dict) -> str:
    """`Name · WR3` — enough to tell two men of the same name at the same position apart by their slot."""
    pos = row.get("position") or ""
    slot = row.get("depth_slot")
    tail = f" · {pos}{int(slot)}" if pos and slot is not None else (f" · {pos}" if pos else "")
    return f"{row.get('player') or row['player_id']}{tail}"


def release_editor(v: View, chart: pl.DataFrame, team: str, key: str = "release") -> None:
    """Take a man off this roster entirely, and put him back.

    The edit every other control on this page cannot express. Zeroing a man's availability leaves him
    holding his slot in the room -- the chart still has a WR2 who plays no games, and the pool is still
    divided as though somebody is standing there -- which is right for an injury and wrong for a player
    who is on another team. A release removes the row before anything is estimated from it, so the room
    renumbers around the hole, the men behind him are re-priced as what they moved up to, his share of
    every pool goes to whoever is left, and no board, week, export or simulation has a row to count him
    in. It is the answer to "he isn't on this team" in the window where our sources still say he is.
    """
    if chart.is_empty():
        st.info("No projected players on this roster.")
        return
    gone = released_frame(v, team)
    gone_ids = gone["player_id"].to_list()
    on = chart.select([c for c in ("player_id", *RELEASED_COLUMNS) if c in chart.columns])
    names = {r["player_id"]: release_label(r) for r in on.rows(named=True)}
    names.update({r["player_id"]: release_label(r) for r in gone.rows(named=True)})
    options = [*on["player_id"].to_list(), *gone_ids]

    head = st.columns([5, 1])
    lost = float(gone["fantasy_points"].sum() or 0) if "fantasy_points" in gone.columns else 0.0
    head[0].markdown(
        f"**Nobody is off this roster.** Choose a man to remove him from the projection entirely."
        if not gone_ids else
        f"**{len(gone_ids)} off this roster**, worth {lost:.0f} projected points before the cut — "
        "their share of every pool now belongs to the men who are left."
    )
    if head[1].button("↺ roster", key=_keyed(f"{key}:{team}:resetroster"), disabled=not gone_ids,
                      help="put everybody back on this team's chart"):
        reinstate(*gone_ids)
        st.rerun()

    picked = st.multiselect(
        "Off the roster", options, default=gone_ids, key=_keyed(f"{key}:{team}"),
        format_func=lambda pid: names.get(pid, pid),
        placeholder="nobody — type a name to take him off this team",
        help="He leaves the chart, the room closes up behind him, and nothing downstream counts him.",
    )
    add = [pid for pid in picked if pid not in set(gone_ids)]
    back = [pid for pid in gone_ids if pid not in set(picked)]
    if add:
        release(*add)
    if back:
        reinstate(*back)
    if add or back:
        st.rerun()

    if not gone.is_empty():
        focus_table(
            gone.drop("player_id"),
            [c for c in ("player", "position", "depth_slot", "games", "fantasy_points") if c in gone.columns],
            key=f"{key}:{team}:gone",
            config={**fixed(1, "games", "fantasy_points")},
            height=min(80 + 35 * gone.height, 300),
            label_text="what the projection had on them before the cut",
        )
    note(
        "This is a season-long fact about who is on the team, not a game he misses: for one week, set "
        "`chance he plays` to 0 in **Week → 🎛️ Adjust this game** instead. A release is applied before "
        "the chart is settled, so it is the one edit that removes a row rather than changing a number in "
        "one — his slot is gone rather than empty, the men behind him move up and are re-priced off the "
        "slot they moved to, and the team's targets, carries and touchdowns are divided among the "
        "players who remain rather than partly wasted on somebody who is not there. Putting him back "
        "drops the edit, so the chart follows our sources again. The last man in a room cannot be "
        "released — a pool with nobody to divide it is not a projection — and **Edits** lists every "
        "release beside the rest of the scenario."
    )


# --------------------------------------------------------------------------- #
# the ratings behind a player
# --------------------------------------------------------------------------- #
RATING_COLUMNS = ("metric", "units", "obs", "n", "prior_slot", "curve", "prior", "used",
                  "own_weight", "k", "seasons_used", "source")


def one_row_per_metric(rated: pl.DataFrame) -> pl.DataFrame:
    """Collapse a metric the stacked frames both carry, keeping the pool share's heading.

    `dropback_share` is a participation metric -- is he the starter -- and a pool share -- whose
    dropbacks are these -- so it arrives from two frames with identical numbers and two different
    `kind`s. Left alone it draws two meters for one estimate, each with a ✎ writing the same override,
    which is how the board raised a duplicate-key error the first time a quarterback was selected.
    Keeping the share is not a decision about the number, only about the heading to look under.
    """
    if rated.is_empty() or "metric" not in rated.columns:
        return rated
    return (rated.with_columns((pl.col("kind") != "share").alias("_second")).sort("_second")
            .unique(subset="metric", keep="first", maintain_order=True).drop("_second"))


def ratings(v: View, player_id: str) -> pl.DataFrame:
    """Every estimate behind one player, and what the projection actually used.

    Three frames stacked -- availability, shares, rates -- with two more columns than the estimator
    produces: `applied`, what the projection used after the override layer, and `override`, what was
    typed. Together they answer the only two questions a rating raises. What does the model think, and
    is that still what is being used.
    """
    got = []
    for kind, frame in (("availability", participation_detail(v)), ("share", share_detail(v)),
                        ("rate", rate_detail(v))):
        mine = frame.filter(pl.col("player_id") == player_id)
        if mine.is_empty():
            continue
        keep = [c for c in RATING_COLUMNS if c in mine.columns]
        got.append(mine.select(keep).with_columns(pl.lit(kind).alias("kind")))
    got.extend(_games_rating(v, player_id))
    if not got:
        return pl.DataFrame()
    out = pl.concat(got, how="diagonal_relaxed")

    out = one_row_per_metric(out)

    # what the projection ran on, which is the estimate only while nobody has edited it
    used: dict[str, float] = {}
    for frame in (participation(v), shares_wide(v), rates_wide(v)):
        mine = frame.filter(pl.col("player_id") == player_id)
        if mine.is_empty():
            continue
        for name, val in mine.row(0, named=True).items():
            if isinstance(val, bool) or not isinstance(val, (int, float)) or name in used:
                continue
            used[name] = float(val)
    applied = pl.DataFrame({"metric": list(used), "applied": list(used.values())},
                           schema={"metric": pl.String, "applied": pl.Float64})

    edits = {o.field: o for o in live().touching("player", player_id)}
    marks = pl.DataFrame(
        {"metric": list(edits),
         "override": [("= " if o.mode == "set" else "x ") + f"{o.value:g}" for o in edits.values()]},
        schema={"metric": pl.String, "override": pl.String},
    )
    out = out.join(applied, on="metric", how="left").join(marks, on="metric", how="left")
    return out.with_columns(
        pl.col("override").is_not_null().alias("edited"),
        pl.col("metric").is_in(list(overrides.PLAYER_FIELDS)).alias("editable"),
    ).with_columns(pl.col("override").fill_null("")).sort(["kind", "metric"])


_GAMES_SCHEMA = {"kind": pl.String, "metric": pl.String, "units": pl.String, "obs": pl.Float64,
                 "n": pl.Float64, "prior_slot": pl.Float64, "prior": pl.Float64, "used": pl.Float64,
                 "own_weight": pl.Float64, "source": pl.String}


def _games_rating(v: View, player_id: str) -> list[pl.DataFrame]:
    """`expected_games` as one more rating row, built from the same four numbers as the rest.

    It is not one of the estimator's long-form metrics -- availability is fitted in `roster.py` rather
    than through `estimate.estimate` -- but it is the estimate most worth arguing with, so leaving it
    out of the ratings table would leave out the argument. `used` is read from the *baseline* run so it
    stays the engine's own answer however many times the knob beside it is turned.
    """
    mine = baseline(v).part.filter(pl.col("player_id") == player_id)
    if mine.is_empty():
        return []
    r = mine.row(0, named=True)

    def num(name: str) -> float | None:
        val = r.get(name)
        return None if val is None else float(val)

    return [pl.DataFrame([{
        "kind": "availability", "metric": "expected_games", "units": "games",
        "obs": num("obs_games"), "n": num("n_seasons"), "prior_slot": num("prior_games"),
        "prior": num("prior_games"), "used": num("expected_games"),
        "own_weight": num("games_own_weight"),
        "source": "own games blended with the slot's, x presence x roster status",
    }], schema=_GAMES_SCHEMA)]


def rating_config() -> dict:
    """One config for the ratings table, so `own_weight` is a bar everywhere it appears."""
    return {
        **fixed(4, "obs", "prior_slot", "curve", "prior", "used", "applied"),
        **fixed(1, "n", "k"),
        **bar("own_weight"),
    }


def bar(*columns: str, maximum: float = 1.0) -> dict:
    """A 0-1 column drawn as a bar rather than printed as a number: a weight is read by length."""
    return {c: st.column_config.ProgressColumn(label(c), format="percent", min_value=0.0,
                                               max_value=maximum) for c in columns}


# --------------------------------------------------------------------------- #
# the evidence an override is made on
# --------------------------------------------------------------------------- #
# An override is a claim that the model is wrong about one number. That claim is only worth applying if
# the person making it can see three things the model already knows: what the player himself has done
# and over how much opportunity, what his job is worth to everybody else who has held it, and where the
# number sits in the league. None of that is new arithmetic -- every piece is a frame the run already
# produced or a record already in the lake -- but until it is beside the input box the honest override
# and the guess look identical afterwards, so this section exists to put it there.
GAMES_METRIC = "expected_games"          # availability, which is not one of the estimator's metrics
QUANTILES = ((0.1, "p10"), (0.25, "p25"), (0.5, "median"), (0.75, "p75"), (0.9, "p90"))


@st.cache_data(show_spinner="reading what he actually did, season by season", max_entries=32)
def metric_seasons(metrics: tuple[str, ...], player_ids: tuple[str, ...],
                   seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """The record: each metric as it was measured, per season, unweighted and unshrunk.

    Keyed on the players asked for rather than the whole league, because the panel wants one man and a
    room table wants a dozen, and reading the metric for nine hundred is a second nobody asked for.
    """
    from src.model import estimate

    return estimate.season_history(metrics, player_ids, seasons)


@st.cache_data(show_spinner="reading attendance records")
def attendance(seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """Games played per season against his team's games. The record behind `expected_games`."""
    return roster.attendance_history(seasons)


PER_GAME_ALREADY = ("seconds_per_play", "yards_per_attempt", "yards_per_carry", "success_rate",
                    "implied_points")


@st.cache_data(show_spinner="reading what these teams have actually run")
def team_seasons(seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """Team totals per season, from the same tables the team model is fitted on."""
    return history.team_seasons(seasons)


def team_history(v: View, team: str, fields: tuple[str, ...] | list[str],
                 seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """What this offence has actually run per game, by season, with the projection and the league under it.

    The evidence for a team edit, which is the override with the least intuition behind it: nobody knows
    from memory whether 34 dropbacks a game is a lot, and the number is worth arguing with only against
    what this team has run and what the rest of the league is projected to run. Counts are per game so
    the rows compare; the rates are already per play and are left alone.
    """
    keep = [f for f in fields if f in team_seasons(seasons).columns or f in environment(v).columns]
    if not keep:
        return pl.DataFrame()
    rows = []
    hist = team_seasons(seasons).filter(pl.col("team") == team).sort("season")
    for r in hist.rows(named=True):
        games = max(float(r.get("games") or 0.0), 1e-9)
        rows.append({"row": str(r["season"]), **{
            f: (None if r.get(f) is None else
                float(r[f]) if f in PER_GAME_ALREADY else float(r[f]) / games)
            for f in keep}})
    env = environment(v)
    mine = env.filter(pl.col("team") == team)
    for name, frame in ((f"{v.season} projected", mine), (f"{v.season} league mean", env)):
        if frame.is_empty():
            continue
        rows.append({"row": name, **{f: (float(frame[f].mean()) if f in frame.columns else None)
                                     for f in keep}})
    return pl.DataFrame(rows, schema={"row": pl.String, **{f: pl.Float64 for f in keep}})


@st.cache_data(show_spinner="reading the league on this metric", max_entries=48)
def metric_values(v: View, metric: str) -> pl.DataFrame:
    """One row per player for a single metric: the estimate, its evidence, and what the projection ran on.

    Assembled from the frames the run already produced, so a room table and a league percentile are the
    same numbers the ratings table shows for one man. `estimate` is the estimator's own answer and
    `applied` is what was used; they differ exactly where somebody has typed over it.
    """
    if metric == GAMES_METRIC:
        base = baseline(v).part
        ident = [c for c in ("player_id", "player", "position", "team", "depth_slot", "slot_bucket")
                 if c in base.columns]
        out = base.select(
            *ident,
            pl.col("status"),
            pl.col("obs_games").alias("obs"),
            pl.col("n_seasons").alias("n"),
            pl.col("prior_games").alias("prior"),
            pl.col("games_if_available").alias("blend"),
            pl.col("presence"),
            pl.col("status_factor"),
            pl.col("expected_games").alias("estimate"),
            pl.col("games_own_weight").alias("own_weight"),
            pl.lit("seasons").alias("units"),
            pl.lit("own record blended with his slot's, x presence x roster status").alias("source"),
        )
        return out.join(
            participation(v).select("player_id", pl.col("expected_games").alias("applied")),
            on="player_id", how="left",
        )

    for detail, wide in ((share_detail(v), shares_wide(v)), (rate_detail(v), rates_wide(v)),
                         (participation_detail(v), participation(v))):
        if detail.is_empty() or "metric" not in detail.columns:
            continue
        mine = detail.filter(pl.col("metric") == metric)
        if mine.is_empty():
            continue
        keep = [c for c in ("player_id", "player", "position", "team", "depth_slot", "slot_bucket",
                            "units", "k", "obs", "n", "prior_slot", "curve", "prior", "used",
                            "own_weight", "source") if c in mine.columns]
        out = mine.select(keep).rename({"used": "estimate"})
        if metric in wide.columns:
            out = out.join(wide.select("player_id", pl.col(metric).alias("applied")),
                           on="player_id", how="left")
        return out
    return pl.DataFrame()


def metric_spread(v: View, metric: str) -> pl.DataFrame:
    """What the metric is worth across the league, by position and by depth slot within it.

    Nobody knows on its own whether 0.24 is a lot. Everybody knows once the position's quartiles are
    beside it, and a slot row says whether the number is unusual for a man in that job specifically --
    which is the comparison a depth-chart argument is actually about.
    """
    d = metric_values(v, metric)
    if d.is_empty() or "estimate" not in d.columns:
        return pl.DataFrame()
    d = d.filter(pl.col("estimate").is_not_null())
    agg = [pl.len().alias("players"),
           *[pl.col("estimate").quantile(q).alias(name) for q, name in QUANTILES],
           pl.col("estimate").max().alias("best")]
    by_slot = (d.filter(pl.col("depth_slot") <= 4).group_by("position", "depth_slot").agg(*agg)
               .rename({"depth_slot": "slot"}).with_columns(pl.col("slot").cast(pl.Int32)))
    by_pos = (d.group_by("position").agg(*agg)
              .with_columns(pl.lit(None, pl.Int32).alias("slot")).select(by_slot.columns))
    return pl.concat([by_pos, by_slot], how="vertical_relaxed").sort("position", "slot",
                                                                    nulls_last=False)


def percentile_of(values: pl.Series, x: float | None) -> float | None:
    """Where one number sits in a column of them, as a share below it. `None` when there is nothing."""
    if x is None:
        return None
    v = values.drop_nulls()
    return None if v.len() == 0 else float((v <= float(x)).mean())


def collapse_seasons(long: pl.DataFrame, keys: tuple[str, ...] = ("player_id",)) -> pl.DataFrame:
    """One row per key per season, for a record that has a player traded mid-year in it twice.

    A season split across two teams is two rows of the same record, and the two do not average: the
    honest combination of a 22% target share over 40 targets and a 9% over 90 is the ratio of the
    totals, not the mean of the ratios. Where there is no denominator to add -- games played -- the
    counts themselves sum, which is the same rule applied to the same question.
    """
    if long.is_empty():
        return long
    by = [*keys, "season"]
    dupes = [c for c in by if c in long.columns]
    if len(dupes) != len(by) or long.select(by).is_duplicated().sum() == 0:
        return long
    has_ratio = "num" in long.columns and "n" in long.columns
    adds = {"num", "n", "games", "team_games"} | (set() if has_ratio else {"value"})
    agg = [(pl.col(c).sum() if c in adds else pl.col(c).last())
           for c in long.columns if c not in by]
    out = long.sort(by).group_by(by).agg(*agg)
    if has_ratio:
        out = out.with_columns(
            pl.when(pl.col("n") > 0).then(pl.col("num") / pl.col("n")).alias("value")
        )
    if "games" in out.columns and "team_games" in out.columns and "rate" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("team_games") > 0)
            .then(pl.col("games") / pl.col("team_games")).alias("rate")
        )
    return out.select(long.columns).sort(by)


def season_wide(long: pl.DataFrame, value: str = "value",
                seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """A long per-season frame as one row per player: a column per season, and a list to draw.

    The list is the point. Five numbers in a cell is a trend, and a trend is what separates a 26% target
    share a player has held for four seasons from one he reached once -- two records that shrink to the
    same estimate and mean entirely different things about next year.

    Every season asked for gets a column whether or not anybody has a number in it, so the trends share
    an x-axis: a line drawn over the four seasons a player was in the league, beside one drawn over five,
    would put his rookie year under somebody's second season and read as a decline. The list fills a
    missing season with zero because a chart cannot draw a gap; the column beside it keeps the null,
    which is the honest form of "he was not there".
    """
    if long.is_empty() or value not in long.columns:
        return pl.DataFrame(schema={"player_id": pl.String})
    wide = collapse_seasons(long).pivot(on="season", index="player_id", values=value)
    if not seasons:
        return wide
    wide = wide.with_columns([pl.lit(None, pl.Float64).alias(str(s)) for s in seasons
                              if str(s) not in wide.columns])
    cols = [str(s) for s in seasons]
    return wide.select("player_id", *cols).with_columns(
        pl.concat_list([pl.col(c).fill_null(0.0) for c in cols]).alias("record")
    )


def season_config(seasons: tuple[int, ...] = HISTORY_VIEW, digits: int = 3) -> dict:
    """Column config for a season block: the years as numbers, the trend as a line."""
    return {**fixed(digits, *[str(s) for s in seasons]),
            "record": st.column_config.LineChartColumn("trend")}


def room_evidence(v: View, metric: str, player_ids: tuple[str, ...] = ()) -> pl.DataFrame:
    """One metric across a room: the estimate, the evidence under it, and the last five seasons.

    The table that makes an override *logical* rather than merely possible. A share is zero-sum inside a
    team, so the man being edited and the men it would come from belong on one screen with each of their
    own records beside their estimate -- otherwise the reader is arguing about one number in isolation
    and normalisation quietly settles the argument afterwards.
    """
    vals = metric_values(v, metric)
    if vals.is_empty():
        return vals
    if player_ids:
        vals = vals.filter(pl.col("player_id").is_in(list(player_ids)))
    if vals.is_empty():
        return vals
    who = tuple(vals["player_id"].to_list())
    if metric == GAMES_METRIC:
        long = attendance().filter(pl.col("player_id").is_in(list(who))).rename({"games": "value"})
    else:
        long = metric_seasons((metric,), who)
    out = vals.join(season_wide(long), on="player_id", how="left")
    front = [c for c in ("player", "position", "depth_slot", "obs", "n", "own_weight", "prior",
                         "estimate", "applied") if c in out.columns]
    rest = [c for c in out.columns if c not in front and c != "player_id"]
    return with_edits(out, "player", "player_id").select("player_id", *front, *rest).sort(
        "estimate", descending=True, nulls_last=True
    )


EVIDENCE_COLUMNS = ("engine", "obs", "n", "own_weight", "prior")


def with_evidence(v: View, df: pl.DataFrame, metric: str, key_col: str = "player_id",
                  seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """Put the evidence for one metric beside a frame that is about to be edited.

    The editable column keeps its own name and its own value -- the grid is still editing what the
    projection ran on -- and the evidence arrives under names that cannot collide with it: `engine` is
    what the estimator said, `obs` and `n` are his own record and its size, `prior` is the job, and the
    season columns are the record year by year. A grid locks everything it is not told to edit, so all
    of this is read-only by construction.
    """
    if df.is_empty() or key_col not in df.columns:
        return df
    vals = metric_values(v, metric)
    if vals.is_empty():
        return df
    who = tuple(df[key_col].to_list())
    keep = [c for c in ("obs", "n", "own_weight", "prior") if c in vals.columns]
    out = df.join(
        vals.filter(pl.col("player_id").is_in(list(who)))
        .select("player_id", pl.col("estimate").alias("engine"), *keep)
        .rename({"player_id": key_col}),
        on=key_col, how="left",
    )
    if metric == GAMES_METRIC:
        long = attendance(seasons).filter(pl.col("player_id").is_in(list(who))).rename(
            {"games": "value"})
    else:
        long = metric_seasons((metric,), who, seasons)
    wide_hist = season_wide(long, seasons=seasons)
    if not wide_hist.is_empty() and wide_hist.width > 1:
        out = out.join(wide_hist.rename({"player_id": key_col}), on=key_col, how="left")
    return out


def evidence_config(metric: str, digits: int = 3, seasons: tuple[int, ...] = HISTORY_VIEW) -> dict:
    """Column config for whatever `with_evidence` added: the weight as a bar, the record as a line."""
    return {**fixed(digits, "engine", "obs", "prior"), **fixed(1, "n"), **bar("own_weight"),
            **season_config(seasons, digits)}


def room_panel(v: View, metric: str, player_ids: tuple[str, ...], key: str,
               position: str | None = None) -> None:
    """A room on one metric: where the position sits, then every man's estimate and his own record.

    The team-page counterpart of `override_panel`. The grid above it edits a dozen numbers at once,
    which is the fast way to work and the easy way to type a number nobody could defend; this is the
    check on it -- the position's quartiles, and each man's five seasons beside the estimate.
    """
    digits = 2 if metric == GAMES_METRIC else 3
    spread = metric_spread(v, metric)
    if not spread.is_empty():
        shown = spread if position is None else spread.filter(pl.col("position") == position)
        if not shown.is_empty():
            st.caption(f"What {label(metric)} is worth around the league, by depth slot")
            table(shown, digits=digits, height=200)
    room = room_evidence(v, metric, player_ids)
    if room.is_empty():
        note("Nobody in this room has this metric.")
        return
    st.caption("This room: the estimate, the evidence under it, and the record season by season")
    table(room.drop("player_id"), digits=digits,
          config={**evidence_config(metric, digits), **fixed(digits, "estimate", "applied"),
                  **bar("own_weight")},
          height=min(80 + 35 * room.height, 460))
    note("`estimate` is the engine's, `applied` is what ran, `own_weight` is how much of the estimate "
         "is the man rather than his job. A number with a low weight and no record behind it is an "
         "assumption about a role — which is the kind worth overriding.")


def with_edits(df: pl.DataFrame, level: str, key_col: str) -> pl.DataFrame:
    """Add `edited` and `edits` to any frame keyed on a team code or a player id.

    So that a table of a hundred players says which three of them carry an override, and what it says,
    without the reader having to hold the Edits page in their head.
    """
    by_key: dict[str, list[str]] = {}
    for o in live().items:
        if o.level != level:
            continue
        op = "=" if o.mode == "set" else "x"
        wk = f" wk{o.week}" if o.week is not None else ""
        by_key.setdefault(o.key, []).append(f"{o.field}{wk} {op}{o.value:g}")
    if not by_key or key_col not in df.columns:
        return df.with_columns(pl.lit(False).alias("edited"), pl.lit("").alias("edits"))
    marks = pl.DataFrame(
        {key_col: list(by_key), "edits": [" · ".join(v) for v in by_key.values()]},
        schema={key_col: df.schema[key_col], "edits": pl.String},
    )
    return df.join(marks, on=key_col, how="left").with_columns(
        pl.col("edits").is_not_null().alias("edited"),
    ).with_columns(pl.col("edits").fill_null(""))


# --------------------------------------------------------------------------- #
# the editing widgets, so every page edits the same way
# --------------------------------------------------------------------------- #
def _keyed(name: str) -> str:
    """A widget key that changes whenever the scenario does, so no widget outlives its edit.

    Streamlit remembers a widget by its key. Without this, dropping an edit and leaving the widget
    holding the value would re-apply it on the very next rerun -- a reset button that resets nothing,
    which is the one thing a reset button must not be. Tying the key to the scenario's digest makes
    every editor rebuild itself from the scenario after any change, so what is on screen is what ran.
    """
    return f"{name}@{live().digest}"


def knob(
    level: str,
    key_value: str,
    field_name: str,
    base: float | None,
    week: int | None = None,
    fmt: str = "%.4f",
    widget_key: str | None = None,
    modes: tuple[str, ...] = overrides.MODES,
) -> None:
    """One `BASE / x Adj / USED` triplet: what it was, what to do to it, what it becomes.

    The workbook's one good idea, as three columns and a button. `base` is the engine's own number, so
    it is shown rather than edited; the edit is either an absolute value or a multiplier, and the
    result is written into the live scenario. Resetting removes the edit rather than setting it back,
    which matters: a reset knob follows the engine again if the data underneath it moves.
    """
    sc = live()
    existing = next(
        (o for o in sc.items
         if o.level == level and o.key == key_value and o.field == field_name and o.week == week),
        None,
    )
    wk = _keyed(widget_key or f"{level}:{key_value}:{field_name}:{week}")
    cols = st.columns([2, 2, 2, 2, 1])
    cols[0].markdown(f"**{label(field_name)}**" + (f" · wk{week}" if week is not None else ""))
    cols[1].markdown(f"base `{'—' if base is None else fmt % base}`")
    mode = cols[2].selectbox(
        "mode", modes, key=f"{wk}:mode", label_visibility="collapsed",
        index=modes.index(existing.mode) if existing and existing.mode in modes else 0,
    )
    default = float(existing.value) if existing else (1.0 if mode == "multiply" else (base or 0.0))
    value = cols[3].number_input("value", value=default, key=f"{wk}:value",
                                 label_visibility="collapsed", format=fmt, step=None)
    if cols[4].button("↺", key=f"{wk}:reset", help="drop this edit and follow the engine again",
                      disabled=existing is None):
        drop_edit(level, key_value, field_name, week)
        st.rerun()
    unchanged = (mode == "multiply" and value == 1.0) or (base is not None and value == base
                                                          and mode == "set")
    if existing is None and unchanged:
        return
    if existing is not None and existing.mode == mode and float(existing.value) == float(value):
        return
    if unchanged and existing is not None:
        drop_edit(level, key_value, field_name, week)
    else:
        edit(Override(level=level, key=key_value, field=field_name, mode=mode, value=float(value),
                      week=week, base=base))
    st.rerun()


def edits_from_grid(
    before: pl.DataFrame,
    after: pl.DataFrame,
    level: str,
    key_col: str,
    fields: tuple[str, ...] | list[str],
    week_col: str | None = None,
    base_frame: pl.DataFrame | None = None,
    require_base: bool = False,
) -> list[Override]:
    """Turn an edited `st.data_editor` grid into overrides: one per cell whose value moved.

    Typing over a number is the whole interaction on the team page, so the grid is the editor and this
    is the translation. Only cells that actually changed become edits -- a grid rerun with nothing
    typed produces nothing -- and an existing edit keeps its original `base`, so the recorded "what it
    was" stays the engine's number rather than becoming last edit's answer.

    `base_frame` is the engine's own numbers, keyed the same way, and is what a new edit records as its
    base where it has them. It matters on a grid of a whole roster, where the number on screen can
    itself be the consequence of somebody else's edit: a receiver pushed to slot 2 by the promotion
    above him must not have 2 recorded as what the published chart said.

    `require_base` drops a typed cell that has no engine number behind it at all -- a completion
    percentage on a running back. The engine reports such an edit unapplied rather than obeying it, so
    writing it down would put a row in the edit list that never moves a projection.
    """
    keep = [f for f in fields if f in before.columns and f in after.columns]
    if not keep or before.height != after.height:
        return []
    cols = [key_col] + ([week_col] if week_col else []) + keep
    existing = {o.id: o for o in live().items}
    engine = _base_lookup(base_frame, key_col, keep, week_col)
    out: list[Override] = []
    for was, now in zip(before.select(cols).rows(named=True),
                        after.select(cols).rows(named=True), strict=False):
        key_value = was[key_col]
        week = int(was[week_col]) if week_col else None
        for f in keep:
            old, new = was[f], now[f]
            if new is None or old == new:
                continue
            prior = existing.get((level, key_value, f, week))
            if prior is not None:
                base = prior.base
            elif (key_value, week, f) in engine:
                base = engine[(key_value, week, f)]
            else:
                base = None if old is None else float(old)
            if require_base and base is None:
                continue
            out.append(Override(level=level, key=key_value, field=f, mode="set",
                                value=float(new), week=week, base=base))
    return out


def _base_lookup(base_frame: pl.DataFrame | None, key_col: str, fields: list[str],
                 week_col: str | None) -> dict[tuple[str, int | None, str], float]:
    """`(key, week, field) -> the engine's number`, from a frame shaped like the grid it explains."""
    if base_frame is None or base_frame.is_empty() or key_col not in base_frame.columns:
        return {}
    have = [f for f in fields if f in base_frame.columns]
    wk = week_col if week_col and week_col in base_frame.columns else None
    if not have:
        return {}
    out: dict[tuple[str, int | None, str], float] = {}
    for row in base_frame.select([key_col] + ([wk] if wk else []) + have).rows(named=True):
        week = int(row[wk]) if wk and row[wk] is not None else None
        for f in have:
            if row[f] is not None:
                out[(row[key_col], week, f)] = float(row[f])
    return out


def grid(
    df: pl.DataFrame,
    level: str,
    key_col: str,
    fields: tuple[str, ...] | list[str],
    key: str,
    week_col: str | None = None,
    digits: int = 3,
    config: dict | None = None,
    height: int | str = "auto",
    hide: tuple[str, ...] = (),
    base_frame: pl.DataFrame | None = None,
    require_base: bool = False,
    batch: bool = True,
) -> None:
    """An editable table. Type over numbers; `batch` decides whether each one lands on its own.

    Everything not in `fields` is locked, because those columns are either identity or outputs -- a
    projected total is the product of the inputs beside it, and typing over the product would be a
    number the engine immediately contradicts.

    `batch` is the default because a sheet is where somebody changes eight numbers, and writing on the
    keystroke made that eight scenarios, eight engine runs and eight full redraws -- with the scroll
    position lost each time, since a new digest rebuilds the editor. Held instead: the editor keeps its
    own pending cells (which costs nothing, the projection is unchanged and therefore still cached), the
    count is reported, and Apply writes them as one batch. Discard rebuilds the editor from the
    scenario. `batch=False` restores the write-on-keystroke behaviour for a sheet where one cell is the
    whole interaction.

    `depth_slot` is a rank rather than a value, so a cell holding one reads as "his number" while what
    it actually does is renumber the room around him. It is editable here anyway, but only on the team
    sheet, where the room is on screen in slot order and the renumbering is therefore something the
    reader watches happen rather than something that happens to them. `depth_editor` remains the way to
    move one man when his room is not what you are looking at.
    """
    editable = [f for f in fields if f in df.columns]
    locked = [c for c in df.columns if c not in editable]
    after = st.data_editor(
        df, key=batch_key(key), hide_index=True, width="stretch", height=height,
        disabled=locked, num_rows="fixed",
        # the editable columns are formatted last but for the caller: a percent format on a cell
        # somebody has to type into shows 28.00% and takes 0.28, and one of those is a typo waiting
        column_config={**auto_config(df, digits),
                       **{c: None for c in hide if c in df.columns},
                       **fixed(digits, *editable), **(config or {})},
    )
    edits = edits_from_grid(df, after, level, key_col, editable, week_col,
                            base_frame=base_frame, require_base=require_base)
    if not batch:
        if edits:
            edit(*edits)
            st.rerun()
        return
    named = (display_names(view(), tuple(dict.fromkeys(o.key for o in edits)))
             if edits and level == "player" else None)
    apply_bar(edits, key, names=named)


def _salt(key: str) -> int:
    """A counter per editor, so Discard can rebuild one the scenario has not changed."""
    return int(st.session_state.get(f"salt:{key}", 0))


def _bump(key: str) -> None:
    st.session_state[f"salt:{key}"] = _salt(key) + 1


def batch_key(key: str) -> str:
    """The widget key a batched editor must use for `apply_bar`'s Discard to be able to rebuild it.

    A `data_editor` holds its own pending edits, which is what makes batching free — and also means the
    only way to throw them away is to hand Streamlit a key it has never seen. Any hand-rolled editor that
    ends in `apply_bar` keys itself through here rather than through `_keyed` directly.
    """
    return _keyed(f"{key}#{_salt(key)}")


def apply_bar(edits: Sequence[Override], key: str, names: dict[str, str] | None = None,
              idle: str = "Type over any number in the table; nothing is written until you apply.",
              ) -> bool:
    """`N edits pending · Apply · Discard`, and the writing of them. The other half of `batch`.

    Shared by every batched surface so the bargain is always the same one: what is typed is visible,
    counted and reversible before it is a scenario, and applying it costs one run rather than one per
    number.
    """
    if not edits:
        st.caption(idle)
        return False
    where = ""
    if names:
        who = list(dict.fromkeys(names.get(o.key, o.key) for o in edits))
        where = " · " + (", ".join(who[:3]) + (f" and {len(who) - 3} more" if len(who) > 3 else ""))
    left, right, spare = st.columns([2, 1, 6], vertical_alignment="center")
    go = left.button(f"Apply {len(edits)} change{'s' if len(edits) != 1 else ''}", type="primary",
                     width="stretch", key=f"{key}:apply:{_salt(key)}",
                     help="write these into the live scenario and re-run the projection once")
    if right.button("Discard", width="stretch", key=f"{key}:discard:{_salt(key)}",
                    help="forget what is typed and read the scenario again"):
        _bump(key)
        st.rerun()
    spare.caption(
        f"pending: " + " · ".join(f"**{label(o.field)}**"
                                  + (f" wk{o.week}" if o.week is not None else "")
                                  + f" {'=' if o.mode == 'set' else 'x'}{o.value:g}"
                                  for o in edits[:4])
        + (f" and {len(edits) - 4} more" if len(edits) > 4 else "") + where
    )
    if go:
        edit(*edits)
        _bump(key)
        st.rerun()
    return False


def edited_badge(v: View, level: str, key_value: str) -> None:
    """Say so, wherever an edited thing is shown."""
    touching = live().touching(level, key_value)
    if touching:
        st.info(" · ".join(o.label for o in touching), icon="✏️")


def edit_list(level: str, key_value: str, heading: str | None = None,
              v: View | None = None, key: str = "edit") -> bool:
    """Every edit on one thing: what it was, what it is now, whether it has gone stale, and a drop button.

    The badge says an edit exists; this says what it did. Beside the number it changed, because an
    override the reader has to go to another page to identify is an override they will forget they made.

    Given a `View`, the engine's *current* number comes from the provenance log as well, and is shown
    when it has moved away from the base the edit was recorded against. That gap is the one thing a
    saved scenario cannot tell you on its own: an edit typed in August against 0.24 is a different
    claim in October when the estimator has come round to 0.30 by itself.

    `key` scopes the drop buttons to the caller, because since the redesign one man's edits can honestly
    appear twice on a page -- in the panel beside a list and again in the section about him -- and two
    identical widget keys are an exception rather than two working buttons.
    """
    items = live().touching(level, key_value)
    if not items:
        return False
    if heading:
        st.markdown(heading)
    now = {}
    if v is not None:
        prov = provenance(v)
        if not prov.is_empty():
            now = {(r["field"], r["week"]): r for r in
                   prov.filter((pl.col("level") == level) & (pl.col("key") == key_value))
                   .rows(named=True)}
    for o in items:
        row = st.columns([3, 2, 2, 2, 1])
        row[0].markdown(f"**{label(o.field)}**" + (f" · wk{o.week}" if o.week is not None else ""))
        row[1].markdown(f"base `{'—' if o.base is None else f'{o.base:.4g}'}`")
        row[2].markdown(("now `= " if o.mode == "set" else "now `x ") + f"{o.value:g}`")
        rec = now.get((o.field, o.week))
        engine = None if rec is None else rec.get("base_now")
        if engine is not None and (o.base is None or abs(float(engine) - float(o.base)) > 1e-6):
            row[3].markdown(f":orange[engine now `{float(engine):.4g}`]")
        elif rec is not None and not rec.get("applied", True):
            row[3].markdown(f":red[not applied: {rec.get('reason') or 'no matching rows'}]")
        if row[4].button("↺", key=_keyed(f"{key}:{level}:{key_value}:{o.field}:{o.week}"),
                         help="drop this edit and follow the engine again"):
            drop_edit(level, key_value, o.field, o.week)
            st.rerun()
    return True


def quick_choices(metric: str, row: dict, spread: dict | None = None) -> list[tuple[str, float]]:
    """The arguments somebody is actually about to make about this number, as named values.

    Every candidate is a number already on screen -- his own record, his job's prior, the position's
    median, the top of the position, a full slate of games -- so a one-click override is still an
    override made on evidence rather than a nudge in a direction. Pure, and separately tested: what a
    button offers is a claim about the metric and deserves to be checked without a browser.
    """
    est = row.get("estimate")
    out: list[tuple[str, float]] = []

    def add(name: str, value: float | None) -> None:
        if value is None:
            return
        v = float(value)
        if est is not None and abs(v - float(est)) < 1e-9:
            return                                    # a button that changes nothing is furniture
        if any(abs(v - had) < 1e-9 for _, had in out):
            return
        out.append((name, v))

    if metric == GAMES_METRIC:
        full = float(REG_WEEKS - 1)
        # the healthy-starter case first, because it is the common one and the one the estimator is
        # most reluctant about: shrinkage pulls a man towards his slot's average attendance, and an
        # elite starter nobody has reported hurt is not an average starter.
        add("every game", full)
        add("one missed", full - 1.0)
        add("three missed", full - 3.0)
        if (row.get("n") or 0) > 0:
            add("his own average", row.get("obs"))
    else:
        if (row.get("n") or 0) > 0:
            add("his own record", row.get("obs"))
        add("his job's prior", row.get("prior"))
        if spread:
            add("position median", spread.get("median"))
            add("top of position", spread.get("p90"))
    return out


def quick_set(level: str, key_value: str, field_name: str, choices: list[tuple[str, float]],
              base: float | None, key: str, fmt: str = "%.3f") -> None:
    """Named values as buttons: one click writes the override, with the engine's number as its base."""
    if not choices:
        return
    cols = st.columns(len(choices))
    for col, (name, value) in zip(cols, choices, strict=True):
        if col.button(f"{name} · {fmt % value}",
                      key=_keyed(f"{key}:quick:{field_name}:{name}"), width="stretch",
                      help=f"set {label(field_name)} to {fmt % value}"):
            edit(Override(level=level, key=key_value, field=field_name, mode="set",
                          value=float(value), week=None, base=base))
            st.rerun()


def _spread_row(spread: pl.DataFrame, position: str | None, slot: int | None) -> dict | None:
    """The quantile row that matches a player's job, preferring his slot over his position."""
    if spread.is_empty() or position is None:
        return None
    mine = spread.filter(pl.col("position") == position)
    if mine.is_empty():
        return None
    if slot is not None:
        exact = mine.filter(pl.col("slot") == int(slot))
        if not exact.is_empty():
            return exact.row(0, named=True)
    fallback = mine.filter(pl.col("slot").is_null())
    return fallback.row(0, named=True) if not fallback.is_empty() else mine.row(0, named=True)


def override_panel(v: View, player_id: str, metric: str, key: str = "metric",
                   base: float | None = None) -> None:
    """One number, everything known about it, and the ways to change it -- on one screen.

    This is the answer to "override on reliable data": the estimator's number, what the projection is
    actually running on, the player's own record season by season, what the job is worth to everyone
    who has held it, where he sits in the league, and only then the input box. The order is the
    argument -- evidence, then comparison, then the edit -- and the room table underneath is there
    because a share is zero-sum: raising his takes it from the men listed below him.
    """
    vals = metric_values(v, metric)
    row = {}
    if not vals.is_empty():
        mine = vals.filter(pl.col("player_id") == player_id)
        row = mine.row(0, named=True) if not mine.is_empty() else {}
    est = row.get("estimate") if row.get("estimate") is not None else base
    applied = row.get("applied")
    digits = 2 if metric == GAMES_METRIC else 3
    fmt = f"%.{digits}f"
    spread = metric_spread(v, metric)
    against = _spread_row(spread, row.get("position"), row.get("depth_slot"))

    cards = st.columns(4)
    cards[0].metric("engine says", "—" if est is None else fmt % est,
                    help="the estimator's own number, before any edit")
    delta = None if (applied is None or est is None or abs(applied - est) < 1e-9) else applied - est
    cards[1].metric("projection is using", "—" if applied is None else fmt % applied,
                    delta=None if delta is None else fmt % delta,
                    help="what actually ran; it differs from the engine only where somebody typed")
    n, obs = row.get("n"), row.get("obs")
    cards[2].metric("his own record", "—" if obs is None else fmt % obs,
                    help=f"unshrunk, over {n:,.0f} {row.get('units', 'observations')}" if n
                    else "no record of his own")
    cards[3].metric("his job is worth", "—" if row.get("prior") is None else fmt % row["prior"],
                    help="what everybody who has held this position and depth slot averaged")

    weight = row.get("own_weight")
    if weight is not None:
        st.progress(min(max(float(weight), 0.0), 1.0),
                    text=f"the estimate is {float(weight):.0%} his own record and "
                         f"{1 - float(weight):.0%} his job's — "
                         + ("thin evidence, so the model is mostly assuming"
                            if float(weight) < 0.4 else "his own play is carrying it"))

    left, right = st.columns([3, 2])
    with left:
        if metric == GAMES_METRIC:
            long = attendance().filter(pl.col("player_id") == player_id).rename({"games": "value"})
            caption = "games played, per season"
        else:
            long = metric_seasons((metric,), (player_id,))
            caption = f"{label(metric)}, per season, with the opportunity behind it"
        if long.is_empty():
            note("No record of his own in the seasons on view — the number above is his job's average.")
        else:
            st.caption(caption)
            long = collapse_seasons(long)
            show = ["season", "value"] + [c for c in ("n", "num", "team_games", "rate")
                                          if c in long.columns]
            table(long.select(show).sort("season", descending=True), digits=digits,
                  config={**fixed(digits, "rate"),
                          "value": st.column_config.NumberColumn(label(metric),
                                                                 format=f"%.{digits}f")},
                  height=220)
    with right:
        if against:
            st.caption(f"{against['position']}"
                       + (f" · slot {against['slot']}" if against.get("slot") is not None
                          else " · all slots")
                       + f" · {against['players']} players")
            pct = percentile_of(vals["estimate"], est)
            if pct is not None:
                st.markdown(f"he is at the **{pct:.0%}** mark of everyone the model gives this metric")
            table(pl.DataFrame([{k: against[k] for k in ("p10", "p25", "median", "p75", "p90", "best")
                                 if k in against}]), digits=digits, height=80)

    st.markdown("**Set it to something known**")
    quick_set("player", player_id, metric, quick_choices(metric, {**row, "estimate": est}, against),
              base=est, key=f"{key}:{player_id}", fmt=fmt)
    knob("player", player_id, metric, base=est, fmt=fmt,
         widget_key=f"{key}:{player_id}:{metric}")
    note(
        "`base` is the estimator's own number, however many times the knob is turned. Type an absolute "
        "value with `set` or a factor with `multiply`; ↺ drops the edit and the estimator has the "
        "metric back — including any later change in the data underneath it."
    )

    if row.get("team") and row.get("position"):
        room = room_evidence(v, metric, tuple(
            vals.filter((pl.col("team") == row["team"]) & (pl.col("position") == row["position"]))
            ["player_id"].to_list()
        ))
        if not room.is_empty():
            with st.expander(f"The rest of the {row['team']} {row['position']}s on this metric"):
                note("Shares are zero-sum inside a team: what you give him is taken from these rows "
                     "when the pool is normalized.")
                table(room.drop("player_id"), digits=digits,
                      config={**season_config(digits=digits), **bar("own_weight")})


def metric_knob(v: View, player_id: str, bases: dict[str, float], key: str = "metric") -> None:
    """Pick one of a player's own estimates and override it, without leaving the page.

    The grids edit a room at a time and the Edits page edits a list; this is the third case, and
    the common one: reading one player, disbelieving one number, changing that number where it is shown.

    `bases` is metric -> the estimator's own value, not the edited one, so the label beside each metric
    in the picker is what the model said rather than what the last edit made of it. The picker names the
    evidence as well as the metric, because which number is worth arguing with is itself a judgement:
    one resting on 9 routes and one resting on 900 should not read the same in a dropdown.
    """
    editable = [m for m in bases if m in overrides.PLAYER_FIELDS]
    if not editable:
        st.caption("Nothing here is editable: every number in this table is derived from another one.")
        return
    edited = {o.field for o in live().touching("player", player_id)}

    def describe(m: str) -> str:
        mark = "✏️ " if m in edited else ""
        b = bases.get(m)
        return f"{mark}{label(m)}" + ("" if b is None else f"  ·  engine {b:,.3f}")

    picked = st.selectbox("Which number do you want to argue with?", editable, key=_keyed(key),
                          format_func=describe)
    override_panel(v, player_id, picked, key=key, base=bases.get(picked))


# --------------------------------------------------------------------------- #
# the override, beside the number it changes
# --------------------------------------------------------------------------- #
# `override_panel` is the full argument -- four cards, the record season by season, the position's
# quartiles, the room -- and it is the right surface when arguing with a number is the reason you opened
# the page. It is the wrong surface for the other case, which is the common one: reading a player, or a
# room, or one afternoon, and disagreeing with a number in passing. That used to mean leaving what you
# were reading, picking the metric out of a dropdown and losing your place.
#
# So the same evidence, folded into a popover that opens where the number is drawn. Everything in it is
# already on screen elsewhere or one cached call away; nothing new is computed, and the write path is the
# same `Override` the grids and the panel produce -- so an edit made here resets, reports and reconciles
# like any other.
def _record_line(v: View, player_id: str, metric: str, digits: int) -> str:
    """His own record, season by season, as one line of text. Compact enough for a popover."""
    if metric == GAMES_METRIC:
        long = attendance().filter(pl.col("player_id") == player_id).rename({"games": "value"})
    else:
        long = metric_seasons((metric,), (player_id,))
    if long.is_empty():
        return ""
    long = collapse_seasons(long).sort("season")
    return " · ".join(f"{int(r['season'])} **{fmt_num(r['value'], digits)}**"
                      for r in long.rows(named=True) if r.get("value") is not None)


def knob_popover(v: View, player_id: str, metric: str, base: float | None = None,
                 key: str = "knob", week: int | None = None, digits: int | None = None) -> None:
    """The ✎ beside a number: what is known about it, the named values, and the exact box.

    Deliberately not a shrunk `override_panel`. What survives the shrinking is what an override actually
    turns on -- his own record and how much of it there is, what the job is worth, where the number sits
    at the position -- and what is dropped is the room table, which needs width and is on the page the
    ✎ was clicked from anyway.
    """
    if metric not in overrides.PLAYER_FIELDS and metric not in overrides.GAME_ALL_FIELDS:
        return
    if _POPOVER_DEPTH:
        return                            # already inside one; a popover cannot hold a popover
    digits = digits if digits is not None else (2 if metric == GAMES_METRIC else 3)
    fmt = f"%.{digits}f"
    existing = next((o for o in live().touching("player", player_id)
                     if o.field == metric and o.week == week), None)
    scope = f" · week {week}" if week is not None else ""
    with popover("✏️" if existing else "✎", help=f"adjust {label(metric)}{scope}"):
        vals = metric_values(v, metric)
        row: dict = {}
        if not vals.is_empty():
            mine = vals.filter(pl.col("player_id") == player_id)
            row = mine.row(0, named=True) if not mine.is_empty() else {}
        est = row.get("estimate") if row.get("estimate") is not None else base
        against = _spread_row(metric_spread(v, metric), row.get("position"), row.get("depth_slot"))

        st.markdown(f"**{label(metric)}**" + (f" · week {week}" if week is not None else ""))
        kv(**{"engine says": fmt_num(est, digits), "running": fmt_num(row.get("applied"), digits),
              "his record": fmt_num(row.get("obs"), digits), "his job": fmt_num(row.get("prior"), digits),
              "sample": None if row.get("n") is None
              else f"{float(row['n']):,.0f} {row.get('units') or ''}".strip()})
        weight = row.get("own_weight")
        if weight is not None:
            st.html(meter_html("how much of it is him", float(weight), maximum=1.0, digits=2,
                               percent=True, thin=True,
                               foot=("thin evidence — mostly his job's average"
                                     if float(weight) < 0.4 else "his own play is carrying it",)))
        if against:
            pct = percentile_of(vals["estimate"], est) if not vals.is_empty() else None
            st.caption(
                f"{against['position']}"
                + (f" slot {against['slot']}" if against.get("slot") is not None else "")
                + f" · median {fmt_num(against.get('median'), digits)}"
                + f" · p90 {fmt_num(against.get('p90'), digits)}"
                + (f" · he is at the {pct:.0%} mark" if pct is not None else "")
            )
        record = _record_line(v, player_id, metric, digits)
        if record:
            st.caption("his record")
            st.markdown(record)

        st.divider()
        choices = quick_choices(metric, {**row, "estimate": est}, against)
        for i in range(0, len(choices), 2):
            cols = st.columns(2)
            for col, (name, value) in zip(cols, choices[i:i + 2], strict=False):
                if col.button(f"{name} · {fmt % value}", width="stretch",
                              key=_keyed(f"{key}:{player_id}:{metric}:{week}:{name}")):
                    edit(Override(level="player", key=player_id, field=metric, mode="set",
                                  value=float(value), week=week, base=est))
                    st.rerun()
        box, setter, dropper = st.columns([3, 2, 1])
        typed = box.number_input(
            "exact", value=float(existing.value) if existing else float(est or 0.0), format=fmt,
            step=None, key=_keyed(f"{key}:{player_id}:{metric}:{week}:exact"),
            label_visibility="collapsed",
        )
        if setter.button("set", width="stretch", type="primary",
                         key=_keyed(f"{key}:{player_id}:{metric}:{week}:set")):
            edit(Override(level="player", key=player_id, field=metric, mode="set",
                          value=float(typed), week=week, base=est))
            st.rerun()
        if dropper.button("↺", disabled=existing is None, help="drop this edit, follow the engine again",
                          key=_keyed(f"{key}:{player_id}:{metric}:{week}:reset")):
            drop_edit("player", player_id, metric, week)
            st.rerun()
        if existing is not None:
            st.caption(f"edited: {existing.mode} {existing.value:g} "
                       f"(engine had {fmt_num(existing.base, digits)})")


# --------------------------------------------------------------------------- #
# the bench: one man, every number he has, typed in one pass
# --------------------------------------------------------------------------- #
# The three editing surfaces that came before this one each solve a real problem and share a defect. The
# ✎ popover is one number in its context; `grid` is one room across a few columns; the Edits page is
# one field across a population. Disagreeing with four things about one player fits none of them: it is
# four popovers, each of which closes on the write it makes, or it is a horizontal scroll through a
# nine-hundred-row sheet looking for his line. And a per-game version of the same disagreement meant a
# different page again.
#
# So: one man, every number he has, one row each, with an empty column to type into and a week scope over
# the top. Nothing is written until Apply, so six changes are one batch, one engine run and one redraw.
# It is deliberately not new evidence -- every column in it is a column `ratings` already produced, and
# the ✎ popover remains the place to go when the question is *should* this number move rather than *make*
# it move.
BENCH_SETS = ("what moves it", "shares", "rates", "availability", "everything")
BENCH_TYPED = ("set to", "x by", "drop")
BENCH_OPEN = "bench:open"                       # the session key the whole app opens the dialog through


def bench_fields(v: View, player_id: str, column_set: str = "what moves it", *,
                 per_week: bool = False) -> list[str]:
    """Which of his knobs the bench shows, in the order they move a projection.

    `per_week` is not a filter on presentation, it is a filter on what a week can carry: `expected_games`
    is a count of games and `depth_slot` is a place in an ordering, and neither means anything about one
    afternoon. `p_play` is the other way round -- it is how "he is out in week 5" is said, and there is no
    season-wide version of it.
    """
    position = None
    part = participation(v).filter(pl.col("player_id") == player_id)
    if not part.is_empty():
        position = part.row(0, named=True).get("position")
    return bench_field_set(column_set, position, per_week=per_week)


def bench_field_set(column_set: str, position: str | None, *, per_week: bool = False) -> list[str]:
    """`bench_fields` without the lookup, so which knobs a scope offers is testable on its own."""
    if column_set == "availability":
        chosen: tuple[str, ...] = (GAMES_METRIC, overrides.DEPTH_FIELD, "p_play", "snap_share",
                                   "route_participation", "rush_participation")
    elif column_set == "shares":
        chosen = tuple(f for f in overrides.PLAYER_FIELDS if knob_kind(f) in ("pool", "presence"))
    elif column_set == "rates":
        chosen = tuple(f for f in overrides.PLAYER_FIELDS if knob_kind(f) == "rate")
    elif column_set == "everything":
        chosen = tuple(overrides.FIELDS["player"])
    else:
        chosen = ("p_play", *knobs_for(position), overrides.DEPTH_FIELD)
    allowed = (set(overrides.GAME_ALL_FIELDS) if per_week
               else set(overrides.PLAYER_FIELDS) | {overrides.DEPTH_FIELD})
    return [f for f in dict.fromkeys(chosen) if f in allowed]


def bench_bases(v: View, player_id: str, fields: Sequence[str],
                weeks: Sequence[int] = ()) -> dict[tuple[str, int | None], float]:
    """The engine's own number for every cell the bench can write, keyed `(field, week)`.

    From the *baseline* run, never from the live one, for the same reason the game sheet is: the number on
    screen in a scenario where his room has already been edited is the consequence of that edit, and
    recording it as "what the engine said" would make the second edit undroppable.
    """
    out: dict[tuple[str, int | None], float] = {}
    base = baseline(v)
    if weeks:
        want = [int(w) for w in weeks]
        for frame in (base.opp, base.adj):
            if frame is None or frame.is_empty() or "week" not in frame.columns:
                continue
            have = [f for f in fields if f in frame.columns]
            if not have:
                continue
            mine = frame.filter((pl.col("player_id") == player_id)
                                & pl.col("week").is_in(want))
            for row in mine.select("week", *have).rows(named=True):
                for f in have:
                    if row[f] is not None:
                        out.setdefault((f, int(row["week"])), float(row[f]))
        return out
    for frame in (base.part, base.shares, base.rates):
        if frame is None or frame.is_empty():
            continue
        have = [f for f in fields if f in frame.columns]
        mine = frame.filter(pl.col("player_id") == player_id) if have else frame.head(0)
        if mine.is_empty():
            continue
        row = mine.row(0, named=True)
        for f in have:
            if row.get(f) is not None:
                out.setdefault((f, None), float(row[f]))
    return out


def bench_frame(v: View, player_id: str, fields: Sequence[str], *,
                week: int | None = None) -> pl.DataFrame:
    """One row a knob: what it is, what the engine said, what is running, and the evidence in two numbers.

    The three empty columns on the end are the whole point of the surface. `set to` and `x by` are the two
    modes an override has, side by side rather than behind a dropdown, and `drop` is there so undoing four
    edits is also one pass rather than four buttons.
    """
    rated = ratings(v, player_id)
    known = {r["metric"]: r for r in rated.rows(named=True)} if not rated.is_empty() else {}
    engine = bench_bases(v, player_id, fields, () if week is None else (week,))
    running: dict[str, float] = {}
    if week is None:
        for frame in (participation(v), shares_wide(v), rates_wide(v)):
            mine = frame.filter(pl.col("player_id") == player_id)
            if mine.is_empty():
                continue
            for name, val in mine.row(0, named=True).items():
                if name in fields and not isinstance(val, bool) and isinstance(val, (int, float)):
                    running.setdefault(name, float(val))
    else:
        run = projection(v)
        for frame in (run.opp, run.adj):
            if frame is None or frame.is_empty() or "week" not in frame.columns:
                continue
            have = [f for f in fields if f in frame.columns]
            mine = (frame.filter((pl.col("player_id") == player_id) & (pl.col("week") == int(week)))
                    if have else frame.head(0))
            if mine.is_empty():
                continue
            row = mine.row(0, named=True)
            for f in have:
                if row.get(f) is not None:
                    running.setdefault(f, float(row[f]))
    edits = {(o.field, o.week): o for o in live().touching("player", player_id)}
    rows = []
    for f in fields:
        r = known.get(f, {})
        got = edits.get((f, week))
        rows.append({
            "field": f,
            "rating": label(f),
            # neither of these two is one of the four kinds: a slot is a place in an ordering and
            # `p_play` is one afternoon's availability, so they get their own marks rather than being
            # filed under "his own rate", which is where falling through would put them
            "kind": ("🪜" if f == overrides.DEPTH_FIELD else
                     "📅" if f == "p_play" else KIND_ICON.get(knob_kind(f), "·")),
            "engine": engine.get((f, week)),
            "running": running.get(f),
            "his record": None if r.get("obs") is None else float(r["obs"]),
            "his job": None if r.get("prior") is None else float(r["prior"]),
            "him": None if r.get("own_weight") is None else float(r["own_weight"]),
            "edited": "" if got is None else ("= " if got.mode == "set" else "x ") + f"{got.value:g}",
            "set to": None,
            "x by": None,
            "drop": False,
        })
    return pl.DataFrame(rows, schema={
        "field": pl.String, "rating": pl.String, "kind": pl.String, "engine": pl.Float64,
        "running": pl.Float64, "his record": pl.Float64, "his job": pl.Float64, "him": pl.Float64,
        "edited": pl.String, "set to": pl.Float64, "x by": pl.Float64, "drop": pl.Boolean,
    })


def bench_edits(after: pl.DataFrame, player_id: str, weeks: Sequence[int],
                bases: dict[tuple[str, int | None], float],
                ) -> tuple[list[Override], list[tuple[str, int | None]]]:
    """The typed bench as overrides and drops. Pure, so what a typed cell means is testable without a browser.

    A row can say three things and they are read in the order somebody means them: a dropped edit is a
    drop whatever else is in the row, then an absolute value, then a factor. A factor of exactly one and a
    value equal to the engine's own number are both dropped rather than written, because an override that
    changes nothing is a row in the edit list that will be mistaken for one that does.
    """
    if after.is_empty():
        return [], []
    spans: list[int | None] = [int(w) for w in weeks] if weeks else [None]
    writes: list[Override] = []
    drops: list[tuple[str, int | None]] = []
    for row in after.rows(named=True):
        field_name = row["field"]
        for week in spans:
            base = bases.get((field_name, week))
            if row.get("drop"):
                drops.append((field_name, week))
                continue
            typed, factor = row.get("set to"), row.get("x by")
            if typed is not None:
                if base is not None and abs(float(typed) - float(base)) < 1e-12:
                    continue
                writes.append(Override(level="player", key=player_id, field=field_name, mode="set",
                                       value=float(typed), week=week, base=base))
            elif factor is not None and abs(float(factor) - 1.0) > 1e-12:
                writes.append(Override(level="player", key=player_id, field=field_name,
                                       mode="multiply", value=float(factor), week=week, base=base))
    return writes, drops


def bench(v: View, player_id: str, *, key: str = "bench", compact: bool = False) -> None:
    """The whole editor: who he is, the scope, the knobs, and one Apply.

    The scope control is the part that earns the surface. A season override and a run of week overrides
    are the same claim over a different span -- "he is a bigger part of this offence than the model
    thinks" against "he is on a pitch count in December" -- and they were on different pages, so the
    second one mostly did not get made. Here it is a toggle over the same rows, and picking three weeks
    writes three edits with each week's own engine number recorded as its base.
    """
    b = board(v).filter(pl.col("player_id") == player_id)
    row = b.row(0, named=True) if not b.is_empty() else {}
    name = row.get("player") or player_id
    position, team = row.get("position"), row.get("team")

    st.markdown(f"#### {name}"
                + (f" · {position} · {team}" if position else "")
                + (f" — {float(row['fantasy_points']):.1f} pts" if row.get("fantasy_points") else ""))
    mine = live().touching("player", player_id)
    chips(
        f"{position} rank {int(row['position_rank'])}" if row.get("position_rank") else "",
        f"{float(row['expected_games']):.1f} games" if row.get("expected_games") else "",
        f"{len(mine)} edit{'s' if len(mine) != 1 else ''} on him" if mine else "no edits on him yet",
        edited=bool(mine),
    )

    scope_col, set_col = st.columns([2, 3], vertical_alignment="bottom")
    span = scope_col.segmented_control("Applies to", ["the season", "some weeks"], default="the season",
                                       key=f"{key}:span") or "the season"
    per_week = span == "some weeks"
    picked: list[int] = []
    if per_week:
        his = team_weeks(v, team) if team else weeks(v)
        picked = st.multiselect(
            "Which weeks", his, default=[], key=f"{key}:weeks",
            format_func=lambda w: f"week {int(w)}",
            help="Every week chosen gets its own edit, against its own engine number.",
        )
    column_set = set_col.segmented_control("Show", BENCH_SETS, default="what moves it",
                                           key=f"{key}:set") or "what moves it"

    fields = bench_fields(v, player_id, column_set, per_week=per_week)
    if not fields:
        st.info("Nothing on this list can be edited for one week." if per_week
                else "No editable knobs for him.")
        return
    if per_week and not picked:
        st.info("Pick the weeks this applies to. Until then there is nothing to write against.",
                icon="📅")
        return

    shown = None if not picked else int(min(picked))
    df = bench_frame(v, player_id, fields, week=shown)
    if per_week:
        others = len(picked) - 1
        st.caption(f"`engine` and `running` are week {shown}'s own numbers"
                   + (f"; the other {others} chosen week{'s' if others > 1 else ''} "
                      f"take{'' if others > 1 else 's'} the same value against "
                      f"{'their' if others > 1 else 'its'} own base" if others else "")
                   + ". `his record` and `his job` are the season estimate behind them.")

    sheet = f"{key}:sheet"                # the editor Discard has to be able to rebuild, salt and all
    after = st.data_editor(
        df, key=batch_key(sheet), hide_index=True, width="stretch",
        height=min(38 * (df.height + 1) + 3, 560) if not compact else 300,
        num_rows="fixed", disabled=[c for c in df.columns if c not in BENCH_TYPED],
        column_config={
            "field": None,
            "rating": st.column_config.TextColumn("rating", width="medium"),
            "kind": st.column_config.TextColumn("", width="small",
                                                help="📅 availability · ⚖️ a zero-sum pool · "
                                                     "🕒 presence · 🎯 his own rate"),
            **fixed(3, "engine", "running", "his record", "his job"),
            **bar("him"),
            "edited": st.column_config.TextColumn("edit", width="small",
                                                  help="what is already written on this knob"),
            "set to": st.column_config.NumberColumn(
                "set to", format="%.3f", width="small",
                help="an absolute value — leave it empty to change nothing"),
            "x by": st.column_config.NumberColumn(
                "x by", format="%.3f", width="small",
                help="a factor on the engine's number — 1.1 is ten per cent more"),
            "drop": st.column_config.CheckboxColumn(
                "drop", width="small", help="remove the edit on this knob and follow the engine again"),
        },
    )
    writes, drops = bench_edits(after, player_id, picked, bench_bases(v, player_id, fields, picked))
    drops = [(f, w) for f, w in drops
             if any(o.field == f and o.week == w for o in live().touching("player", player_id))]

    if not writes and not drops:
        st.caption("Type a number into **set to** or a factor into **x by**, on as many rows as you "
                   "like, then apply them together. Nothing is written until you do.")
    else:
        left, right, spare = st.columns([2, 1, 5], vertical_alignment="center")
        parts = [f"{len(writes)} change{'s' if len(writes) != 1 else ''}" if writes else "",
                 f"{len(drops)} drop{'s' if len(drops) != 1 else ''}" if drops else ""]
        if left.button("Apply " + " and ".join(p for p in parts if p), type="primary",
                       width="stretch", key=f"{sheet}:apply:{_salt(sheet)}"):
            for field_name, week in drops:
                drop_edit("player", player_id, field_name, week)
            if writes:
                edit(*writes)
            _bump(sheet)
            st.rerun()
        if right.button("Discard", width="stretch", key=f"{sheet}:discard:{_salt(sheet)}"):
            _bump(sheet)
            st.rerun()
        spare.caption(" · ".join(
            [f"**{label(o.field)}** {'=' if o.mode == 'set' else 'x'}{o.value:g}" for o in writes[:5]]
            + [f"drop **{label(f)}**" for f, _ in drops[:3]])
            + (f" · over {len(picked)} weeks" if len(picked) > 1 else ""))

    if mine:
        with st.expander(f"The {len(mine)} edit{'s' if len(mine) != 1 else ''} already on him"):
            edit_list("player", player_id, v=v, key=f"{key}:list")


def open_bench(player_id: str, week: int | None = None) -> None:
    """Ask for the editor, from anywhere. The dialog itself is opened by `controls` on the next run.

    Routed through session state rather than opened in place because a dialog has to be raised from the
    page body, and the buttons that want one are inside sidebars, columns and panels. One flag means every
    page gets the same editor from `ui.controls` without knowing it is there.

    `week` pre-scopes it, which is the whole reason a per-game surface can hand off to the same editor: a
    ✎ clicked on a week-5 line opens on week 5 rather than on the season, so the edit lands where the
    reader was looking. Written as widget state, which Streamlit reads as the scope control's default.
    """
    st.session_state[BENCH_OPEN] = player_id
    st.session_state["bench:dialog:span"] = "the season" if week is None else "some weeks"
    st.session_state["bench:dialog:weeks"] = [] if week is None else [int(week)]
    st.rerun()


def bench_target() -> str | None:
    return st.session_state.get(BENCH_OPEN)


def edit_button(player_id: str, *, key: str, label_text: str = "✎ Adjust", week: int | None = None,
                help_text: str = "every number behind his projection, in one editable list",
                width: str = "content") -> None:
    """The way into the bench, beside whatever is being read at the time."""
    n = len(live().touching("player", player_id))
    if st.button(label_text + (f" · {n}" if n else ""), key=key, width=width,
                 help=help_text + (f" — opening on week {int(week)}" if week is not None else "")):
        open_bench(player_id, week)


def bench_dialog(v: View) -> None:
    """Raise the editor if something asked for it. Called once by `controls`, so every page has it."""
    player_id = bench_target()
    if not player_id:
        return
    b = board(v).filter(pl.col("player_id") == player_id)
    who = b.row(0, named=True) if not b.is_empty() else {}
    title = "✎ " + str(who.get("player") or player_id)
    if who.get("position"):
        title += f" · {who['position']} · {who['team']}"

    def body() -> None:
        bench(v, player_id, key="bench:dialog")
        if st.button("Close", key="bench:dialog:close", width="stretch"):
            st.session_state.pop(BENCH_OPEN, None)
            st.rerun()

    st.dialog(title, width="large")(body)()


# --------------------------------------------------------------------------- #
# a player's ratings, drawn rather than tabulated
# --------------------------------------------------------------------------- #
# The table this replaces had eleven numeric columns and the reader's job was to diff two of them in their
# head, twenty times. What is actually being asked of each row is one question -- did the estimator
# believe the player or his job, and how much evidence does it have -- and that is a bar with two ticks on
# it. The ticks are `obs` and `prior`; the fill is what the projection ran on. When the fill sits on the
# left tick the model believes the man, when it sits on the right one it is describing a role.
def rating_axis(row: dict) -> float | None:
    """The top of the track for one estimate: 1.0 for a share, headroom over the numbers for a rate."""
    vals = [abs(float(row[c])) for c in ("obs", "prior", "used", "applied")
            if row.get(c) is not None]
    if not vals:
        return None
    top = max(vals)
    return 1.0 if top <= 1.0 else top * 1.15


def rating_meters(v: View, player_id: str, rated: pl.DataFrame, key: str = "ratings",
                  editable: bool = True) -> None:
    """One meter per estimate, each with the ✎ that changes it.

    Sorted so the arguable ones come first: a low `own_weight` is the model saying *I am describing the
    job, not the man*, which is the kind of number worth an override, and burying it under six
    well-measured ones in alphabetical order is how it goes unnoticed.
    """
    if rated.is_empty():
        st.info("No estimates for him — a player with no history and no slot the estimator recognises.")
        return
    order = rated.with_columns(
        pl.col("own_weight").fill_null(0.0).alias("_w"),
        pl.col("used").fill_null(0.0).abs().alias("_v"),
    ).sort(["_w", "_v"], descending=[False, True])
    for i, r in enumerate(order.rows(named=True)):
        metric = r["metric"]
        digits = 2 if metric == GAMES_METRIC else 3
        can = editable and metric in overrides.PLAYER_FIELDS
        wide, narrow = st.columns([11, 1], vertical_alignment="center")
        with wide:
            applied, used = r.get("applied"), r.get("used")
            moved = (applied is not None and used is not None
                     and abs(float(applied) - float(used)) > 1e-9)
            foot = [f"his record {fmt_num(r.get('obs'), digits)}"
                    + (f" over {float(r['n']):,.0f} {r.get('units') or ''}".rstrip()
                       if r.get("n") else " (none)"),
                    f"his job {fmt_num(r.get('prior'), digits)}",
                    f"{float(r['own_weight']):.0%} him" if r.get("own_weight") is not None else "",
                    f"✏️ engine had {fmt_num(used, digits)}" if moved else ""]
            st.html(meter_html(
                label(metric), applied if applied is not None else used,
                maximum=rating_axis(r), digits=digits,
                marks=(("his record", r.get("obs")), ("his job", r.get("prior"))),
                foot=[f for f in foot if f],
            ))
        with narrow:
            if can:
                # the row's place in the list is in the key as well as the metric, so a frame that
                # somehow carries a metric twice draws two working ✎ rather than raising on the second
                knob_popover(v, player_id, metric, base=used, key=f"{key}:meter:{i}")


# --------------------------------------------------------------------------- #
# charts, where a chart is the honest shape
# --------------------------------------------------------------------------- #
# `st.bar_chart` with a player's name on the x axis draws the names vertically and unreadably, which is
# why so much of this app fell back to tables with a bar in a cell. Altair ships with Streamlit, so a
# ranked comparison can be horizontal bars with the names beside them -- the shape the comparison
# actually is.
def _axis(title: str | None = None) -> dict:
    return {"title": title, "grid": False, "domain": False, "tickSize": 0, "labelFontSize": 11}


def rank_bars(df: pl.DataFrame, value: str, name: str = "player", *, height: int = 360,
              colour: str | None = None, digits: int | None = None, top: int = 20) -> None:
    """A ranked comparison as horizontal bars: the names readable, the gaps visible.

    The gap is the point. A table of the twenty most projected targets says who is first; the bars say
    whether first is ten clear of second or in a flat pack of nine, which is the difference between a
    ranking that decides something and one that does not.
    """
    if df.is_empty() or value not in df.columns or name not in df.columns:
        st.info("Nothing to draw.")
        return
    d = df.head(top)
    digits = digits if digits is not None else digits_for(value, 1)
    bars = (
        alt.Chart(d, height=height)
        .mark_bar(cornerRadiusEnd=3, opacity=0.9, color=colour or "#4c78a8")
        .encode(
            x=alt.X(f"{value}:Q", axis=alt.Axis(**_axis(label(value)))),
            y=alt.Y(f"{name}:N", sort="-x", axis=alt.Axis(**_axis(None))),
            tooltip=[c for c in (name, "position", "team", value) if c in d.columns],
        )
    )
    text = bars.mark_text(align="left", dx=4, fontSize=11, opacity=0.75).encode(
        text=alt.Text(f"{value}:Q", format=f",.{digits}f"))
    st.altair_chart(bars + text, width="stretch")


def season_bars(df: pl.DataFrame, x: str, y: str, *, mark: object = None, height: int = 240,
                digits: int = 1) -> None:
    """A record by season with the projection in it, the projected bar picked out from the measured ones.

    Two claims of different kinds -- what happened, and what is expected to happen -- so they are drawn
    on one axis and coloured differently rather than concatenated into a table where the last row is the
    odd one out and nothing says so.
    """
    if df.is_empty() or x not in df.columns or y not in df.columns:
        st.info("No history to draw.")
        return
    d = df.with_columns(
        pl.when(pl.col(x).cast(pl.String) == str(mark)).then(pl.lit("projected"))
        .otherwise(pl.lit("measured")).alias("kind")
    )
    bars = (
        alt.Chart(d, height=height)
        .mark_bar(cornerRadiusEnd=3, opacity=0.95)
        .encode(
            x=alt.X(f"{x}:O", axis=alt.Axis(**_axis(None)), sort=None),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(**_axis(label(y)))),
            color=alt.Color("kind:N", legend=None,
                            scale=alt.Scale(domain=["measured", "projected"],
                                            range=["#6f8fbf", "#e45756"])),
            tooltip=[c for c in (x, y, "team", "games", "kind") if c in d.columns],
        )
    )
    text = bars.mark_text(dy=-8, fontSize=11, opacity=0.8).encode(
        text=alt.Text(f"{y}:Q", format=f",.{digits}f"))
    st.altair_chart(bars + text, width="stretch")


def week_bars(df: pl.DataFrame, y: str = "fantasy_points", *, height: int = 220,
              band: tuple[str, str] | None = None) -> None:
    """The seventeen games, with the simulated band behind them where one has been run."""
    if df.is_empty() or y not in df.columns:
        st.info("No weeks to draw.")
        return
    base = alt.Chart(df, height=height)
    layers = []
    if band and all(c in df.columns for c in band):
        layers.append(base.mark_area(opacity=0.18, color="#6f8fbf").encode(
            x=alt.X("week:O", axis=alt.Axis(**_axis(None)), sort=None),
            y=alt.Y(f"{band[0]}:Q", axis=alt.Axis(**_axis(label(y)))),
            y2=alt.Y2(f"{band[1]}:Q"),
        ))
    layers.append(base.mark_bar(cornerRadiusEnd=2, opacity=0.9, color="#4c78a8").encode(
        x=alt.X("week:O", axis=alt.Axis(**_axis("week")), sort=None),
        y=alt.Y(f"{y}:Q", axis=alt.Axis(**_axis(label(y)))),
        tooltip=[c for c in ("week", "opponent", y, "p_play", "implied_points") if c in df.columns],
    ))
    st.altair_chart(alt.layer(*layers), width="stretch")


def scatter(df: pl.DataFrame, x: str, y: str, *, name: str = "player", height: int = 320,
            diagonal: bool = False) -> None:
    """Two numbers about the same players, which is the shape "projection against what happened" is."""
    if df.is_empty() or x not in df.columns or y not in df.columns:
        st.info("Nothing to draw.")
        return
    pts = (
        alt.Chart(df, height=height)
        .mark_circle(size=64, opacity=0.6)
        .encode(
            x=alt.X(f"{x}:Q", axis=alt.Axis(**_axis(label(x)))),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(**_axis(label(y)))),
            tooltip=[c for c in (name, "position", "team", x, y) if c in df.columns],
        )
    )
    if not diagonal:
        st.altair_chart(pts, width="stretch")
        return
    lo = float(min(df[x].min() or 0.0, df[y].min() or 0.0))
    hi = float(max(df[x].max() or 1.0, df[y].max() or 1.0))
    line = (alt.Chart(pl.DataFrame({"a": [lo, hi], "b": [lo, hi]}))
            .mark_line(strokeDash=[4, 4], opacity=0.5, color="grey")
            .encode(x="a:Q", y="b:Q"))
    st.altair_chart(pts + line, width="stretch")


# --------------------------------------------------------------------------- #
# one player, as a panel
# --------------------------------------------------------------------------- #
# The detail half of the list-and-detail pattern, and the reason it is here rather than on a page: the
# board's right-hand panel and the player page are the same thing at two sizes, and when they were two
# implementations they disagreed -- different tiles, different rounding, one of them missing the override
# marks. One function, two sizes.
PANEL_METRICS = 5           # how many estimates a compact panel draws before the rest go in the expander


def player_panel(v: View, row: dict, *, key: str = "panel", compact: bool = False,
                 metrics: int = PANEL_METRICS, estimates: bool = True, weeks: bool = True,
                 history: bool = True) -> None:
    """Everything about one player worth seeing without scrolling, and the ✎ on every number of it."""
    pid = row.get("player_id")
    if pid is None:
        st.info("No player selected.")
        return
    position, team = row.get("position"), row.get("team")
    head, act = st.columns([8, 2], vertical_alignment="center")
    head.markdown(f"### {row.get('player')} · {position} {team}")
    with act:
        # every panel in the app is a place somebody decides a number is wrong, so every panel is a
        # place the editor opens from -- one button, wherever the man is already on screen
        edit_button(pid, key=f"{key}:bench:{pid}", width="stretch")
    chips(
        rookie=bool(row.get("is_rookie")),
        changed_team=bool(row.get("changed_team")),
        startable="good:startable" if row.get("startable") else False,
        status=(f"warn:{row.get('status')}" if row.get("status") not in (None, "ACT") else False),
        edited="edit:carries an override" if live().touching("player", pid) else False,
    )
    tiles([
        {"name": "fantasy points", "value": row.get("fantasy_points"), "highlight": True,
         "sub": f"{fmt_num(row.get('points_per_game'), 2)} / game"},
        {"name": f"{position} rank", "value": row.get("position_rank"), "digits": 0,
         "sub": f"{fmt_num(row.get('overall_rank'), 0)} overall"},
        {"name": "games", "value": row.get("games"), "sub": f"tier {fmt_num(row.get('tier'), 0)}"},
        {"name": "vs avg starter", "value": row.get("vs_starter"), "signed": True,
         "sub": f"drop to next {fmt_num(row.get('drop_next'), 1)}"},
        {"name": "vs last season", "value": row.get("delta_points"), "signed": True,
         "sub": (f"{fmt_num(row.get('last_points'), 0)} in "
                 f"{fmt_num(row.get('last_games'), 0)} g" if row.get("last_points") else "no record")},
    ])
    st.markdown(f"**{stat_line(row, position)}**")

    if live().touching("player", pid):
        with st.container(border=True):
            edit_list("player", pid, heading="**Your overrides on him**", v=v, key=f"{key}:edits")

    if estimates:
        rated = ratings(v, pid)
        section("What the number is made of",
                "One meter per estimate. The fill is what the projection ran on; the two ticks are his "
                "own record and what his job is worth to everybody who has held it. A fill sitting on "
                "the left-hand tick is a measurement of the man; one sitting on the right is an "
                "assumption about a role, and the assumptions are sorted to the top because they are "
                "the ones worth arguing with. ✎ opens the evidence and changes it where it stands.")
        if rated.is_empty():
            st.info("No estimates for him.")
        else:
            arguable = rated.filter(pl.col("used").is_not_null())
            head = arguable if not compact else arguable.sort(
                pl.col("own_weight").fill_null(0.0)).head(metrics)
            rating_meters(v, pid, head, key=f"{key}:top")
            rest = arguable.filter(~pl.col("metric").is_in(head["metric"].to_list()))
            if not rest.is_empty():
                with st.expander(f"The other {rest.height} estimates"):
                    rating_meters(v, pid, rest, key=f"{key}:rest")

    if weeks:
        wk = weekly(v).filter(pl.col("player_id") == pid).sort("week")
        if not wk.is_empty():
            section("Week by week", "Each bar is an expectation with availability already in it: a bye "
                                    "is absent, and a week he is 60% likely to play is 60% of a line.")
            week_bars(wk)
    if history:
        hist = actuals(HISTORY_VIEW, v.scoring).filter(pl.col("player_id") == pid)
        if not hist.is_empty() and "games" in hist.columns:
            past = hist.select(
                pl.col("season").cast(pl.String),
                (pl.col("fantasy_points") / pl.col("games").cast(pl.Float64))
                .alias("points_per_game"),
            )
            both = pl.concat([past, pl.DataFrame({
                "season": [str(v.season)],
                "points_per_game": [float(row.get("points_per_game") or 0.0)],
            })], how="vertical_relaxed").sort("season")
            section("Against his own record",
                    "Points per game on the current scoring, so a season cut short by injury does not "
                    "read as a decline. The red bar is the projection.")
            season_bars(both, "season", "points_per_game", mark=str(v.season), digits=2)


def scenario_banner(v: View) -> None:
    """On every page: whether what you are looking at is the engine's answer or yours.

    Silent on the baseline, because a banner that is always there is not read. Otherwise it names the
    scenario and how far it has moved the board, so no page can show edited numbers without saying so.
    """
    sc = v.scenario
    if sc.is_baseline:
        return
    moved = board_diff(v)
    parts = [f"**{live().name}** · `{sc.digest}`"]
    if sc.items:
        parts.append(f"{len(sc.items)} edit{'s' if len(sc.items) != 1 else ''}")
    if sc.league or sc.scoring:
        parts.append(", ".join(f"{k}={val}" for k, val in sorted(sc.league.items()))
                     or f"scoring {sc.scoring}")
    if sc.k_scale != 1.0:
        parts.append(f"k×{sc.k_scale:g}")
    parts.append(f"{moved.height} players moved")
    st.info(" · ".join(parts) + " — every number below includes the edits.", icon="✏️")


# The sidebar lists eight pages by filename, which says where they are and not what they answer. This
# says what each one is for, in the order the work is done -- and the order is deliberate: everything
# that *reads* the projection comes first, narrowing from the board to one man and then widening back
# out to his week and the game inside it; then who is hurt; then the record it is all a claim against,
# the model's own report card included; and only then the two pages that act on it rather than read it
# -- the edits and the exports. A user who never opens the last two still has a complete tool.
# The order is the order the work is done in: the league first, because that is where a projection is
# checked against instinct; then the team you are fixing, then the man inside it, then his week.
PAGE_MAP = (
    ("pages/1_League.py", "🏆", "League",
     "projected records, all 32 teams, and how much of the league you have been through"),
    ("pages/2_Team.py", "🏟️", "Team",
     "the depth chart, the rooms, how the work divides — and what one edit does to all of it"),
    ("pages/3_Player.py", "🧍", "Player",
     "one man: every number behind him, the knob beside each, and him against three others"),
    ("pages/4_Week.py", "🗓️", "Week",
     "a week at a time: the board, one game, the knobs for that afternoon, and the schedule behind it"),
    ("pages/5_Availability.py", "🚑", "Availability", "who is hurt, and how many games each man gets"),
    ("pages/6_History.py", "📚", "History",
     "what happened — the league, a team, a man, and the model scored against all three"),
    ("pages/7_Edits.py", "✏️", "Edits",
     "every edit in one list, the league's own knobs, and two scenarios side by side"),
    ("pages/8_Exports.py", "📤", "Exports", "the board out to CSV, Excel or a sheet"),
)


def page_map() -> None:
    """What each page answers, on the board, because a numbered filename is not a description."""
    with st.expander("Where everything is"):
        columns = st.columns(3)
        for i, (path, icon, name, what) in enumerate(PAGE_MAP):
            with columns[i % 3]:
                try:
                    st.page_link(path, label=f"**{name}**", icon=icon)
                except Exception:              # noqa: BLE001
                    # no page set to resolve against when a page is run on its own, as the smoke
                    # harness does; the description is the part worth keeping
                    st.markdown(f"{icon} **{name}**")
                st.caption(what)


# The Team page's own team selectbox key, held here so any page can aim it. A league table is only worth
# reading if the row you found something wrong in is one click from the place it gets fixed, and Streamlit
# pages are separate scripts sharing one session state -- so "work on this team" is a written key plus a
# page switch, not an argument passed anywhere.
FOCUS_TEAM = "team:pick"


def focus_team(code: str) -> None:
    _state()[FOCUS_TEAM] = code


def page_link(path: str, label_text: str, icon: str = "") -> None:
    """`st.page_link` that degrades to a caption when there is no page set to resolve against.

    Which is every test run: `AppTest` runs one page as the whole app, so a link to a sibling page is a
    `KeyError` rather than a link, and a pointer between pages is not worth failing a page over.
    """
    try:
        st.page_link(path, label=label_text, icon=icon or None)
    except Exception:                     # noqa: BLE001
        st.caption(label_text)


def work_on_team(code: str, key: str, label_text: str = "Work on this team →") -> None:
    """A button that opens the Team page already on `code`."""
    if st.button(label_text, key=key, width="stretch"):
        focus_team(code)
        try:
            st.switch_page("pages/2_Team.py")
        except Exception:                     # noqa: BLE001 -- no page set to switch within, in tests
            st.caption(f"Open Team from the sidebar — it will open on {code}.")


def team_nav(v: View, teams: Sequence[str], picked: str, key: str = "team:nav") -> None:
    """◀ ▶ across the league, plus a jump to the next team nobody has touched.

    A pass over the league is thirty-two sittings in a row, and reaching back up to a dropdown between
    each one is the part that makes it feel like thirty-two. The buttons write the Team page's own
    selectbox key from a callback, which runs before the widget is built on the next rerun -- so the
    dropdown moves with them rather than fighting them.

    "Next untouched" is the one that matters on the second sitting: it walks the same order and stops at
    the first team carrying no edits, which is the same definition of *done* the league page's coverage
    tab reads, so the two can never disagree about where you are.
    """
    order = list(teams)
    if picked not in order:
        return
    i = order.index(picked)
    done = coverage(v)
    reviewed = set(done.filter(pl.col("reviewed"))["team"].to_list())
    ahead = [t for t in [*order[i + 1:], *order[:i]] if t not in reviewed]

    def go(code: str) -> None:
        _state()[FOCUS_TEAM] = code

    cols = st.columns([1, 1, 2, 5], vertical_alignment="center")
    cols[0].button("◀ prev", key=f"{key}:prev", width="stretch", on_click=go,
                   args=(order[(i - 1) % len(order)],), help=f"{order[(i - 1) % len(order)]}")
    cols[1].button("next ▶", key=f"{key}:next", width="stretch", on_click=go,
                   args=(order[(i + 1) % len(order)],), help=f"{order[(i + 1) % len(order)]}")
    cols[2].button(f"next untouched{f' · {ahead[0]}' if ahead else ''}", key=f"{key}:fresh",
                   width="stretch", disabled=not ahead, on_click=go, args=(ahead[0] if ahead else picked,),
                   help="the next team in this order that carries no edits at all")
    cols[3].caption(
        f"team {i + 1} of {len(order)} · {len(reviewed)} of {len(order)} carry edits · this one is "
        f"**{'done' if picked in reviewed else 'not touched yet'}**"
    )


# --------------------------------------------------------------------------- #
# availability, which is the one term the projection states rather than fits
# --------------------------------------------------------------------------- #
# Every other input is estimated from something the player did. Availability is partly asserted: a
# roster status is a fact about a list, and the factor attached to it is a judgement. So the readers
# here put the assertion, its population and the injury record on one surface -- because the honest
# form of "we assume a PUP player is 55% available" is to say it beside the eleven men it is deciding.
FULL_SEASON = float(REG_WEEKS - 1)          # 17 games in an 18-week calendar

STATUS_MEANING = {
    "ACT": "active", "DEV": "practice squad, promotable", "E14": "exempt, expected back",
    "PUP": "physically unable to perform", "NON": "non-football injury", "SUS": "suspended",
    "RES": "injured reserve", "CUT": "released", "RET": "retired", "EXE": "exempt list",
    "TRC": "reserve, traded",
}


@st.cache_data(show_spinner="reading the injury report")
def injuries(seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """One row per player-season on the league's own injury report, regular season only."""
    return history.injury_report(seasons)


def injury_record(player_ids: tuple[str, ...] = (),
                  seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """How much of the last few seasons a man has spent on the report, and with what.

    The record an availability override is argued from. `weeks_out` is the count that matters -- a Friday
    designation of Out is the league saying he will not play, which is as close to observed missed time
    as a report gets -- and `weeks_listed` includes every Questionable, so the two are kept apart rather
    than added. `report` is the per-season count as a line, because four ones and a nine are the same
    total and a completely different player.
    """
    rep = injuries(seasons)
    if rep.is_empty():
        return pl.DataFrame(schema={"player_id": pl.String})
    if player_ids:
        rep = rep.filter(pl.col("player_id").is_in(list(player_ids)))
    if rep.is_empty():
        return pl.DataFrame(schema={"player_id": pl.String})
    return rep.sort(["player_id", "season"]).group_by("player_id").agg(
        pl.len().alias("seasons_hurt"),
        pl.col("weeks_out").sum().alias("weeks_out"),
        pl.col("weeks_out").max().alias("worst_season_out"),
        pl.col("weeks_listed").sum().alias("weeks_listed"),
        pl.col("weeks_dnp").sum().alias("weeks_dnp"),
        pl.col("weeks_out").alias("report"),
        pl.col("main_injury").drop_nulls().last().alias("main_injury"),
    )


def status_assumptions(v: View) -> pl.DataFrame:
    """The stated availability factor per roster status, and how many players it is deciding.

    A factor of 0.30 on a status nobody holds is a footnote; the same factor on eleven starters is the
    projection. The population is the difference, so it is in the table rather than in a caption.
    """
    counts = participation(v).group_by("status").agg(pl.len().alias("players"))
    rows = [{"status": s, "means": STATUS_MEANING.get(s, s), "availability": float(f)}
            for s, f in v.settings.status_availability.items()]
    return (
        pl.DataFrame(rows, schema={"status": pl.String, "means": pl.String,
                                   "availability": pl.Float64})
        .join(counts, on="status", how="left")
        .with_columns(pl.col("players").fill_null(0))
        .sort(["players", "availability"], descending=[True, True])
    )


def availability_board(v: View, seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """Every player, what the projection has him there for, and the record behind it.

    One frame rather than one per section, because the availability question gets asked three ways on one
    page -- who is not active, who the model has missing time, who you have already edited -- and those
    are the same rows filtered differently. `review` says which of those a man is in, so the page can put
    the twenty players worth an argument above the nine hundred that are not.
    """
    part = participation(v)
    keep = [c for c in ("player_id", "player", "position", "team", "depth_slot", "avail_slot",
                        "status", "status_factor", "presence", "expected_games", "active_weeks",
                        "games_if_available", "obs_games", "prior_games", "n_seasons", "is_rookie",
                        "charted", "age", "years_exp") if c in part.columns]
    men = part.select(keep)
    b = board(v)
    men = men.join(
        b.select([c for c in ("player_id", "games", "fantasy_points", "points_per_game",
                              "position_rank", "startable") if c in b.columns]),
        on="player_id", how="left",
    ).join(injury_record(tuple(men["player_id"].to_list()), seasons), on="player_id", how="left")
    men = with_evidence(v, men, GAMES_METRIC, seasons=seasons)
    men = with_edits(men, "player", "player_id")
    return men.with_columns(
        pl.when(pl.col("status") != "ACT")
        .then(pl.concat_str([pl.lit("not active — "), pl.col("status")]))
        .when(pl.col("edited")).then(pl.lit("you have edited him"))
        .when((pl.col("depth_slot") == 1) & (pl.col("expected_games") < FULL_SEASON - 2.0))
        .then(pl.lit("a starter the model has missing time"))
        .when((pl.col("n_seasons") >= 2) & (pl.col("expected_games") < pl.col("obs_games") - 3.0))
        .then(pl.lit("projected under his own record"))
        .otherwise(pl.lit("")).alias("review"),
    ).sort(["expected_games", "fantasy_points"], descending=[False, True], nulls_last=True)


def team_weeks(v: View, team: str) -> list[int]:
    """The weeks this team actually plays. Not 1 to 17: a bye is a missing week, not a zero."""
    env = environment(v).filter(pl.col("team") == team)
    return sorted(int(w) for w in env["week"].unique().to_list())


def games_from_week(v: View, team: str, week: int) -> int:
    """How many games are left if he is back for `week` -- what a return date is worth as a count.

    The engine's availability is a season count spread evenly over the weeks, with no per-week hook to
    hang a return date on (`expected_games` is applied to a frame that has no weeks in it). So a return
    date is translated here into the number of games it implies, off this team's own schedule, which is
    what makes the bye fall in the right place.
    """
    return sum(1 for w in team_weeks(v, team) if w >= int(week))


def set_games(player_id: str, games: float, base: float | None, note_text: str = "") -> None:
    """Write an availability override, with the engine's own number as its base."""
    edit(Override("player", player_id, GAMES_METRIC, "set", float(games), base=base,
                  note=note_text))


# --------------------------------------------------------------------------- #
# the season week by week
# --------------------------------------------------------------------------- #
# The engine is per-week all the way through -- `compose.weekly` is 17 rows a player and the season is
# their sum -- so a page that only ever shows the sum is hiding half of what was computed. A bye is the
# clearest case: two men with the same total are a different lineup problem if one of them is away in
# week 12 and the other in week 6.
WEEK_CONTEXT = ("opponent", "is_home", "p_play", "spread", "total", "implied_points", "has_market")


def weeks(v: View) -> list[int]:
    """Every week the projection covers."""
    return sorted(int(w) for w in weekly(v)["week"].unique().to_list())


def byes(v: View, week: int) -> list[str]:
    """The teams not playing that week. Absent rows rather than zero rows, so it is a set difference."""
    playing = set(weekly(v).filter(pl.col("week") == int(week))["team"].unique().to_list())
    return sorted(set(board(v)["team"].unique().to_list()) - playing)


def week_board(v: View, week: int) -> pl.DataFrame:
    """One week, everybody: the matchup, the chance he plays, and the line he is projected for.

    `week_rank` is ranked inside the position on that week alone, which is the number a start/sit
    argument is actually about -- a season rank cannot say that a WR2 has the best matchup on the board.
    """
    wk = weekly(v).filter(pl.col("week") == int(week))
    if wk.is_empty():
        return wk
    want = ["player_id", "player", "position", "team", "depth_slot", "status", *WEEK_CONTEXT,
            *[c for c in ALL_STATS if c in wk.columns], "fantasy_points"]
    out = wk.select([c for c in dict.fromkeys(want) if c in wk.columns]).with_columns(
        pl.col("fantasy_points").rank("min", descending=True).over("position").cast(pl.Int32)
        .alias("week_rank"),
    )
    return with_edits(out, "player", "player_id").sort("fantasy_points", descending=True)


def weekly_matrix(v: View, player_ids: tuple[str, ...] = (),
                  value: str = "fantasy_points") -> pl.DataFrame:
    """A player per row, a week per column: the board as a season rather than as a total.

    The bye is the point of it. A total says a receiver is worth 210 points; the row says which fortnight
    he is not there for. The week columns keep their nulls -- a bye is not a zero-point game -- and `path`
    fills them only because a line cannot be drawn through a gap.
    """
    wk = weekly(v)
    if player_ids:
        wk = wk.filter(pl.col("player_id").is_in(list(player_ids)))
    if wk.is_empty() or value not in wk.columns:
        return pl.DataFrame(schema={"player_id": pl.String})
    wide = wk.pivot(on="week", index=["player_id", "player", "position", "team"], values=value)
    cols = [str(w) for w in weeks(v) if str(w) in wide.columns]
    filled = [pl.col(c).fill_null(0.0) for c in cols]
    return wide.select("player_id", "player", "position", "team", *cols).with_columns(
        pl.concat_list(filled).alias("path"),
        pl.sum_horizontal(filled).alias("season_points"),
    ).sort("season_points", descending=True)


# --------------------------------------------------------------------------- #
# one game
# --------------------------------------------------------------------------- #
# The weekly board is a league-wide slice of the same frame; this is the other cut, and the one a
# projection is usually argued about in: two offences, the score each is projected for, and the two box
# scores that produced it. Nothing here is a new model. The score is the environment's `implied_points`,
# the box score is `compose.weekly` filtered to a `game_id` and summed, and the check underneath is
# `compose.team_check` narrowed to one game -- because a game view that computed its own total would be
# a second projection of the same afternoon, and the two would disagree by Thursday.
#
# `spread`, `total` and `implied_points` are blended against the market independently, so a side's
# points and the line will not reconcile to the last tenth. They are shown as three numbers rather than
# forced into one, since making them agree here would mean inventing a fourth.
GAME_OUTCOME = ("implied_points", "spread", "total", "has_market", "market_points", "model_points",
                "plays", "dropbacks", "seconds_per_play", "red_zone_trips", "pass_tds", "rush_tds",
                "yards_per_attempt", "yards_per_carry", "success_rate", "rest_days", "div_game",
                "roof", "surface", "temp", "wind")

# A team's half of the box score: the players' own stats, added up. The only place a team total on this
# surface may come from, so the line on screen is the line the roster projects.
TEAM_BOX = ("dropbacks", "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
            "sacks", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
            "receiving_yards", "receiving_tds", "fumbles_lost", "fantasy_points")

# player stat -> the environment column it was divided out of. `compose.RECONCILE` names the
# `team_`-prefixed columns of the joined frame; the environment carries them unprefixed. The two yardage
# rows are dropped rather than derived: the team layer projects yards as a count times a rate and does
# not force the roster to match it, so a gap there is a disagreement worth having rather than a broken
# pool, and History's ⚖️ Share sums tab is where it is reported.
GAME_RECONCILE = {stat: col.removeprefix("team_")
                  for stat, col in compose.RECONCILE.items()
                  if col not in compose.DERIVED_TEAM}

GAME_SHEET_ALWAYS = tuple(overrides.GAME_ONLY_FIELDS)      # `p_play` -- the one knob only a game has


def _sided(env: pl.DataFrame) -> pl.DataFrame:
    """Tag each of a game's two rows with a side, home first.

    By position within the game rather than by `is_home` alone: a neutral-site game can have nobody at
    home, and a pair of rows still has to come out in a stable order for the join below to mean
    anything.
    """
    return (
        env.sort(["game_id", "is_home", "team"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("game_id").alias("side"))
    )


def game_index(v: View, week: int | None = None) -> pl.DataFrame:
    """Every game as one row: who is playing, what each side scores, and the line.

    The fixture picker on the Week page, and readable enough to be the schedule itself. `game` is the
    fixture as it would be written -- away at home, `vs` when neither side is really at home.
    """
    env = environment(v)
    if week is not None:
        env = env.filter(pl.col("week") == int(week))
    if env.is_empty():
        return pl.DataFrame(schema={"game_id": pl.String, "game": pl.String})
    sides = _sided(env)
    neutral = (pl.col("neutral_site") if "neutral_site" in sides.columns
               else pl.lit(False)).fill_null(False).alias("neutral")
    home = sides.filter(pl.col("side") == 0).select(
        "game_id", *[c for c in ("week", "gameday") if c in sides.columns],
        pl.col("team").alias("home"),
        pl.col("implied_points").alias("home_points"),
        "spread", "total", "has_market", neutral,
    )
    away = sides.filter(pl.col("side") == 1).select(
        "game_id", pl.col("team").alias("away"), pl.col("implied_points").alias("away_points"),
    )
    out = home.join(away, on="game_id", how="inner").with_columns(
        pl.concat_str([
            pl.col("away"),
            pl.when(pl.col("neutral")).then(pl.lit(" vs ")).otherwise(pl.lit(" @ ")),
            pl.col("home"),
        ]).alias("game"),
        # the spread is the home side's, so a negative number is the home team favoured; naming the
        # favourite saves every reader having to remember which way round that is
        pl.when(pl.col("spread") <= 0.0).then(pl.col("home")).otherwise(pl.col("away"))
        .alias("favourite"),
    )
    order = ["game_id", "game", "week", "gameday", "away", "home", "away_points", "home_points",
             "total", "spread", "favourite", "has_market"]
    return out.select([c for c in order if c in out.columns]).sort(
        [c for c in ("week", "game_id") if c in out.columns])


def game_sides(v: View, game_id: str) -> pl.DataFrame:
    """One game's two environment rows, away first, which is the order a fixture is written in."""
    env = environment(v).filter(pl.col("game_id") == str(game_id))
    if env.is_empty():
        return env
    keep = ["side", "team", "opponent", "is_home", *[c for c in GAME_OUTCOME if c in env.columns]]
    return _sided(env).select(keep).sort("side", descending=True)


def game_teams(v: View, game_id: str) -> list[str]:
    """The two teams in a game, away first."""
    return game_sides(v, game_id)["team"].to_list()


def team_box(v: View, game_id: str) -> pl.DataFrame:
    """The box score by side: every man's projected stats added up, with the game's own numbers beside.

    A sum over the roster, not a second projection -- which is exactly why it is worth showing next to
    `implied_points`. The players are what the score is made of, so if the two sides of this table do
    not look like the same afternoon, one of the edits above is the reason.
    """
    wk = weekly(v).filter(pl.col("game_id") == str(game_id))
    if wk.is_empty():
        return wk
    have = [c for c in TEAM_BOX if c in wk.columns]
    box = wk.group_by("team").agg([pl.col(c).sum().alias(c) for c in have])
    env = game_sides(v, game_id)
    keep = ["side", "team", "opponent", "is_home",
            *[c for c in ("implied_points", "spread", "total", "has_market") if c in env.columns]]
    return env.select(keep).join(box, on="team", how="left").sort("side", descending=True).drop("side")


def player_box(v: View, game_id: str, team: str | None = None,
               positions: Sequence[str] = ()) -> pl.DataFrame:
    """One game, one team, a man a row: the stat line the projection has for him in that game.

    A weekly line is an expectation including availability, so `p_play` is beside it rather than applied
    and forgotten -- a back at 60% to play is showing 60% of a line, and that is a different number from
    a back who is certain to play and split the work.
    """
    wk = weekly(v).filter(pl.col("game_id") == str(game_id))
    if team:
        wk = wk.filter(pl.col("team") == team)
    if positions:
        wk = wk.filter(pl.col("position").is_in(list(positions)))
    if wk.is_empty():
        return wk
    want = ["player_id", "player", "position", "team", "depth_slot", "status", "p_play",
            *stat_columns(list(positions)), "fantasy_points"]
    out = wk.select([c for c in dict.fromkeys(want) if c in wk.columns])
    return with_edits(out, "player", "player_id").sort("fantasy_points", descending=True,
                                                       nulls_last=True)


def game_reconcile(v: View, game_id: str) -> pl.DataFrame:
    """Do the two box scores still add up to the two offences they were divided out of?

    `compose.team_check` asks this over the whole season; this is the same question about the game on
    screen, and it is on the page because per-game editing is the one thing that could break it. A
    counted pool -- targets, carries, dropbacks -- must match to rounding whatever has been typed above,
    because an edit lands before the division rather than after it. A row that does not match is a bug
    report, not a modelling opinion.
    """
    wk = weekly(v).filter(pl.col("game_id") == str(game_id))
    env = environment(v).filter(pl.col("game_id") == str(game_id))
    if wk.is_empty() or env.is_empty():
        return pl.DataFrame(schema={"stat": pl.String})
    have = {s: c for s, c in GAME_RECONCILE.items() if s in wk.columns and c in env.columns}
    players = wk.group_by("team").agg([pl.col(s).sum().alias(s) for s in have])
    joined = env.select("team", *dict.fromkeys(have.values())).join(players, on="team", how="inner")
    rows = []
    for team_row in joined.rows(named=True):
        for stat, col in have.items():
            rows.append({
                "team": team_row["team"], "stat": stat, "from_the_players": team_row[stat],
                "from_the_offence": team_row[col],
                "gap": team_row[stat] - team_row[col],
                "from_a_pool": stat in {p.name for p in opportunity.POOLS if p.exclusive},
            })
    return pl.DataFrame(rows).sort(["team", "stat"])


def game_fields(column_set: str, positions: Sequence[str] = ()) -> tuple[str, ...]:
    """Which columns a chosen set means on a game sheet: the roster sheet's sets, minus the season.

    The same named groups as the roster sheet, so a reader who knows where target share lives finds it
    in the same place -- but filtered to what a single game can actually carry. The slot and the games
    played are one number for the whole season and are dropped here rather than shown locked: a cell
    that cannot mean anything in this context is worse than an absent one. `p_play` replaces them, and
    is the knob this surface exists for -- the chance he plays *this* week.
    """
    chosen = [f for f in sheet_fields(column_set, positions) if f in overrides.GAME_ALL_FIELDS]
    return tuple(dict.fromkeys(GAME_SHEET_ALWAYS + tuple(chosen)))


def _game_inputs(run: overrides.Run, game_id: str, team: str, fields: Sequence[str],
                 positions: Sequence[str] = ()) -> pl.DataFrame:
    """One team-game's editable inputs, each read from the frame the engine edits it on.

    Two frames, because the engine has two per-game seams and they are not interchangeable: `opp`
    carries availability and the shares as they were *before* the pool was divided, and `adj` carries
    the rates as they were before the stat line was made of them. Reading a share off the wrong one
    would show the reader a number the projection did not divide.
    """
    opp, adj = run.opp, run.adj
    where = (pl.col("game_id") == str(game_id)) & (pl.col("team") == team)
    rows = opp.filter(where)
    if positions:
        rows = rows.filter(pl.col("position").is_in(list(positions)))
    if rows.is_empty():
        return rows
    ident = [c for c in ("player_id", "player", "position", "team", "depth_slot", "status", "week")
             if c in rows.columns]
    out = rows.select(*ident, *[f for f in fields if f in rows.columns])
    rest = [f for f in fields if f not in out.columns and f in adj.columns]
    if rest:
        out = out.join(adj.filter(where).select("player_id", *rest), on="player_id", how="left")
    return out


def game_sheet(v: View, game_id: str, team: str, fields: Sequence[str],
               positions: Sequence[str] = ()) -> pl.DataFrame:
    """A team's half of one game as an editable table: what he is projected for, then the inputs.

    The projected line sits between the identity and the knobs for the same reason it does on the
    roster sheet -- an edit is made to move one of those numbers, so it belongs in the same row as the
    number being moved rather than on the page you go to afterwards.
    """
    out = _game_inputs(projection(v), game_id, team, fields, positions)
    if out.is_empty():
        return out
    wk = weekly(v).filter((pl.col("game_id") == str(game_id)) & (pl.col("team") == team))
    outputs = [c for c in ("fantasy_points", "targets", "carries", "attempts") if c in wk.columns]
    if outputs:
        out = out.join(wk.select("player_id", *outputs), on="player_id", how="left")
    empty = [f for f in fields
             if f in out.columns and f not in GAME_SHEET_ALWAYS and out[f].null_count() == out.height]
    out = with_edits(out.drop(empty), "player", "player_id").drop("edits")
    out = out.with_columns(pl.col("position")
                           .replace_strict({p: i for i, p in enumerate(POSITIONS)}, default=99,
                                           return_dtype=pl.Int32).alias("_pos"))
    out = out.sort([c for c in ("_pos", overrides.DEPTH_FIELD) if c in out.columns],
                   nulls_last=True).drop("_pos")
    shown = ["player_id", "edited", *[c for c in ("player", "position", "depth_slot", "status")
                                      if c in out.columns], *outputs,
             *[f for f in fields if f in out.columns],
             *[c for c in ("week",) if c in out.columns]]
    return out.select(list(dict.fromkeys(shown)))


def game_sheet_baseline(v: View, game_id: str, team: str, fields: Sequence[str],
                        positions: Sequence[str] = ()) -> pl.DataFrame:
    """The same inputs with no edits at all, which is what a new per-game edit records as its base.

    It matters more here than on the roster sheet. The number on screen in a game where a starter has
    been ruled out is the *consequence* of that edit -- the backup's share went up because the pool was
    divided without the starter -- and recording that as "what the engine said" would make the second
    edit in a game undroppable: reset it and it would go back to the first edit's answer.
    """
    return _game_inputs(baseline(v), game_id, team, fields, positions)


def game_env_fields(v: View) -> tuple[str, ...]:
    """The team-level numbers a game has, editable one game at a time."""
    env = environment(v)
    return tuple(c for c in overrides.TEAM_FIELDS if c in env.columns)


def game_env_sheet(v: View, game_id: str, fields: Sequence[str]) -> pl.DataFrame:
    """Both offences in one game as two editable rows: how much there is to divide, before who takes it.

    The other half of a per-game adjustment, and the half to reach for first. Moving a team's dropbacks
    for one week changes every share-holder's count in proportion and leaves the division alone;
    moving a man's share moves the work between teammates and leaves the team where it was.
    """
    env = environment(v).filter(pl.col("game_id") == str(game_id))
    if env.is_empty():
        return env
    keep = [c for c in fields if c in env.columns]
    return _sided(env).select("side", "team", "week", "opponent", *keep).sort(
        "side", descending=True).drop("side")


def game_season_effect(v: View, game_id: str) -> pl.DataFrame:
    """What the edits have done to the season totals of the men in this game.

    The point of editing a game rather than a season is that the season follows, and this is where that
    is shown rather than asserted: the same board diff every other page reads, narrowed to the players
    on screen. A per-game edit that moves a season total by a tenth is a per-game edit, and one that
    moves it by forty is a claim about the whole year.
    """
    wk = weekly(v).filter(pl.col("game_id") == str(game_id))
    diff = board_diff(v)
    if wk.is_empty() or diff.is_empty():
        return pl.DataFrame(schema={"player_id": pl.String})
    return diff.filter(pl.col("player_id").is_in(wk["player_id"].unique().to_list()))


# --------------------------------------------------------------------------- #
# many at once
# --------------------------------------------------------------------------- #
# The knobs edit one number and the grids edit one room. This is the third scale, and the one a whole
# board needs: "every quarterback at slot 1 plays 16.5 games", "every rookie receiver at nine tenths of
# his share". It is still one `Override` per player -- there is no bulk object in the scenario, because a
# bulk edit somebody cannot drop one row of is a bulk edit they will clear entirely.
BULK_FIELDS = tuple(f for f in overrides.PLAYER_FIELDS)


def bulk_population(v: View, positions: list[str] | tuple[str, ...] = (),
                    teams: list[str] | tuple[str, ...] = (), slots: tuple[int, int] = (1, 12),
                    statuses: list[str] | tuple[str, ...] = (), rookies_only: bool = False,
                    search: str = "") -> pl.DataFrame:
    """Who a bulk edit would be aimed at: the filter, applied to the roster the projection ran on."""
    part = participation(v)
    out = part.select(
        [c for c in ("player_id", "player", "position", "team", "depth_slot", "status", "is_rookie",
                     "years_exp", "expected_games") if c in part.columns]
    )
    if positions:
        out = out.filter(pl.col("position").is_in(list(positions)))
    if teams:
        out = out.filter(pl.col("team").is_in(list(teams)))
    if statuses:
        out = out.filter(pl.col("status").is_in(list(statuses)))
    if rookies_only and "is_rookie" in out.columns:
        out = out.filter(pl.col("is_rookie"))
    lo, hi = int(slots[0]), int(slots[1])
    out = out.filter(pl.col("depth_slot").is_between(lo, hi))
    if search.strip():
        out = out.filter(pl.col("player").str.to_lowercase().str.contains(search.lower().strip()))
    return out.sort(["position", "team", "depth_slot"])


def bulk_preview(v: View, player_ids: tuple[str, ...], field: str, mode: str,
                 value: float) -> pl.DataFrame:
    """What the bulk edit would do, per player, before it is written.

    Only the men the metric exists for: aiming a target-share edit at a room that includes the
    quarterback is a filter mistake, and it should be visible as thirty rows rather than thirty-two
    rather than as two silently discarded overrides. `now` is what the projection is running on and
    `engine` is the estimator's own number, which is what gets recorded as the base.
    """
    vals = metric_values(v, field)
    if vals.is_empty() or not player_ids:
        return pl.DataFrame()
    mine = vals.filter(pl.col("player_id").is_in(list(player_ids)))
    if mine.is_empty():
        return pl.DataFrame()
    now = pl.col("applied") if "applied" in mine.columns else pl.col("estimate")
    after = (pl.lit(float(value)) if mode == "set" else now * float(value))
    return mine.select(
        "player_id",
        *[c for c in ("player", "position", "team", "depth_slot") if c in mine.columns],
        pl.col("estimate").alias("engine"),
        now.alias("now"),
        after.alias("after"),
    ).with_columns((pl.col("after") - pl.col("now")).alias("change")).filter(
        pl.col("now").is_not_null() & (pl.col("change").abs() > 1e-9)
    ).sort("change", descending=True)


def bulk_edits(preview: pl.DataFrame, field: str, mode: str, value: float,
               why: str = "") -> list[Override]:
    """The preview as overrides: one per player, each with its own base and a shared note."""
    if preview.is_empty():
        return []
    return [
        Override("player", r["player_id"], field, mode, float(value),
                 base=None if r.get("engine") is None else float(r["engine"]), note=why)
        for r in preview.rows(named=True)
    ]


# --------------------------------------------------------------------------- #
# two to four players, side by side
# --------------------------------------------------------------------------- #
# A board is read as an order, but a decision is made between two or three men at a time, and the
# comparison a total cannot settle is *why*: same points, one of them on volume and one on efficiency,
# one of them with a record and one with a prior. So a comparison shows the stat line, then the
# estimates the line came from with their sample sizes, then what each man has actually done.
def display_names(v: View, player_ids: tuple[str, ...]) -> dict[str, str]:
    """Player id to a column heading, disambiguated by team only where two names collide."""
    b = board(v).filter(pl.col("player_id").is_in(list(player_ids)))
    counts: dict[str, int] = {}
    for name in b["player"].to_list():
        counts[name] = counts.get(name, 0) + 1
    return {r["player_id"]: (r["player"] if counts.get(r["player"], 0) < 2
                             else f"{r['player']} ({r['team']})")
            for r in b.rows(named=True)}


def compare_lines(v: View, player_ids: tuple[str, ...]) -> pl.DataFrame:
    """The board rows for the men being compared, with the stat line for the positions on screen."""
    b = board(v)
    mine = b.filter(pl.col("player_id").is_in(list(player_ids)))
    if mine.is_empty():
        return mine
    positions = mine["position"].unique().to_list()
    front = [c for c in ("player", "position", "team", "depth_slot", "status", "position_rank",
                         "tier", "games", "fantasy_points", "points_per_game", "vs_starter",
                         "drop_next", "delta_points") if c in mine.columns]
    stats = [c for c in stat_columns(positions) if c in mine.columns and c not in front]
    return mine.select(*front, *stats)


def compare_ratings(v: View, player_ids: tuple[str, ...]) -> pl.DataFrame:
    """Every estimate the men have in common, side by side, with the sample size under each.

    One row per metric and one column per player, because that is the shape the question has: not "what
    is his target share" but "which of these two does the model believe about, and on how much".
    """
    names = display_names(v, player_ids)
    out: pl.DataFrame | None = None
    for pid in player_ids:
        rated = ratings(v, pid)
        if rated.is_empty() or "used" not in rated.columns:
            continue
        who = names.get(pid, pid)
        mine = rated.select(
            "metric",
            *[c for c in ("kind", "units") if c in rated.columns],
            pl.col("used").alias(who),
            pl.col("n").alias(f"n · {who}"),
        ).unique(subset="metric", keep="first")
        out = mine if out is None else out.join(
            mine.select([c for c in mine.columns if c not in ("kind", "units")]),
            on="metric", how="full", coalesce=True,
        )
    if out is None:
        return pl.DataFrame()
    order = ["kind", "metric", "units"]
    return out.select([c for c in order if c in out.columns]
                      + [c for c in out.columns if c not in order]).sort(
        [c for c in ("kind", "metric") if c in out.columns])


def compare_weeks(v: View, player_ids: tuple[str, ...]) -> pl.DataFrame:
    """A row per week and a column per player. A gap is a bye, which is why it is left as a null."""
    names = display_names(v, player_ids)
    wk = weekly(v).filter(pl.col("player_id").is_in(list(player_ids)))
    if wk.is_empty():
        return pl.DataFrame()
    wide = wk.select("week", "player_id", "fantasy_points").pivot(
        on="player_id", index="week", values="fantasy_points").sort("week")
    return wide.rename({c: names[c] for c in wide.columns if c in names})


def compare_history(v: View, player_ids: tuple[str, ...],
                    seasons: tuple[int, ...] = HISTORY_VIEW) -> pl.DataFrame:
    """What each of them has actually done, per season, on the scoring the projection is using.

    Points per game rather than totals, and the projected season on the end of the same table: a season
    cut short by injury is not a decline in ability, and the comparison is being made against next year.
    """
    names = display_names(v, player_ids)
    hist = actuals(seasons, v.scoring).filter(pl.col("player_id").is_in(list(player_ids)))
    rows = pl.DataFrame(schema={"season": pl.Int32, "player_id": pl.String, "value": pl.Float64})
    if not hist.is_empty():
        rows = hist.group_by(["player_id", "season"]).agg(
            pl.col("fantasy_points").sum().alias("points"), pl.col("games").sum().alias("games"),
        ).select(
            pl.col("season").cast(pl.Int32), "player_id",
            (pl.col("points") / pl.max_horizontal(pl.col("games"), pl.lit(1))).alias("value"),
        )
    b = board(v).filter(pl.col("player_id").is_in(list(player_ids)))
    projected = b.select(pl.lit(v.season, pl.Int32).alias("season"), "player_id",
                         pl.col("points_per_game").alias("value"))
    both = pl.concat([rows, projected], how="vertical_relaxed")
    if both.is_empty():
        return pl.DataFrame()
    wide = both.pivot(on="player_id", index="season", values="value").sort("season")
    return wide.rename({c: names[c] for c in wide.columns if c in names})


# --------------------------------------------------------------------------- #
# one number, and everything it moves
# --------------------------------------------------------------------------- #
# Editing a rating is typing into a cell, and a cell says nothing about what it did. The engine's own
# shape is the answer: a share is a slice of a pool that sums to one, so a target given to one man is
# taken from his room and the team's total does not move at all. None of that is visible in a grid of
# shares, and all of it is visible in a before and after.
#
# Three parts, in the order the surface reads them: who is producing this offence (`contributions`,
# `pool_facts`), which numbers are worth reaching for at all (`KNOBS`, `knob_kind`, `KNOB_HELP`), and
# then one candidate edit run through the whole engine *before* it is committed (`ripple`).

# The pools somebody argues about, in the order they matter, named as a reader would say them. Every
# pool the engine divides is in `opportunity.POOLS`; one or two of those (`designed_qb_rushes`) are
# internal plumbing rather than anything a player claims, so this is a shortlist and not that tuple.
CONTRIBUTION_POOLS = (
    ("targets", "Targets"),
    ("air_yards", "Air yards"),
    ("carries", "Carries"),
    ("dropbacks", "Dropbacks"),
    ("rz_targets", "Red-zone targets"),
    ("rz_carries", "Red-zone carries"),
    ("inside_5_carries", "Carries inside the five"),
    ("short_yardage_carries", "Carries in short yardage"),
    ("late_down_targets", "Third and fourth-down targets"),
    ("receiving_tds", "Receiving touchdowns"),
    ("rushing_tds", "Rushing touchdowns"),
    ("passing_tds", "Passing touchdowns"),
    ("offense_snaps", "Snaps"),
    ("routes", "Routes run"),
    ("rush_plays", "Runs he is on the field for"),
    ("designed_qb_rushes", "Designed quarterback runs"),
)
POOL_NAME = dict(CONTRIBUTION_POOLS)


def pool_for_field(field: str) -> str | None:
    """The exclusive pool a share metric divides, if it divides one -- which is what makes it zero-sum.

    `dropback_share` is both a participation metric and a claim on an exclusive pool, and the exclusive
    reading is the one that matters: a dropback goes to exactly one quarterback.
    """
    for p in opportunity.POOLS:
        if field in p.shares and p.exclusive:
            return p.name
    return None


def shared_pool_for_field(field: str) -> str | None:
    """The non-exclusive pool a participation metric measures. Five men are on the field for one snap."""
    for p in opportunity.POOLS:
        if field in p.shares and not p.exclusive:
            return p.name
    return None


KIND_NOTE = {
    "availability": "how many of the seventeen he is there for — every count he has scales with it",
    "pool": "his slice of a pool that sums to one — zero-sum, so his room pays for what he gains",
    "presence": "how much of the offence he is on the field for — not divided, so nobody else moves",
    "rate": "what he does with what he gets — his own number, and nobody else's line rides on it",
}
KIND_ICON = {"availability": "📅", "pool": "⚖️", "presence": "🕒", "rate": "🎯"}


def knob_kind(field: str) -> str:
    """Which of the four kinds of number this is, which is what decides who else moves when it does."""
    if field in overrides.AVAILABILITY_FIELDS:
        return "availability"
    if pool_for_field(field):
        return "pool"
    if shared_pool_for_field(field):
        return "presence"
    return "rate"


# What each knob actually does, in a sentence. A reader who knows what `tprr` stands for still does not
# know whether editing it moves a teammate, and that is the thing the page has to say.
KNOB_HELP = {
    "expected_games": "How many of the seventeen he plays. The fastest way to run a holdout or an "
                      "injury, and the biggest single lever on a season total.",
    "snap_share": "The share of the offence's snaps he is on the field for. Snaps are shared, so "
                  "raising it takes nothing from anybody.",
    "route_participation": "The share of the team's dropbacks he runs a route on. His targets are this "
                           "times his share of what gets thrown.",
    "rush_participation": "The share of the team's designed runs he is on the field for.",
    "dropback_share": "The share of the team's dropbacks he takes. One quarterback takes a dropback, so "
                      "the room is filled in depth order: the starter gets his claim and the man behind "
                      "him gets what is left.",
    "designed_rush_share": "How much of the team's designed running the quarterback does himself.",
    "target_share": "His slice of the team's targets. Zero-sum: what he gains, his receiving room loses.",
    "air_yards_share": "His slice of the ball in the air — where the deep targets go, and most of what "
                       "separates a possession receiver from a number one.",
    "rz_target_share": "His slice of the targets inside the twenty, which is where receiving "
                       "touchdowns come from.",
    "late_down_target_share": "His slice of the third and fourth-down targets — the trusted-hands share.",
    "rec_td_share": "His slice of the team's touchdown passes.",
    "carry_share": "His slice of the team's carries. Zero-sum inside the backfield.",
    "clean_rush_share": "His slice of the carries that are not scrambles or sacks.",
    "rz_carry_share": "His slice of the carries inside the twenty.",
    "inside_5_carry_share": "His slice of the carries inside the five — the goal-line back's number.",
    "short_yardage_carry_share": "His slice of the short-yardage carries.",
    "rush_td_share": "His slice of the team's rushing touchdowns.",
    "qb_rush_td_share": "The quarterback's slice of the team's rushing touchdowns.",
    "pass_td_share": "His slice of the team's touchdown passes as the thrower.",
    "catch_rate": "The share of his targets he catches. His own number.",
    "yards_per_target": "Yards per target — volume times this is his receiving yards.",
    "adot": "How far downfield his average target goes.",
    "tprr": "Targets per route run: how often he is thrown to when he is out there.",
    "yards_per_carry": "Yards per carry. His own number.",
    "rush_success_rate": "How often his carries stay on schedule.",
    "fumble_rate": "How often he puts it on the ground.",
    "attempt_rate": "The share of his dropbacks that become throws rather than sacks or scrambles.",
    "completion_pct": "The share of his attempts he completes.",
    "yards_per_attempt": "Yards per attempt — his volume times this is his passing yards.",
    "pass_td_rate": "Touchdowns per attempt.",
    "int_rate": "Interceptions per attempt.",
    "sack_rate": "The share of his dropbacks that end in a sack.",
    "scramble_rate": "The share of his dropbacks he runs on.",
    "yards_per_clean_rush": "Yards per carry on his designed runs and scrambles.",
    "air_yards_per_attempt": "How far downfield he throws on average.",
    "scramble_ypc": "Yards per scramble.",
    "designed_rush_ypc": "Yards per designed quarterback run.",
    "qb_fumble_rate": "How often he fumbles.",
}

# The knobs worth reaching for first, per position and in the order they move a projection. Every name
# in `overrides.PLAYER_FIELDS` is editable somewhere; a room table showing twelve of them at once makes
# the two that matter exactly as hard to find as the ten that do not, which is the complaint this
# shortlist answers. Volume before efficiency in every list, because volume is what moves a season.
KNOBS = {
    "QB": ("dropback_share", "expected_games", "attempt_rate", "yards_per_attempt", "pass_td_rate",
           "completion_pct", "designed_rush_share", "yards_per_clean_rush", "int_rate", "sack_rate",
           "scramble_rate", "pass_td_share"),
    "RB": ("carry_share", "expected_games", "target_share", "yards_per_carry", "rush_td_share",
           "rz_carry_share", "inside_5_carry_share", "snap_share", "catch_rate", "yards_per_target",
           "rec_td_share", "short_yardage_carry_share"),
    "WR": ("target_share", "expected_games", "yards_per_target", "rec_td_share", "catch_rate",
           "air_yards_share", "route_participation", "rz_target_share", "adot", "tprr",
           "late_down_target_share", "snap_share"),
    "TE": ("target_share", "expected_games", "yards_per_target", "rec_td_share", "catch_rate",
           "route_participation", "air_yards_share", "rz_target_share", "adot", "tprr", "snap_share"),
}


def knobs_for(position: str | None) -> tuple[str, ...]:
    """The shortlist for a position, or every player field for a position without one."""
    return KNOBS.get(position or "", tuple(overrides.PLAYER_FIELDS))


def knob_option(field: str) -> str:
    """A knob as one line in a picker: what it is called, and which kind of number it is."""
    return f"{KIND_ICON.get(knob_kind(field), '·')}  {label(field)}"


# --------------------------------------------------------------------------- #
# the whole roster, one row a man, every rating in it editable
# --------------------------------------------------------------------------- #
# The other two player surfaces are one man at a time and one field across a filtered population.
# This is the third shape, and the one a roster review actually asks for: players down the side, the
# numbers that decide their projection across the top, so a receiving room can be argued about in slot
# order without opening a page per man. Nothing here computes a projection -- it is an assembly of the
# frames the run already produced, so a cell in the sheet is the same number the player page shows,
# and typing over it writes the same override the knob beside it would.
#
# Thirty-nine editable fields do not fit on a screen and would not be read if they did, so the columns
# come in sets. The default set is position-aware -- the same shortlist the What-if tab offers, which
# is ordered by how much a number moves a season -- and the named groups follow the engine's own
# stages, because that is the order a disagreement is actually resolved in: is he playing, how much of
# the offence is he on the field for, what slice does he take, how well does he do with it.
SHEET_ALWAYS = (overrides.DEPTH_FIELD, GAMES_METRIC)

SHEET_GROUPS: dict[str, tuple[str, ...]] = {
    "Depth & availability": ("snap_share", "route_participation", "rush_participation",
                             "dropback_share", "designed_rush_share"),
    "Volume shares": ("target_share", "carry_share", "air_yards_share", "route_participation",
                      "late_down_target_share", "clean_rush_share", "dropback_share"),
    "Scoring shares": ("rz_target_share", "rz_carry_share", "inside_5_carry_share",
                       "short_yardage_carry_share", "rec_td_share", "rush_td_share",
                       "pass_td_share", "qb_rush_td_share"),
    "Catching & running rates": ("catch_rate", "yards_per_target", "adot", "tprr",
                                 "yards_per_carry", "rush_success_rate", "fumble_rate"),
    "Passing rates": ("attempt_rate", "completion_pct", "yards_per_attempt", "air_yards_per_attempt",
                      "pass_td_rate", "int_rate", "sack_rate", "scramble_rate",
                      "yards_per_clean_rush", "designed_rush_ypc", "scramble_ypc", "qb_fumble_rate"),
}
SHEET_KNOBS = "What moves it"                 # position-aware, so not a fixed set of columns
SHEET_EVERYTHING = "Everything editable"
SHEET_COLUMN_SETS = (SHEET_KNOBS, *SHEET_GROUPS, SHEET_EVERYTHING)
SHEET_ORDERS = ("team, position, slot", "position, team, slot", "projected points", "name")


def sheet_fields(column_set: str, positions: Sequence[str] = ()) -> tuple[str, ...]:
    """Which columns a chosen set means, given who is on screen.

    The slot and the games always come first and are always editable: they are the two knobs a roster
    review reaches for before any share, and a table that showed a receiver's target share while making
    him read his availability somewhere else would have missed the point of being one table.
    """
    if column_set == SHEET_EVERYTHING:
        # every player field except the release, which is not a cell in a sheet: it takes a row out
        # rather than putting a number in one, so it lives on the depth chart as its own control
        chosen: tuple[str, ...] = tuple(f for f in overrides.FIELDS["player"]
                                        if f != overrides.ROSTER_FIELD)
    elif column_set in SHEET_GROUPS:
        chosen = SHEET_GROUPS[column_set]
    else:
        picked = [p for p in POSITIONS if p in (positions or POSITIONS)]
        chosen = tuple(dict.fromkeys(f for p in picked for f in knobs_for(p)))
    return tuple(dict.fromkeys(SHEET_ALWAYS + chosen))


def _sheet_values(part: pl.DataFrame, shares: pl.DataFrame, rates: pl.DataFrame,
                  fields: Sequence[str]) -> pl.DataFrame:
    """`player_id` plus one column per field, each taken from whichever stage owns it.

    Four names -- snaps, routes, designed runs, dropbacks -- are both a participation metric and a pool
    the engine divides, and the pool is the one the projection multiplies through, so the share frame
    wins wherever they collide. That is the same precedence `metric_values` reads a single knob with,
    which is what keeps the sheet and the player page from disagreeing about one number.
    """
    own = [f for f in fields if f in SHEET_ALWAYS and f in part.columns]
    out = part.select("player_id", *own)
    for src in (shares, rates, part):
        take = [f for f in fields if f in src.columns and f not in out.columns]
        if take:
            out = out.join(src.select("player_id", *take), on="player_id", how="left")
    return out


def player_sheet(v: View, fields: Sequence[str], positions: Sequence[str] = (),
                 teams: Sequence[str] = (), search: str = "",
                 order: str = SHEET_ORDERS[0]) -> pl.DataFrame:
    """A roster as one editable table: who he is, what he is projected for, then the ratings behind it.

    The two projected totals sit between the identity and the knobs on purpose -- an edit is made to
    change one of them, so it should be in the same row as the number being changed rather than on the
    page you go to afterwards.

    A column nobody on screen has a number for is dropped rather than shown empty: a completion
    percentage in a receiving room is not a cell to fill in, it is a column that does not apply. With
    two positions on screen it cannot be dropped, because it applies to one of them and not the other,
    so those cells stay blank -- and `require_base` on the grid is what stops a blank one being typed
    into and recorded as an edit the engine would only report unapplied.
    """
    part, shares, rates = participation(v), shares_wide(v), rates_wide(v)
    ident = [c for c in ("player_id", "player", "position", "team", "status") if c in part.columns]
    out = part.select(ident).join(_sheet_values(part, shares, rates, fields), on="player_id",
                                 how="left")
    b = board(v)
    totals = [c for c in ("fantasy_points", "points_per_game") if c in b.columns]
    if totals:
        out = out.join(b.select("player_id", *totals), on="player_id", how="left")
    out = apply_filters(out, list(positions), list(teams), search)
    empty = [f for f in fields
             if f in out.columns and f not in SHEET_ALWAYS and out[f].null_count() == out.height]
    out = out.drop(empty)
    out = with_edits(out, "player", "player_id").drop("edits")

    keys = {"team, position, slot": ["team", "_pos", overrides.DEPTH_FIELD],
            "position, team, slot": ["_pos", "team", overrides.DEPTH_FIELD],
            "name": ["player"]}
    out = out.with_columns(pl.col("position")
                           .replace_strict({p: i for i, p in enumerate(POSITIONS)}, default=99,
                                           return_dtype=pl.Int32).alias("_pos"))
    if order == "projected points" and "fantasy_points" in out.columns:
        out = out.sort("fantasy_points", descending=True, nulls_last=True)
    else:
        out = out.sort([c for c in keys.get(order, keys["team, position, slot"])
                        if c in out.columns])
    shown = ["player_id", "edited", *[c for c in ("player", "position", "team", "status")
                                      if c in out.columns], *totals,
             *[f for f in fields if f in out.columns]]
    return out.select(list(dict.fromkeys(shown)))


def sheet_config(df: pl.DataFrame) -> dict:
    """The columns in the sheet that are not three-decimal numbers: the pencil, the slot, the games."""
    out: dict = {"edited": st.column_config.CheckboxColumn("✏️", help="he carries an override",
                                                           width="small")}
    if overrides.DEPTH_FIELD in df.columns:
        out[overrides.DEPTH_FIELD] = st.column_config.NumberColumn(
            "slot", min_value=1, max_value=20, step=1, format="%d",
            help="his rank in his room. Type a number and the room is rebuilt around him: everybody "
                 "else keeps the chart's order in the slots that are left, and he is re-priced as a "
                 "man in the job he moved into.")
    if GAMES_METRIC in df.columns:
        out[GAMES_METRIC] = st.column_config.NumberColumn(
            "games", min_value=0.0, max_value=FULL_SEASON, step=0.5, format="%.1f",
            help="how many of the seventeen he plays, availability included")
    return out


def sheet_baseline(v: View, fields: Sequence[str]) -> pl.DataFrame:
    """The sheet's own columns at the engine's numbers, which is what an edit records as its base."""
    b = baseline(v)
    return _sheet_values(b.part, b.shares, b.rates, fields)


@st.cache_data(show_spinner="working out who is producing this offence", max_entries=48)
def contributions(v: View, team: str, pool: str) -> pl.DataFrame:
    """Everybody on one roster claiming part of one team pool, biggest claim first.

    The share is the availability-weighted one the engine divided the pool with, not the share he takes
    while he is on the field: the question here is who is producing the *season*, and a starter
    projected for nine games is not producing a starter's season. `of_team` is his projected count over
    the team's, which is the number a reader actually wants -- "he is a quarter of this passing game".
    """
    p = opportunity.BY_POOL.get(pool)
    opp = opportunity_frame(v)
    if p is None or opp.is_empty() or pool not in opp.columns:
        return pl.DataFrame()
    mine = opp.filter(pl.col("team") == team)
    if mine.is_empty():
        return pl.DataFrame()
    team_col = f"team_{p.team_col}"
    per_week = mine.group_by("week").agg(pl.col(team_col).first().alias("team_count"))
    team_season = float(per_week["team_count"].sum())
    per = mine.group_by("player_id").agg(
        pl.col("player").first(),
        pl.col("position").first(),
        pl.col("depth_slot").first(),
        pl.col("status").first(),
        pl.col("expected_games").first(),
        pl.col(f"share_{pool}").mean().alias("share"),
        pl.col(pool).sum().alias("projected"),
    )
    out = per.join(
        board(v).select("player_id", "fantasy_points", "points_per_game", "position_rank"),
        on="player_id", how="left",
    ).with_columns(
        (pl.col("projected") / pl.lit(team_season)).alias("of_team") if team_season > 0
        else pl.lit(None, pl.Float64).alias("of_team")
    )
    return with_edits(out, "player", "player_id").filter(pl.col("projected") > 1e-9).sort(
        "projected", descending=True
    )


def pool_facts(v: View, team: str, pool: str) -> dict:
    """The arithmetic around one split: what the team has, what the roster claims, what scaling did."""
    p = opportunity.BY_POOL.get(pool)
    opp = opportunity_frame(v).filter(pl.col("team") == team)
    if p is None or opp.is_empty():
        return {}
    per_week = opp.group_by("week").agg(pl.col(f"team_{p.team_col}").first().alias("team_count"))
    team_season = float(per_week["team_count"].sum())
    con = contributions(v, team, pool)
    claimed = float(con["projected"].sum()) if not con.is_empty() else 0.0
    facts = {
        "pool": pool, "name": POOL_NAME.get(pool, label(pool)), "exclusive": bool(p.exclusive),
        "queue": bool(p.queue), "shares": tuple(p.shares), "players": con.height,
        "team_per_game": team_season / max(per_week.height, 1), "team_season": team_season,
        "claimed": claimed, "claimed_pct": (claimed / team_season) if team_season > 0 else None,
    }
    audit = team_pools(v).filter((pl.col("team") == team) & (pl.col("pool") == pool))
    if not audit.is_empty():
        row = audit.row(0, named=True)
        facts.update({k: row.get(k) for k in ("raw_sum", "measured_target", "factor", "gap_pct")})
    return facts


# The thing a sheet of shares cannot say about itself. Twelve numbers can each be defensible and add up to
# something impossible, and the reader finds out afterwards -- from a projection quietly scaled down, or
# not at all. So the sums belong under the sheet, beside what the pool actually has to give.
def pool_sums(v: View, team: str, fields: Sequence[str],
              player_ids: Sequence[str] = ()) -> pl.DataFrame:
    """Every pool the given shares feed: what this room claims, what the roster claims, what is there.

    `room` and `roster` are the numbers as typed, which is what the sheet above shows. `as run` is the
    roster's claim weighted by availability -- a man playing half the season claims half a share of the
    season's targets -- and that is the quantity normalisation actually sees, so it ties to the team
    page's own pool audit rather than being a second opinion about it.

    `there to divide` is measured from played seasons rather than assumed to be 1.0: a pool that really
    came back at 0.97 has 3% of its events going to somebody the player tables do not carry, and scaling
    a roster up to 1.0 would invent them.
    """
    wide = shares_wide(v)
    want = {f for f in fields if f in wide.columns}
    mine = participation(v).filter(pl.col("team") == team)
    if not want or mine.is_empty():
        return pl.DataFrame()
    have = list(dict.fromkeys(s for p in opportunity.POOLS for s in p.shares if s in wide.columns))
    grid = mine.select("player_id", "active_weeks").join(
        wide.select("player_id", *have), on="player_id", how="left")
    room = grid.filter(pl.col("player_id").is_in(list(player_ids))) if player_ids else grid
    audit = team_pools(v).filter(pl.col("team") == team)

    rows = []
    for p in opportunity.POOLS:
        parts = [s for s in p.shares if s in have]
        if not parts or not set(parts) & want:
            continue
        # the row is about the pool, so it sums every share dividing it even when the sheet only carries
        # one of them -- a carry sheet without `clean_rush_share` on it is still competing with it
        claim = pl.sum_horizontal([pl.col(s).fill_null(0.0) for s in parts])
        seen = audit.filter(pl.col("pool") == p.name)
        r = seen.row(0, named=True) if not seen.is_empty() else {}
        rows.append({
            "pool": POOL_NAME.get(p.name, label(p.name)),
            "shares": " + ".join(label(s) for s in parts),
            "room": float(room.select(claim.sum()).item() or 0.0),
            "roster": float(grid.select(claim.sum()).item() or 0.0),
            "as run": float(grid.select((claim * pl.col("active_weeks")).sum()).item() or 0.0),
            "there to divide": r.get("measured_target"),
            "over %": r.get("gap_pct"),
            "handling": ("not divided — several men are on the same snap" if not p.exclusive
                         else "filled in depth order" if p.queue else "scaled to the pool"),
        })
    return pl.DataFrame(rows, schema={"pool": pl.String, "shares": pl.String, "room": pl.Float64,
                                      "roster": pl.Float64, "as run": pl.Float64,
                                      "there to divide": pl.Float64, "over %": pl.Float64,
                                      "handling": pl.String})


def contribution_shift(before: View, after: View, team: str, pool: str) -> pl.DataFrame:
    """The same split under two scenarios: one row per man, what he has now and what he would have.

    Drawn as two bars per player, which is the picture a share edit deserves and a grid cannot give:
    one bar goes up and the ones under it go down by the same amount.
    """
    was = contributions(before, team, pool)
    now = contributions(after, team, pool)
    if was.is_empty() and now.is_empty():
        return pl.DataFrame()
    keep = ["player_id", "player", "position", "depth_slot"]
    a = was.select(*[c for c in keep if c in was.columns], pl.col("projected").alias("now"),
                   pl.col("share").alias("share_now"))
    b = now.select("player_id", pl.col("projected").alias("after"),
                   pl.col("share").alias("share_after"))
    out = a.join(b, on="player_id", how="full", coalesce=True)
    return out.with_columns(
        (pl.col("after").fill_null(0.0) - pl.col("now").fill_null(0.0)).alias("change")
    ).sort(pl.max_horizontal("now", "after"), descending=True, nulls_last=True)


# The team's own totals, so the third panel can answer "and what did the offence do" -- which for a
# share edit with normalisation on is *nothing*, and that is the lesson rather than a bug.
TEAM_TOTALS = ("fantasy_points", "targets", "receptions", "receiving_yards", "receiving_tds",
               "carries", "rushing_yards", "rushing_tds", "attempts", "passing_yards", "passing_tds")
RIPPLE_STATS = tuple(c for c in overrides.DIFF_COLUMNS if not c.endswith("_rank"))


@dataclass(frozen=True)
class Ripple:
    """One candidate edit run through the whole engine: the man, his room, the offence he came out of.

    `payload` is the candidate scenario's content, so a page can draw any frame under it -- the split
    before and after comes from calling `contributions` twice rather than from arithmetic on a page.
    """

    player_id: str
    player: str
    team: str
    field: str
    mode: str
    value: float
    season: int
    engine: float | None            # what the estimator says
    now: float | None               # what the projection is running on
    after: float | None             # what the candidate would make it
    pool: str | None                # the exclusive pool it claims, if it claims one
    payload: str
    me: pl.DataFrame                # stat, before, after, change -- the man edited
    movers: pl.DataFrame            # his teammates, biggest move first
    elsewhere: pl.DataFrame         # anybody off this roster who moved, which should be nobody
    team_totals: pl.DataFrame
    points_before: float
    points_after: float
    his_points: float
    room_points: float
    rank_before: float | None
    rank_after: float | None
    reading: str

    @property
    def candidate(self) -> View:
        return View(payload=self.payload, season=self.season)

    @property
    def net_points(self) -> float:
        return self.his_points + self.room_points

    @property
    def moved(self) -> bool:
        return abs(self.his_points) > 0.05 or not self.movers.is_empty()


def candidate_numbers(v: View, player_id: str, field: str, mode: str,
                      value: float) -> tuple[float | None, float | None, float | None]:
    """What the engine says, what the projection is running on, and what the candidate would make it."""
    vals = metric_values(v, field)
    if vals.is_empty():
        return None, None, None
    mine = vals.filter(pl.col("player_id") == player_id)
    if mine.is_empty():
        return None, None, None
    row = mine.row(0, named=True)
    engine = None if row.get("estimate") is None else float(row["estimate"])
    applied = row.get("applied")
    now = engine if applied is None else float(applied)
    after = float(value) if mode == "set" else (None if now is None else now * float(value))
    return engine, now, after


def _row_for(df: pl.DataFrame, player_id: str) -> dict:
    mine = df.filter(pl.col("player_id") == player_id)
    return mine.row(0, named=True) if mine.height else {}


def _as_float(x: object) -> float | None:
    return None if x is None else float(x)               # type: ignore[arg-type]


def _team_total(df: pl.DataFrame, team: str, column: str) -> float | None:
    if column not in df.columns or "team" not in df.columns:
        return None
    mine = df.filter(pl.col("team") == team)
    return None if mine.is_empty() else float(mine[column].sum())


def _reading(field: str, who: str, his: float, room: float, teammates: int,
             normalising: bool) -> str:
    """The sentence the three panels add up to. Here rather than on the page so a test can hold it."""
    if abs(his) < 0.05 and teammates == 0:
        return (f"Nothing moved. Either that is already the number, or {label(field)} is not in "
                f"{who}'s projection — check the room he is in before typing it again.")
    kind = knob_kind(field)
    net = his + room
    parts = [f"**{who} {his:+.1f} points**"]
    if teammates:
        parts.append(f"his room **{room:+.1f}** across {teammates} teammate"
                     f"{'' if teammates == 1 else 's'}")
    parts.append(f"the offence **{net:+.1f}**")
    if kind == "pool" and normalising and abs(net) < max(0.5, 0.35 * abs(his)):
        tail = (" — the pool is fixed, so most of this is a transfer inside the room rather than new "
                "production: what he gains, they lose.")
    elif kind == "pool" and not normalising:
        tail = (" — with pool normalisation off nothing is taken back from the room, so the offence "
                "now projects this much more than the team it was measured from.")
    elif kind == "pool":
        tail = (" — a share is zero-sum, so the honest reading is whether the losses are the ones you "
                "meant to cause.")
    elif kind == "availability":
        tail = (" — availability scales every count he has, and normalisation hands what he is not "
                "there for to the men behind him.")
    elif kind == "presence":
        tail = (" — participation is not divided, so this takes nothing from anybody: it changes how "
                "much of the pool he is on the field to claim.")
    else:
        tail = (" — a rate is his own, so nobody else's line rides on it." if not teammates
                else " — a rate is his own; anybody else who moved did so through a pool it feeds.")
    return " · ".join(parts) + tail


@st.cache_data(show_spinner="running the whole season with that one number changed", max_entries=16)
def ripple(v: View, player_id: str, field: str, mode: str = "set", value: float = 0.0) -> Ripple:
    """Run the season again with one number changed, and report everything that moved.

    Two runs and a difference, which is the only honest way to answer it: normalisation, the depth
    queue, the touchdown pools and the per-game chain all sit between a share and a projected point, so
    a number worked out on the page would be a different model from the one the board came from. The
    candidate is the *live* scenario plus this one edit, not a bare season, so what is shown is the
    effect of the next edit rather than of every edit at once.
    """
    engine, now, after = candidate_numbers(v, player_id, field, mode, value)
    cand = v.scenario.set(Override("player", player_id, field, mode, float(value), base=engine))
    before, later = board(v), projection_of(cand, v.season).board
    was, is_now = _row_for(before, player_id), _row_for(later, player_id)
    team = str(was.get("team") or is_now.get("team") or "")
    stats = [c for c in RIPPLE_STATS if c in before.columns]
    me = pl.DataFrame(
        {"stat": stats,
         "before": [_as_float(was.get(c)) for c in stats],
         "after": [_as_float(is_now.get(c)) for c in stats]},
        schema={"stat": pl.String, "before": pl.Float64, "after": pl.Float64},
    ).with_columns((pl.col("after") - pl.col("before")).alias("change")).filter(
        (pl.col("before").fill_null(0.0) != 0.0) | (pl.col("after").fill_null(0.0) != 0.0)
    )
    d = overrides.diff(before, later, threshold=0.01)
    here = ((pl.col("team").fill_null("") == team) | (pl.col("new_team").fill_null("") == team))
    movers = d.filter(here & (pl.col("player_id") != player_id))
    totals = pl.DataFrame(
        {"stat": list(TEAM_TOTALS),
         "before": [_team_total(before, team, c) for c in TEAM_TOTALS],
         "after": [_team_total(later, team, c) for c in TEAM_TOTALS]},
        schema={"stat": pl.String, "before": pl.Float64, "after": pl.Float64},
    ).with_columns((pl.col("after") - pl.col("before")).alias("change"))
    had = _as_float(was.get("fantasy_points")) or 0.0
    has = _as_float(is_now.get("fantasy_points")) or 0.0
    his = has - had
    room = float(movers["d_fantasy_points"].sum()) if not movers.is_empty() else 0.0
    who = str(was.get("player") or is_now.get("player") or player_id)
    return Ripple(
        player_id=player_id, player=who, team=team, field=field, mode=mode, value=float(value),
        season=v.season, engine=engine, now=now, after=after, pool=pool_for_field(field),
        payload=cand.content_json(), me=me, movers=movers, elsewhere=d.filter(~here),
        team_totals=totals, points_before=had, points_after=has, his_points=float(his),
        room_points=room,
        rank_before=_as_float(was.get("position_rank")),
        rank_after=_as_float(is_now.get("position_rank")),
        reading=_reading(field, who, float(his), room, movers.height, v.settings.normalize_pools),
    )
