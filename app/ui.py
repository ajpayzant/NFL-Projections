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

`streamlit run app/Home.py`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

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
    Settings,
)
from src.data import history, lake  # noqa: E402
from src.model import compose, efficiency, opportunity, overrides, roster, simulate  # noqa: E402
from src.model.overrides import Override, Scenario  # noqa: E402

SCORINGS = ("ppr", "half_ppr", "standard", "ppr_6td")
HISTORY_VIEW = tuple(range(2021, LAST_COMPLETE_SEASON + 1))   # the seasons a player page shows
LIVE = "scenario"                                             # the session-state key


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
    """The scenario being edited. One object, held in session state, replaced on every edit."""
    state = _state()
    if LIVE not in state:
        state[LIVE] = Scenario()
    return state[LIVE]


def set_live(scenario: Scenario) -> None:
    _state()[LIVE] = scenario


def edit(*items: Override) -> None:
    """Record edits against the live scenario. Every page that changes a number comes through here."""
    set_live(live().set(*items))


def reset(**where) -> None:
    set_live(live().clear(**where))


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
    st.title(title)
    sc = live()
    was = sc.league

    with st.sidebar:
        st.caption(f"{PROJ_SEASON} season projections")
        scoring = st.selectbox(
            "Scoring", SCORINGS, index=SCORINGS.index(sc.scoring or overrides.DEFAULT_SCORING),
            format_func=_scoring_label,
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
        # and, worse, make the widgets fight a scenario the Adjustments page had just loaded.
        edited = sc.patch_league(scoring=scoring, normalize_pools=normalize,
                                 use_context_factors=context,
                                 market_weight=None if market else 0.0)
        if edited.league != was or edited.scoring != sc.scoring:
            sc = edited
            set_live(sc)

        st.divider()
        _scenario_panel(sc)
        st.divider()
        _freshness_panel()
    return view()


def _scenario_panel(sc: Scenario) -> None:
    """The scenario's identity and size, on every page. An edit is never invisible."""
    st.markdown(f"**Scenario** `{sc.name}`")
    if sc.is_baseline:
        st.caption("no edits — this is the engine's own answer")
    else:
        applied = len(sc.items)
        bits = [f"{applied} edit{'s' if applied != 1 else ''}"]
        if sc.k_scale != 1.0:
            bits.append(f"k×{sc.k_scale:g}")
        st.caption(" · ".join(bits) + f" · `{sc.digest}`")
    try:
        st.page_link("pages/4_Adjustments.py", label="Adjustments →")
    except Exception:                     # noqa: BLE001
        # A page link resolves against the app's entrypoint. Running one page on its own -- which is
        # what the smoke harness does -- means there is no page set to resolve against, and the link
        # is not worth failing a page over.
        st.caption("Adjustments")


def _scoring_label(name: str) -> str:
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


# --------------------------------------------------------------------------- #
# the projection, under one scenario
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="projecting the season", max_entries=8)
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


@st.cache_data(show_spinner="projecting team season shape")
def shape(v: View) -> pl.DataFrame:
    from src.model import team

    return team.shape_wide(v.season, v.settings)


@st.cache_data(show_spinner="reading the shares behind each answer")
def share_detail(v: View) -> pl.DataFrame:
    from src.model import estimate

    sc = v.scenario
    return estimate.estimate(opportunity.SHARE_METRICS, roster_frame(v), v.season, v.settings,
                             overrides.fitted_for(sc))


@st.cache_data(show_spinner="reading the rates behind each answer")
def rate_detail(v: View) -> pl.DataFrame:
    return efficiency.rate_detail(v.season, v.settings, ros=roster_frame(v),
                                  fitted=overrides.fitted_for(v.scenario))


@st.cache_data(show_spinner="reading the participation behind each answer")
def participation_detail(v: View) -> pl.DataFrame:
    return roster.participation_detail(v.season, v.settings, ros=roster_frame(v),
                                       fitted=overrides.fitted_for(v.scenario))


@st.cache_data(show_spinner="auditing the pools")
def pool_report(v: View) -> pl.DataFrame:
    return opportunity.pool_report(v.season, v.settings, opportunity_frame(v))


@st.cache_data(show_spinner="auditing each team's pools")
def team_pools(v: View) -> pl.DataFrame:
    return opportunity.team_pool_sums(v.season, v.settings, shares=shares_wide(v),
                                      part=participation(v))


@st.cache_data(show_spinner="reconciling players against their teams")
def reconciliation(v: View) -> pl.DataFrame:
    return compose.team_check(weekly(v), v.season, v.settings, env=environment(v))


@st.cache_data(show_spinner="reading what actually happened")
def actuals(seasons: tuple[int, ...], scoring: str) -> pl.DataFrame:
    """Played seasons, scored the same way the projection is. Keyed on scoring, not on the whole
    `View`: an override cannot change what a player did in 2023."""
    return history.player_seasons(seasons, Settings().with_scoring(scoring).scoring)


@st.cache_data(show_spinner="reading defensive history")
def defence_by_position(season: int = LAST_COMPLETE_SEASON) -> pl.DataFrame:
    return history.defense_by_position((season,))


@st.cache_data(show_spinner="comparing against the baseline")
def board_diff(v: View) -> pl.DataFrame:
    return overrides.diff(baseline(v).board, board(v))


@st.cache_data(show_spinner="comparing against the baseline")
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
    out = df.with_columns(pl.col(pl.Float64, pl.Float32).round(digits))
    if per_column:
        out = out.with_columns(
            [pl.col(c).round(d) for c, d in per_column.items() if c in out.columns]
        )
    return out


def table(
    df: pl.DataFrame,
    digits: int = 2,
    per_column: dict[str, int] | None = None,
    height: int | str = "auto",
    config: dict | None = None,
    order: list[str] | None = None,
) -> None:
    """One table call for the whole app: rounded, index hidden, full width."""
    st.dataframe(
        rounded(df, digits, per_column),
        hide_index=True,
        height=height,
        width="stretch",
        column_config=config,
        column_order=order,
    )


def label(column: str) -> str:
    return column.replace("_", " ")


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
    """A one-line caption. The app explains a number where it stands and nowhere else."""
    st.caption(text)


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
        reset(level=level, key=key_value, field_name=field_name, week=week)
        st.rerun()
    unchanged = (mode == "multiply" and value == 1.0) or (base is not None and value == base
                                                          and mode == "set")
    if existing is None and unchanged:
        return
    if existing is not None and existing.mode == mode and float(existing.value) == float(value):
        return
    if unchanged and existing is not None:
        reset(level=level, key=key_value, field_name=field_name, week=week)
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
) -> list[Override]:
    """Turn an edited `st.data_editor` grid into overrides: one per cell whose value moved.

    Typing over a number is the whole interaction on the team page, so the grid is the editor and this
    is the translation. Only cells that actually changed become edits -- a grid rerun with nothing
    typed produces nothing -- and an existing edit keeps its original `base`, so the recorded "what it
    was" stays the engine's number rather than becoming last edit's answer.
    """
    keep = [f for f in fields if f in before.columns and f in after.columns]
    if not keep or before.height != after.height:
        return []
    cols = [key_col] + ([week_col] if week_col else []) + keep
    existing = {o.id: o for o in live().items}
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
            base = prior.base if prior is not None else (None if old is None else float(old))
            out.append(Override(level=level, key=key_value, field=f, mode="set",
                                value=float(new), week=week, base=base))
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
) -> None:
    """An editable table. Type in an editable column and the scenario has the edit on the next rerun.

    Everything not in `fields` is locked, because those columns are outputs or identity: a player's
    depth slot is a fact about the depth chart, and editing it here would edit nothing.
    """
    editable = [f for f in fields if f in df.columns]
    locked = [c for c in df.columns if c not in editable]
    after = st.data_editor(
        df, key=_keyed(key), hide_index=True, width="stretch", height=height,
        disabled=locked, num_rows="fixed",
        column_config={**{c: None for c in hide if c in df.columns},
                       **fixed(digits, *editable), **(config or {})},
    )
    edits = edits_from_grid(df, after, level, key_col, editable, week_col)
    if edits:
        edit(*edits)
        st.rerun()


def edited_badge(v: View, level: str, key_value: str) -> None:
    """Say so, wherever an edited thing is shown."""
    touching = live().touching(level, key_value)
    if touching:
        st.info(" · ".join(o.label for o in touching), icon="✏️")


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
