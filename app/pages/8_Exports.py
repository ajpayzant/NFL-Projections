"""Take the projection with you: CSV, a formatted workbook, or Google Sheets.

The workbook this engine replaces was also how its numbers travelled, so this page has to be as good
as emailing an xlsx. It has one rule, and the whole page is arranged around it: **what leaves is what
is on screen.** Every table offered here comes from the same cached run every other page reads, under
the live scenario, through the same filters — so an export cannot quietly disagree with the board the
user was just looking at.

Two things are deliberately different from the display:

- **Full precision.** The app rounds to two decimals because a screen is for reading. A file is for
  computing with, so the frames are written exactly as the engine produced them.
- **Every column, not the chosen ones.** A page shows the columns worth reading; an export carries the
  rest too, because the one column somebody needs is always the one that was hidden.

`streamlit run app/Home.py` and pick Exports.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.config import PROJ_SEASON                                           # noqa: E402
from src.export import sheets                                                # noqa: E402

view = ui.controls("Exports", icon="📤")
ui.scenario_banner(view)

# Every table the engine produces, with the line that says what it is. The player-level ones take the
# filters; the rest are whole-league or per-team and are exported entire.
PLAYER_LEVEL = {"board", "season", "weekly", "opportunity", "participation", "shares", "rates",
                "provenance", "roster"}

TABLES: dict[str, tuple[str, str]] = {
    "board": ("the ranking board: season totals, ranks, tiers and the comparisons a draft is read for",
              "player"),
    "season": ("season totals per player, before the board's derived columns", "player"),
    "weekly": ("one row per player per game — the frame everything else is summed from", "player"),
    "opportunity": ("counts per player per game: the share of each team pool he is projected to get",
                    "player"),
    "participation": ("expected games, presence, status and the weekly participation rates", "player"),
    "shares": ("every share metric per player, after shrinkage", "player"),
    "rates": ("every efficiency rate per player, after shrinkage", "player"),
    "roster": ("who is on each team, where the depth chart puts him, and his draft capital", "player"),
    "provenance": ("for each estimate: the prior, the observation, the weight, and what was used",
                   "player"),
    "environment": ("the per-game team environment the players were divided out of", "team"),
    "shape": ("each team's projected season shape before it was spread over the schedule", "team"),
    "pools": ("what each pool's shares summed to before and after normalisation", "audit"),
    "team_pools": ("the same audit per team — which rosters the estimator struggles with", "audit"),
    "reconciliation": ("players summed against the team environment they were divided from", "audit"),
    "freshness": ("every table the engine read, how old it is, and which seasons it covers", "audit"),
}

LOADERS = {
    "board": ui.board,
    "season": ui.season_frame,
    "weekly": ui.weekly,
    "opportunity": ui.opportunity_frame,
    "participation": ui.participation,
    "shares": ui.shares_wide,
    "rates": ui.rates_wide,
    "roster": ui.roster_frame,
    "provenance": ui.provenance,
    "environment": ui.environment,
    "shape": ui.shape,
    "pools": ui.pool_report,
    "team_pools": ui.team_pools,
    "reconciliation": ui.reconciliation,
    "freshness": lambda v: ui.freshness(),
}

DEFAULT_CHOICE = ["board", "season", "weekly"]

# --------------------------------------------------------------------------- #
# what to export
# --------------------------------------------------------------------------- #
ui.section(
    "What leaves is what is on screen",
    "Every table offered here comes from the same cached run every other page reads, under the live "
    "scenario and through the filters set below — so an export cannot quietly disagree with the board "
    "you were just looking at. Two things are deliberately different from the display: the numbers go "
    "at **full precision** rather than rounded for reading, and **every column** goes rather than the "
    "ones a page chose to show, because the one column somebody needs is always the hidden one.",
    sub=f"season {view.season} · {len(TABLES)} tables the engine can write",
    level=2,
)

with st.container(border=True):
    left, right = st.columns([3, 2])
    with left:
        chosen = st.multiselect(
            "Tables", list(TABLES), default=DEFAULT_CHOICE,
            help="Each becomes a sheet in the workbook and a CSV of its own.",
        )
    with right:
        stem = st.text_input("File name", value=f"nfl_{PROJ_SEASON}_projections")

    fl, fm, fr = st.columns([2, 2, 3])
    with fl:
        positions = ui.position_filter("exp_pos")
    with fm:
        teams = ui.team_filter(view, "exp_team")
    with fr:
        search = st.text_input("Search", placeholder="part of a name", key="exp_search")

    tog, why = st.columns([2, 5], vertical_alignment="center")
    with tog:
        only_startable = st.toggle("Startable only", value=False, key="exp_startable",
                                   help=f"Inside the position's starter count: {view.settings.starters}")
    with why:
        ui.chips("filters apply to the player-level tables",
                 "team and audit tables go whole — a filtered audit is not an audit")

    # The simulated ranges are offered only when they exist. Simulating just to export would be a
    # ten-second surprise behind a download button.
    sim = ui.sim_controls(view, key="exp_sim")

# --------------------------------------------------------------------------- #
# build the set
# --------------------------------------------------------------------------- #
exports = sheets.ExportSet()
skipped: list[str] = []

for name in chosen:
    note, kind = TABLES[name]
    try:
        frame = LOADERS[name](view)
    except Exception as exc:                                  # noqa: BLE001 -- reported, not raised
        skipped.append(f"{name}: {exc}")
        continue
    if kind == "player":
        if {"position", "player"} <= set(frame.columns):
            frame = ui.apply_filters(frame, positions, teams, search)
        elif "team" in frame.columns and teams:
            frame = frame.filter(pl.col("team").is_in(teams))
        if only_startable and "startable" in frame.columns:
            frame = frame.filter(pl.col("startable"))
    exports.add(name, frame, note)

if sim is not None:
    ranges = sim.season
    if {"position", "player"} <= set(ranges.columns):
        ranges = ui.apply_filters(ranges, positions, teams, search)
    exports.add("ranges", ranges,
                f"simulated season distribution, {sim.draws:,} draws — floors, ceilings, volatility")
    exports.add("range_weekly", sim.weekly, "per-week boom and bust rates from the same simulation")

if skipped:
    st.warning("could not build: " + "; ".join(skipped), icon="⚠️")

if not len(exports):
    st.info("Pick at least one table.")
    st.stop()

# --------------------------------------------------------------------------- #
# what is going
# --------------------------------------------------------------------------- #
manifest = pl.DataFrame([
    {"table": s.name, "rows": s.frame.height, "columns": s.frame.width,
     "not in CSV": ", ".join(sheets.dropped_columns(s.frame)), "what it is": s.note}
    for s in exports.sheets
])
ui.section("What is going",
           "One row per sheet: how many rows it carries, how wide it is, and the line that will be "
           "written into the workbook's README beside it.",
           sub=f"{len(exports)} sheets · under the live scenario and the filters above")
ui.tiles([
    {"name": "sheets", "value": len(exports), "digits": 0, "sub": "one per table chosen"},
    {"name": "rows in all", "value": int(manifest["rows"].sum()), "digits": 0,
     "highlight": True, "sub": "at full precision, not rounded"},
    {"name": "widest table", "value": int(manifest["columns"].max()), "digits": 0,
     "sub": f"{manifest.sort('columns', descending=True).row(0, named=True)['table']} · columns"},
    {"name": "ranges included", "value": None,
     "sub": "yes — the simulation is on" if sim is not None else "no — turn Ranges on above"},
])
ui.table(manifest, height="auto")
lost = [c for s in exports.sheets for c in sheets.dropped_columns(s.frame)]
if lost:
    ui.note(f"`{'`, `'.join(sorted(set(lost)))}` hold a list per row — a player's history in one cell. "
            "CSV and Sheets cannot carry a nested column, so they are dropped there rather than written "
            "as a Python repr. Every other column goes at full precision.")

scenario_label = "baseline" if view.scenario.is_baseline else f"{ui.live().name} ({view.scenario.digest})"
stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
provenance = ui.sim_provenance()

# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
ui.section("Take it with you",
           "The workbook is the one to send: formatted, one sheet per table, with a README naming the "
           "scenario, the season and the moment it was written. A CSV is the one to compute with.",
           sub=f"{scenario_label} · written {stamp}")
book, single = st.columns([2, 3])

with book:
    st.download_button(
        f"Excel workbook — {len(exports)} sheets",
        data=sheets.workbook_bytes(exports, scenario=scenario_label, season=view.season,
                                   provenance=provenance),
        file_name=f"{stem}_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    ui.note("Formatted, one sheet per table, with a README naming the scenario and the moment. The "
            "workbook's structure without its formula engine — the values are the engine's, so no cell "
            "can be edited into disagreeing with its own inputs.")

with single:
    which = st.selectbox("CSV of one table", exports.names, key="exp_csv")
    sheet = exports.named(which)
    if sheet is not None:
        st.download_button(
            f"CSV — {which} ({sheet.frame.height:,} rows)",
            data=sheets.csv_bytes(sheet.frame),
            file_name=f"{stem}_{which}_{stamp}.csv",
            mime="text/csv",
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# Google Sheets
# --------------------------------------------------------------------------- #
ui.section("Google Sheets",
           "The same set written to a live spreadsheet instead of a file, for a league that shares one. "
           "A service account is looked for outside this repository on purpose, so a private key cannot "
           "be committed by accident.",
           sub="optional — a file works without any of this")
status = sheets.sheets_status()

if not status.ready:
    st.info(f"**Not configured.** {status.message}", icon="🔑")
    ui.note(
        "This is a configuration state rather than an error, so the rest of the page still works. A "
        "service account is looked for at `$NFLSP_GSPREAD_JSON`, then "
        "`~/.config/nflsp/service-account.json` — both outside this repository on purpose, so a private "
        "key cannot be committed by accident. The account then needs the Sheets and Drive APIs enabled."
    )
else:
    st.success(f"Ready as `{status.account}`", icon="🔑")
    gl, gr = st.columns([3, 2])
    with gl:
        title = st.text_input("Spreadsheet title", value=f"{stem} {stamp}")
    with gr:
        share_with = st.text_input("Share with", placeholder="you@example.com",
                                   help="A service account owns nothing a person can see. Without an "
                                        "address the sheet exists and is invisible.")
    if st.button("Write to Google Sheets", width="stretch"):
        with st.spinner("writing"):
            try:
                url = sheets.to_google_sheets(exports, title, share_with)
            except Exception as exc:                          # noqa: BLE001 -- shown, not swallowed
                st.error(f"Google rejected the write: {exc}")
            else:
                st.success(f"[Open the spreadsheet]({url})")
                if not share_with:
                    st.warning("Nobody was given access — the sheet is owned by the service account "
                               "and no human can open it yet.", icon="⚠️")

ui.note(f"Scenario: **{scenario_label}** · season {view.season} · {provenance}")
