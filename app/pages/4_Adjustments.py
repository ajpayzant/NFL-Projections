"""Every knob in one place, and the scenario that holds them.

The other pages let you edit the thing you are looking at. This page is the other half: the whole set
of edits as a list, what each one was worth before, what it is worth now, and a button to drop any one
of them. Nothing here is a second model -- an edit made on this page and the same edit made on the
team page are the same override, in the same scenario, producing the same projection.

Three things it exists to make possible:

- **Reset one knob.** Not "start again": drop the one edit you no longer believe and leave the rest.
  A reset knob follows the engine again, so it moves when the data underneath it moves.
- **Reproduce the ablations.** `k_scale` multiplies every fitted shrinkage constant. Zero is a
  player's own history at face value, a large number is his depth slot alone -- the two comparisons
  the backtest scored, available here rather than only in a script.
- **Compare two scenarios side by side.** "My rankings" against "consensus", or against the engine's
  own answer, as one table of who moved and by how much.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.model import overrides                                              # noqa: E402
from src.model.overrides import Override, Scenario                           # noqa: E402

view = ui.controls("Adjustments", icon="🎛️")
ui.scenario_banner(view)
sc = ui.live()
defaults = overrides.league_defaults()

# --------------------------------------------------------------------------- #
# the scenario itself
# --------------------------------------------------------------------------- #
st.subheader("Scenario")
saved = overrides.names()
cols = st.columns([3, 1, 1, 1])
name = cols[0].text_input("Name", value=sc.name, help="what this set of edits is called on disk")
if cols[1].button("Save"):
    keep = sc.rename(name)
    ui.set_live(keep)
    st.success(f"saved to `{overrides.save(keep)}`")
if cols[2].button("Clear all", help="drop every edit and go back to the engine"):
    ui.set_live(Scenario())
    st.rerun()
if cols[3].button("Delete", disabled=sc.name not in saved):
    overrides.delete(sc.name)
    st.rerun()

pick = st.selectbox("Load a saved scenario", ["—", *saved], index=0)
if pick != "—" and pick != sc.name:
    ui.set_live(overrides.load(pick))
    st.rerun()

st.caption(
    f"`{sc.name}` · digest `{sc.digest}` · {len(sc.items)} edits · "
    f"{len(sc.league)} league knobs · k×{sc.k_scale:g}"
    + (f" · saved {sc.updated}" if sc.updated else " · unsaved")
)

# --------------------------------------------------------------------------- #
# league
# --------------------------------------------------------------------------- #
st.subheader("League")
ui.note(
    "What the engine does for everybody. Scoring and the three modelling toggles are in the sidebar "
    "on every page; these are the constants behind them. A knob left at its default is not recorded, "
    "so the baseline stays the baseline."
)
lg = {k: sc.league.get(k, defaults[k]) for k in overrides.LEAGUE_FIELDS}

left, mid, right = st.columns(3)
with left:
    lg["normalize_pools"] = st.toggle("normalize pools", value=lg["normalize_pools"])
    lg["use_context_factors"] = st.toggle("use context factors", value=lg["use_context_factors"])
    lg["schedule_renormalise"] = st.toggle(
        "schedule renormalise", value=lg["schedule_renormalise"],
        help="On, each team's 17 context factors are rescaled to average 1, so the schedule only "
             "redistributes within the season instead of moving the season total.",
    )
    fitted_market = st.toggle("market weight: use the fitted value", value=lg["market_weight"] is None)
    lg["market_weight"] = None if fitted_market else st.slider(
        "market weight", 0.0, 1.0, float(lg["market_weight"] or 0.0), 0.05,
        help="How much of a game's scoring level comes from the posted line rather than our estimate.",
    )
with mid:
    lg["context_k"] = st.number_input(
        "context k", 0.0, 5000.0, float(lg["context_k"]), 25.0,
        help="Shrinks each split's bucket factor toward 1 by n / (n + k) team-games. Larger trusts "
             "the splits less.",
    )
    lg["team_weight_recent"] = st.slider("team weight recent", 0.0, 1.0,
                                        float(lg["team_weight_recent"]), 0.05)
    lg["team_keep_vs_mean"] = st.slider(
        "team keep vs mean", 0.0, 1.0, float(lg["team_keep_vs_mean"]), 0.05,
        help="How much of a team's recent form is kept against the league mean. 0 is every team "
             "average; 1 is last season taken at face value.",
    )
with right:
    k_scale = st.slider(
        "k scale — every shrinkage constant", 0.0, 5.0, float(sc.k_scale), 0.25,
        help="0 is a player's own history at face value; large is his depth-slot prior alone. The two "
             "ablations the backtest scored, both of which lost to the fitted value.",
    )
    lg["games_projected"] = st.number_input("games projected", 1, 17, int(lg["games_projected"]))
    lg["tier_size"] = st.number_input("tier size", 1, 24, int(lg["tier_size"]))
    rec = list(lg["recency"]) + [0.0, 0.0, 0.0]
    weights = st.columns(3)
    lg["recency"] = tuple(
        weights[i].number_input(f"recency {i + 1}", 0.0, 20.0, float(rec[i]), 0.5,
                                label_visibility="visible")
        for i in range(3)
    )

wanted = sc.patch_league(k_scale=k_scale, **lg)
if wanted.league != sc.league or wanted.k_scale != sc.k_scale:
    ui.set_live(wanted)
    st.rerun()

# --------------------------------------------------------------------------- #
# team
# --------------------------------------------------------------------------- #
st.subheader("Team")
env = ui.environment(view)
teams = sorted(env["team"].unique().to_list())
tcols = st.columns([1, 3])
team = tcols[0].selectbox("Team", teams, index=0)
scope = tcols[1].radio("Scope", ["every week", "one week"], horizontal=True)
tenv = env.filter(pl.col("team") == team).sort("week")
fields = tuple(c for c in overrides.TEAM_FIELDS if c in env.columns)

if scope == "one week":
    ui.grid(tenv.select("team", "week", "opponent", *fields), level="team", key_col="team",
            week_col="week", fields=fields, key=f"adj:env:{team}", digits=2, height=640,
            hide=("team",))
    ui.note("Type in a cell to set that week alone. The season average of the column is unchanged "
            "except by what you typed — the other sixteen games are left where the model had them.")
else:
    ui.note(
        "A multiplier applied to all seventeen games at once. `set` for one week lives in the other "
        "scope, because setting the same absolute number in every game would flatten the schedule "
        "into a season average and throw away the per-game chain."
    )
    for f in fields:
        ui.knob("team", team, f, base=float(tenv[f].mean()), fmt="%.3f",
                widget_key=f"adj:allweeks:{team}:{f}", modes=("multiply",))
    ui.note("`base` is this team's season mean; the multiplier scales every one of its games by it.")

# --------------------------------------------------------------------------- #
# player
# --------------------------------------------------------------------------- #
st.subheader("Player")
who = ui.pick_player(view, "Player", key="adj:player")
if who is not None:
    pid = who["player_id"]
    part = ui.participation(view).filter(pl.col("player_id") == pid)
    shares = ui.shares_wide(view).filter(pl.col("player_id") == pid)
    rates = ui.rates_wide(view).filter(pl.col("player_id") == pid)

    st.markdown(f"**{who['player']}** · {who['position']} {who['team']} · "
                f"{who['fantasy_points']:.1f} points, {who['games']:.1f} games")
    frames = {"availability": part, "shares": shares, "rates": rates}
    rows = []
    for kind, frame in frames.items():
        if frame.is_empty():
            continue
        row = frame.row(0, named=True)
        for f, val in row.items():
            if f not in overrides.PLAYER_FIELDS or val is None or not isinstance(val, (int, float)):
                continue
            if any(r["field"] == f for r in rows):
                continue
            rows.append({"kind": kind, "field": f, "used": float(val)})
    long = pl.DataFrame(rows, schema={"kind": pl.String, "field": pl.String, "used": pl.Float64})

    after = st.data_editor(
        long, key=f"adj:player:{pid}", hide_index=True, width="stretch", height=520,
        disabled=["kind", "field"], num_rows="fixed",
        column_config={"used": st.column_config.NumberColumn("used", format="%.4f")},
    )
    typed = [
        Override("player", pid, f, "set", float(new), base=float(old))
        for f, old, new in zip(long["field"], long["used"], after["used"], strict=True)
        if new is not None and new != old
    ]
    if typed:
        ui.edit(*typed)
        st.rerun()
    ui.note(
        "`used` is what the projection above was built from, edits included. Type a new number and it "
        "becomes a `set` override; the multiplier form and the reset live in the list below. Shares "
        "are still subject to pool normalisation — take 30% of the targets and the teammates are "
        "scaled down to keep the team's targets whole."
    )

# --------------------------------------------------------------------------- #
# every edit
# --------------------------------------------------------------------------- #
st.subheader("Every edit")
if not sc.items:
    st.info("No player or team edits. The league knobs above are the only thing separating this "
            "scenario from the engine's own answer.")
else:
    prov = ui.provenance(view)
    st.caption(f"{len(sc.items)} edits, "
               f"{int(prov['applied'].sum()) if not prov.is_empty() else 0} of them applied")
    ui.table(
        prov.select("stage", "level", "key", "field", "week", "mode", "value", "base_recorded",
                    "base_now", "used_now", "rows", "applied", "reason", "note"),
        digits=4, height=320,
    )
    ui.note(
        "One row per edit per frame it was offered to. `base_now` is the engine's own value at the "
        "moment it ran, which is the number to check a stale scenario against: an edit recorded "
        "against a 0.24 target share when the estimator now says 0.19 is an edit worth revisiting. "
        "`applied` false is an edit that found nothing — usually a player who has left."
    )

    st.markdown("**Drop one**")
    for o in sc.items:
        row = st.columns([6, 1])
        row[0].markdown(
            f"`{o.level}` **{o.label}**"
            + (f" · base {o.base:.4f}" if o.base is not None else "")
            + (f" · {o.note}" if o.note else "")
        )
        if row[1].button("↺", key=f"drop:{o.id}", help="drop this edit"):
            ui.reset(level=o.level, key=o.key, field_name=o.field, week=o.week)
            st.rerun()

# --------------------------------------------------------------------------- #
# against the baseline
# --------------------------------------------------------------------------- #
st.subheader("Against the baseline")
if sc.is_baseline:
    st.info("This *is* the baseline.")
else:
    summary = ui.board_diff_summary(view)
    ui.table(summary, config=ui.fixed(1, "abs_points_moved", "net_points_moved", "biggest_gain",
                                     "biggest_loss"))
    ui.note(
        "`net` against `abs` is the reading: a team edit that adds volume moves both together, while a "
        "share edit inside one receiving room shows a large `abs` and a `net` near zero, because "
        "normalisation took from the teammates what it gave the player."
    )
    moved = ui.board_diff(view)
    st.caption(f"{moved.height} players moved by more than 0.05 points")
    ui.table(
        moved.select("player", "position", "team", "fantasy_points", "new_fantasy_points",
                     "d_fantasy_points", "d_targets", "d_carries", "position_rank",
                     "new_position_rank").head(200),
        config={**ui.fixed(1, "fantasy_points", "new_fantasy_points", "d_fantasy_points"),
                **ui.fixed(2, "d_targets", "d_carries")},
        height=520,
    )

# --------------------------------------------------------------------------- #
# two scenarios side by side
# --------------------------------------------------------------------------- #
st.subheader("Compare two scenarios")
choices = ["(live)", "(baseline)", *saved]
ccols = st.columns(2)
a_name = ccols[0].selectbox("A", choices, index=1)
b_name = ccols[1].selectbox("B", choices, index=0)


def resolve(label: str) -> Scenario:
    if label == "(live)":
        return sc
    if label == "(baseline)":
        return Scenario(scoring=sc.scoring)
    return overrides.load(label)


a, b = resolve(a_name), resolve(b_name)
if a.content_json() == b.content_json():
    st.info("A and B are the same scenario.")
else:
    ra, rb = ui.projection_of(a, view.season), ui.projection_of(b, view.season)
    ui.table(
        overrides.summary(ra.board, rb.board),
        config=ui.fixed(1, "abs_points_moved", "net_points_moved", "biggest_gain", "biggest_loss"),
    )
    both = (
        ra.board.select("player_id", "player", "position", "team",
                        pl.col("fantasy_points").alias("a_points"),
                        pl.col("position_rank").alias("a_rank"))
        .join(rb.board.select("player_id", pl.col("fantasy_points").alias("b_points"),
                              pl.col("position_rank").alias("b_rank")),
              on="player_id", how="full", coalesce=True)
        .with_columns((pl.col("b_points") - pl.col("a_points")).alias("d_points"),
                      (pl.col("a_rank") - pl.col("b_rank")).alias("rank_gain"))
        .sort(pl.col("d_points").abs(), descending=True, nulls_last=True)
    )
    ui.table(
        both.drop("player_id").head(200),
        config={**ui.fixed(1, "a_points", "b_points", "d_points")},
        height=520,
    )
    ui.note(
        f"`A` is {a_name} and `B` is {b_name}; `d_points` and `rank_gain` are both B minus A, so a "
        "positive number is a player B likes better. A null on either side is a player one board has "
        "and the other does not."
    )
