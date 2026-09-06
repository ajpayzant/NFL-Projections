"""Every edit in one list: what it was, what the engine has since, and a button to drop any one of them.

The other pages are where an edit gets *made*, because an edit is made while looking at the thing it is
about -- a room, a week, a man. This page is the ledger of them, which is the other half of the same job:
at forty edits nobody remembers what is in a scenario, and an override nobody remembers is an override
nobody will re-read when the estimator moves under it.

So nothing here is a second copy of a control that belongs beside the number it changes. What is here is
what has no home anywhere else:

- **Every edit.** One row each, with the base it was recorded against beside the engine's own number now
  -- *stale* is those two having come apart -- and the ↺ that drops one and leaves the rest.
- **Un-override.** The reverse of the bulk tab, at the same scale: pick the players to stop having an
  opinion about and every edit on each of them goes at once. Which is what makes overriding worth doing
  selectively -- a held share only buys you "these twelve and the engine for the rest" if the thirteenth
  is one button to hand back.
- **Many at once.** The population scale: "every quarterback at slot 1 plays 16.5 games". Still one
  override per player, so any single row of it can be dropped from the list afterwards.
- **League knobs.** The constants behind the model, `k_scale` among them: it multiplies every fitted
  shrinkage constant, and zero is a player's own history at face value while large is his depth slot
  alone -- the two ablations the backtest scored, available here rather than only in a script.
- **Compare scenarios.** "My rankings" against "consensus", or either against the engine's own answer, as
  one table of who moved and by how much.

Where the editors went when this page stopped keeping copies of them: a team's volume, per week or across
all seventeen at once, is the **Team** page's 📈 Team volume tab; a room's ratings, with the record beside
each knob, are its 🎛️ The rooms tab; one man's slot and every rating of his are the **Player** page's
✎ Adjust tab. Naming, opening and starting a scenario are in the sidebar of every page, and every edit is
written to disk as it is made.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.model import opportunity, overrides                                 # noqa: E402
from src.model.overrides import Scenario                                     # noqa: E402

view = ui.controls("Edits", icon="✏️")
ui.scenario_banner(view)
sc = ui.live()
defaults = overrides.league_defaults()

# --------------------------------------------------------------------------- #
# which scenario this is, above everything: it is the frame, not a section
# --------------------------------------------------------------------------- #
# A read-out and one button. Naming a scenario, opening a saved one and starting a new one are in the
# sidebar of every page, which is where they belong now that autosave means Save is not a thing anybody
# has to remember -- and a second copy of them here is how a session came to have two names for one set
# of edits. What the sidebar has no room for is removing a file, so that is what is left.
saved = overrides.names()
with st.container(border=True):
    ui.kv(scenario=sc.name, digest=sc.digest, edits=len(sc.items), league_knobs=len(sc.league),
          k_scale=f"×{sc.k_scale:g}", on_disk=sc.updated or "unsaved")
    row = st.columns([2, 6])
    if row[0].button("Delete from disk", width="stretch", disabled=sc.name not in saved,
                     help="removes the saved file; the edits on screen are left alone"):
        overrides.delete(sc.name)
        st.rerun()
    row[1].caption(
        "Renaming, opening a saved scenario and starting a new one are in the sidebar. Every edit is "
        "autosaved under this name as it is made, so **Delete** is about the file rather than about the "
        "edits in front of you — the next edit writes it again."
    )

TABS = [f"✏️ Every edit ({len(sc.items)})", "🧮 Many at once", "🌍 League knobs",
        "⚖️ Compare scenarios"]
edits_tab, bulk_tab, league_tab, compare_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# every edit: a list on the left, the one edit's own evidence on the right
# --------------------------------------------------------------------------- #
# An override is only trustworthy while somebody can still say why it is there, and the two things that
# rot are the reason and the base: an edit typed in August against a 0.24 target share is a different
# claim in October when the estimator has come round to 0.30 by itself. So the panel leads with those two
# numbers side by side, and the ✎ that changes it again is on the same screen as the ↺ that drops it.
EDIT_LIST = ["level", "who", "number", "change", "base", "engine now", "stale", "applied"]

with edits_tab:
    if not sc.items:
        st.info("No player or team edits. The league knobs are the only thing that could separate this "
                "scenario from the engine's own answer.")
    else:
        prov = ui.provenance(view)
        landed = {} if prov.is_empty() else {
            (r["level"], r["key"], r["field"], r["week"]): r for r in prov.rows(named=True)
        }
        names = {r["player_id"]: f"{r['player']} · {r['position']} {r['team']}"
                 for r in ui.board(view).select("player_id", "player", "position", "team")
                 .rows(named=True)}
        # A released man has no row in the run his own edit produced, so he would read here as a raw id.
        # The baseline is the one run he still exists in, and it is only asked about the ids that are
        # missing — the edited board stays the truth for everybody else.
        missing = [pid for pid in ui.released_ids() if pid not in names]
        if missing:
            names.update({r["player_id"]: f"{r['player']} · {r['position']} {r['team']} · off the roster"
                          for r in ui.baseline(view).board
                          .filter(pl.col("player_id").is_in(missing))
                          .select("player_id", "player", "position", "team").rows(named=True)})

        rows = []
        for o in sc.items:
            rec = landed.get((o.level, o.key, o.field, o.week), {})
            engine_now = rec.get("base_now")
            stale = (engine_now is not None and o.base is not None
                     and abs(float(engine_now) - float(o.base)) > 1e-6)
            rows.append({
                # `Override.id` is a tuple, which is a widget key and not a table cell
                "id": ":".join("" if x is None else str(x) for x in o.id),
                "level": o.level, "who": names.get(o.key, o.key),
                "number": ui.label(o.field) + ("" if o.week is None else f" · wk{o.week}"),
                "change": ("= " if o.mode == "set" else "× ") + f"{o.value:g}",
                "base": None if o.base is None else float(o.base),
                "engine now": None if engine_now is None else float(engine_now),
                "used now": None if rec.get("used_now") is None else float(rec["used_now"]),
                "stale": bool(stale), "applied": bool(rec.get("applied", True)),
                "why": o.note or "", "field": o.field, "key": o.key, "week": o.week,
                "mode": o.mode, "value": float(o.value),
            })
        # The five columns that are legitimately null for most rows are pinned rather than inferred.
        # polars infers a dtype from the first hundred rows only, and `week` is null on every season-wide
        # edit -- so a scenario with a hundred of those before its first per-game one had the column
        # typed as Null and then failed to append the int, taking the page down. Pinning is better than
        # a longer inference window here because it is also right when *every* row is null: the column is
        # still a week number, and a Null column would not read as one in the editor.
        edits = pl.DataFrame(rows, schema_overrides={
            "base": pl.Float64, "engine now": pl.Float64, "used now": pl.Float64,
            "week": pl.Int64, "value": pl.Float64,
        })

        ui.section(
            "Your edits, one at a time",
            "Pick a row and the right-hand panel is that edit: what you asked for, what the engine had "
            "when you asked, what it has now, and what the projection is running on. **stale** is the "
            "third of those having moved away from the second — the estimator has changed its mind "
            "since, and the edit is worth re-reading rather than assumed wrong.",
        )
        stale_n = int(edits["stale"].sum())
        gone = int((~edits["applied"]).sum())
        ui.tiles([
            {"name": "edits", "value": edits.height, "digits": 0, "highlight": True},
            {"name": "on players", "value": int((edits["level"] == "player").sum()), "digits": 0},
            {"name": "on teams", "value": int((edits["level"] == "team").sum()), "digits": 0},
            {"name": "one week only", "value": int(edits["week"].is_not_null().sum()), "digits": 0},
            {"name": "the engine has moved under", "value": stale_n, "digits": 0,
             "tone": "nf-dn" if stale_n else "", "sub": "worth re-reading"},
            {"name": "found nothing to change", "value": gone, "digits": 0,
             "tone": "nf-dn" if gone else "", "sub": "usually a player who has left"},
        ])

        left, right = st.columns([5, 7], gap="medium")
        with left:
            chosen = ui.pick_from(
                edits, key="edits:list", columns=EDIT_LIST, height=520,
                config={"stale": st.column_config.CheckboxColumn("stale", width="small"),
                        "applied": st.column_config.CheckboxColumn("ran", width="small"),
                        "who": st.column_config.TextColumn("who", width="medium"),
                        **ui.fixed(4, "base", "engine now")},
            )
        with right:
            if chosen is None:
                st.info("Pick an edit.")
            else:
                with st.container(border=True):
                    st.markdown(f"**{chosen['who']}** · {chosen['number']} `{chosen['change']}`")
                    if chosen["why"]:
                        st.caption(f"“{chosen['why']}”")
                    ui.tiles([
                        {"name": "you asked for", "value": chosen["value"], "digits": 4,
                         "highlight": True,
                         "sub": "as a multiplier" if chosen["mode"] == "multiply" else "set outright"},
                        {"name": "the engine had", "value": chosen["base"], "digits": 4,
                         "sub": "when you made the edit"},
                        {"name": "the engine has now", "value": chosen["engine now"], "digits": 4,
                         "tone": "nf-dn" if chosen["stale"] else "",
                         "sub": "it has moved since" if chosen["stale"] else "unchanged"},
                        {"name": "the projection ran on", "value": chosen["used now"], "digits": 4,
                         "sub": "" if chosen["applied"] else "this edit changed nothing"},
                    ])
                    if not chosen["applied"]:
                        rec = landed.get((chosen["level"], chosen["key"], chosen["field"],
                                          chosen["week"]), {})
                        st.warning(f"Not applied: {rec.get('reason') or 'no matching rows'}", icon="⚠️")

                    mine = len(sc.touching(chosen["level"], chosen["key"]))
                    act = st.columns([2, 2, 2, 3])
                    if act[0].button("↺ Drop this edit", key=f"drop:{chosen['id']}",
                                     help="follow the engine again on this one number"):
                        ui.drop_edit(chosen["level"], chosen["key"], chosen["field"], chosen["week"])
                        st.rerun()
                    if act[2].button(f"↺ Drop all {mine} on him", key=f"dropall:{chosen['id']}",
                                     disabled=mine < 2,
                                     help="every edit on this player or team, and nobody else's"):
                        ui.drop_all(chosen["level"], chosen["key"])
                        st.rerun()
                    with act[1]:
                        # not for a release: it is not a quantity to re-argue, so the only two things to
                        # do with it are keep it and drop it, and the button beside this one drops it
                        if (chosen["level"] == "player"
                                and chosen["field"] != overrides.ROSTER_FIELD):
                            # the ✎ rather than a second input: changing your mind about an edit is the
                            # same act as making it, and deserves the same evidence beside it
                            ui.knob_popover(view, chosen["key"], chosen["field"],
                                            base=chosen["engine now"], key=f"edits:again:{chosen['id']}",
                                            week=chosen["week"])
                    act[3].caption(
                        "Either button leaves everybody else alone — and returns him to the engine "
                        "rather than to whatever he was mid-session. To walk back a group of players "
                        "at once, use **Un-override** below."
                    )

                if chosen["level"] == "player" and chosen["key"] in names:
                    with st.expander(f"Everything else about {chosen['who']}"):
                        me = ui.board(view).filter(pl.col("player_id") == chosen["key"])
                        if not me.is_empty():
                            ui.player_panel(view, me.row(0, named=True),
                                            key=f"edits:panel:{chosen['id']}", compact=True,
                                            history=False)

        st.divider()
        # The bulk tab writes a population of overrides in one click, so there has to be one place that
        # takes a population of them back off. Not a filter over fields: the unit somebody changes their
        # mind about is a *man* -- "I have an opinion about these twelve receivers and I want the engine
        # to have the rest" -- and dropping him whole is what returns his room to normalising around the
        # players who are still held.
        ui.section(
            "Un-override",
            "Pick the players or teams to stop overriding and every edit on them goes at once. What is "
            "left keeps working the same way: a held share is still held, and the un-edited teammates of "
            "whoever you dropped absorb his share back — not evenly, but in proportion to how much of "
            "the pool each of them is already claiming.",
        )
        held = ([("player", k, n) for k, n in sc.counts("player").items()]
                + [("team", k, n) for k, n in sc.counts("team").items()])
        held.sort(key=lambda t: (-t[2], t[1]))

        def who_label(item: tuple[str, str, int]) -> str:
            level, key, n = item
            name = names.get(key, key) if level == "player" else f"{key} · team"
            return f"{name} — {n} edit{'s' if n != 1 else ''}"

        picked = st.multiselect("Who to stop overriding", held, format_func=who_label,
                                key="edits:unoverride",
                                help="every edit on each one, season-wide and per week")
        going = sum(n for _, _, n in picked)
        un = st.columns([3, 2, 5])
        if un[0].button(f"↺ Drop {going} edits on {len(picked)} of them", type="primary",
                        width="stretch", disabled=not picked, key=ui._keyed("edits:unoverride:go")):
            for level in ("player", "team"):
                ui.drop_all(level, *[k for lv, k, _ in picked if lv == level])
            st.rerun()
        un[1].metric("would remain", len(sc.items) - going)
        un[2].caption(
            "Immediate and autosaved, like every other edit — the safety net is the scenario's own "
            "backups, not a confirmation box. Dropping is per player, so a man you keep is untouched "
            "even where the two of them share a receiving room."
        )

        st.divider()
        ui.section(
            "Where every one of them landed",
            "One row per edit per frame it was offered to. `base_now` is the engine's own value at the "
            "moment it ran, which is the number to check a stale scenario against. A `depth` stage row "
            "is a move on the chart and its numbers are slots. `applied` false is an edit that found "
            "nothing — usually a player who has left.",
            sub=f"{len(sc.items)} edits, "
                f"{int(prov['applied'].sum()) if not prov.is_empty() else 0} applied",
        )
        ui.focus_table(
            prov.select("stage", "level", "key", "field", "week", "mode", "value", "base_recorded",
                        "base_now", "used_now", "rows", "applied", "reason", "note"),
            ["stage", "level", "key", "field", "mode", "value", "base_now", "used_now", "applied"],
            key="edits:prov", digits=4, height=360,
        )

# --------------------------------------------------------------------------- #
# many at once
# --------------------------------------------------------------------------- #
# The third scale of edit. A knob is one number, a grid is one room, and this is a population: "every
# quarterback at slot 1 plays 16.5 games", "every rookie receiver at nine tenths of his share". It is
# still one override per player, so any single row of it can be dropped on the edits tab afterwards.
with bulk_tab:
    ui.section(
        "One edit, a population at a time",
        "Pick who, pick the number, look at what it would do, then write it. Nothing is recorded until "
        "**Apply** — the preview is the whole point of the tab, because a filter that catches thirty men "
        "when you meant twelve is only visible as a list. Each row becomes its own override with its own "
        "base, so a bulk edit can be unpicked one player at a time on the edits tab.",
    )
    part_all = ui.participation(view)
    statuses = sorted(part_all["status"].drop_nulls().unique().to_list()) \
        if "status" in part_all.columns else []
    all_teams = sorted(part_all["team"].drop_nulls().unique().to_list())

    who_cols = st.columns([2, 2, 2, 2])
    with who_cols[0]:
        bpos = st.multiselect("Position", list(ui.POSITIONS), default=["QB"], key="bulk:pos")
    with who_cols[1]:
        bteams = st.multiselect("Team", all_teams, key="bulk:team",
                                help="leave empty for the whole league")
    with who_cols[2]:
        bstat = st.multiselect("Roster status", statuses, key="bulk:status",
                               help="leave empty for every status")
    with who_cols[3]:
        slots = st.slider("Depth slots", 1, 12, (1, 1), key="bulk:slots",
                          help="1 to 1 is the starters; 2 to 12 is everybody behind them")

    more = st.columns([2, 2, 4])
    rookies = more[0].toggle("Rookies only", value=False, key="bulk:rookies")
    bsearch = more[1].text_input("Name contains", key="bulk:search")

    pop = ui.bulk_population(view, bpos, bteams, slots, bstat, rookies, bsearch)
    with more[2]:
        ui.tiles([{"name": "players caught by this filter", "value": pop.height, "digits": 0,
                   "highlight": True}])

    if pop.is_empty():
        st.info("Nobody matches that filter.")
    else:
        with st.expander(f"The {pop.height} players it would touch"):
            ui.table(pop.drop("player_id"), digits=1, height=min(80 + 35 * pop.height, 420))

        st.divider()
        ui.section("The number")
        what = st.columns([3, 2, 2, 3])
        field = what[0].selectbox("Field", list(ui.BULK_FIELDS), format_func=ui.label,
                                  key="bulk:field")
        mode = what[1].radio("How", list(overrides.MODES), horizontal=True, key="bulk:mode",
                             format_func=lambda m: {"set": "set to", "multiply": "multiply by"}[m])
        step = 0.5 if field == ui.GAMES_METRIC else 0.01
        default = (16.5 if field == ui.GAMES_METRIC else 0.20) if mode == "set" else 1.0
        value = what[2].number_input("Value", value=float(default), step=step, format="%.4f",
                                     key=f"bulk:value:{field}:{mode}")
        why = what[3].text_input("Note recorded on every one of them", key="bulk:why",
                                 placeholder="why you believe this")

        preview = ui.bulk_preview(view, tuple(pop["player_id"].to_list()), field, mode, float(value))
        if preview.is_empty():
            st.warning(
                f"Nobody in that population has a `{ui.label(field)}`, or the new value equals what "
                "they are already on — nothing to write.", icon="⚠️")
        else:
            missed = pop.height - preview.height
            digits = 1 if field == ui.GAMES_METRIC else 3
            ui.tiles([
                {"name": "would move", "value": preview.height, "digits": 0, "highlight": True,
                 "sub": f"{missed} of the filter unaffected" if missed else "every man in the filter"},
                {"name": "mean now", "value": preview["now"].mean(), "digits": digits},
                {"name": "mean after", "value": preview["after"].mean(), "digits": digits,
                 "sub": f"{preview['change'].mean():+.{digits}f} each"},
                {"name": "biggest single change", "value": preview["change"].abs().max(),
                 "digits": digits},
            ])
            ui.rank_bars(preview.sort(pl.col("change").abs(), descending=True).head(16), "change",
                         height=420, digits=digits, colour="#54a24b")
            ui.focus_table(
                preview.drop("player_id"),
                ["player", "position", "team", "engine", "now", "after", "change"],
                key=f"bulk:preview:{field}:{mode}", digits=3,
                config=ui.fixed(digits, "engine", "now", "after", "change"),
                height=min(80 + 35 * preview.height, 460),
                note_text=(
                    "`now` is what the projection is running on, edits included, and `engine` is the "
                    "estimator's own number — the base each override is recorded against, so dropping "
                    "one later returns that man to the engine rather than to whatever he was "
                    "mid-session. A player already carrying your own edit on this field will have it "
                    "replaced."
                ),
            )
            apply_cols = st.columns([2, 4])
            if apply_cols[0].button(f"Apply to {preview.height} players", type="primary",
                                    width="stretch", key=ui._keyed("bulk:apply")):
                ui.edit(*ui.bulk_edits(preview, field, mode, float(value), why))
                st.rerun()
            apply_cols[1].caption(
                f"writes {preview.height} overrides: `{ui.label(field)}` {mode} {value:g}"
                + (f" · note “{why}”" if why else "")
            )
            if field in ("target_share", "carry_share", "share_targets", "share_carries"):
                ui.note("A share is zero-sum inside a team: pool normalisation will scale the teammates "
                        "of everybody in this list, so the board moves for men the filter never named.")

# --------------------------------------------------------------------------- #
# league
# --------------------------------------------------------------------------- #
with league_tab:
    ui.section(
        "What the engine does for everybody",
        "Scoring and the three modelling toggles are in the sidebar on every page; these are the "
        "constants behind them. A knob left at its default is not recorded, so the baseline stays the "
        "baseline.",
    )
    lg = {k: sc.league.get(k, defaults[k]) for k in overrides.LEAGUE_FIELDS}
    fitted_tilt_value = float(opportunity.load_tilt().get("pool_tilt", 1.0))
    if sc.league:
        ui.chips(*[f"{ui.label(k)} = {v}" for k, v in sc.league.items()])

    left, mid, right = st.columns(3)
    with left:
        st.markdown("**What the model is allowed to do**")
        lg["normalize_pools"] = st.toggle("normalize pools", value=lg["normalize_pools"])
        lg["lock_edited_shares"] = st.toggle(
            "hold the shares you type", value=lg["lock_edited_shares"],
            help="On, a share you set is delivered at what you typed and the un-edited teammates absorb "
                 "it. Off, your number is rescaled along with everybody else's — so 0.30 arrives as "
                 "0.2545 and the edits list still reports it applied.",
        )
        # The exponent, with the fitted value as its default in the same shape `market_weight` uses. Worth
        # a control rather than a constant precisely because the fit came back indifferent: held-out error
        # cannot tell these apart, so it is a judgement about who is more likely to be wrong, and the
        # person making the judgement should be able to move it and look at the board.
        fitted_tilt = st.toggle("pool tilt: use the fitted value", value=lg["pool_tilt"] is None,
                                help=f"currently {fitted_tilt_value:g}, from "
                                     "data/fitted/pool_tilt.json — and the fit that chose it came back "
                                     "indifferent, so it is a preference rather than a measurement")
        lg["pool_tilt"] = None if fitted_tilt else st.slider(
            "pool tilt", 0.0, 1.0,
            float(lg["pool_tilt"] if lg["pool_tilt"] is not None else fitted_tilt_value), 0.05,
            help="Who pays when a room claims more of a pool than it holds. 1 charges every claim the "
                 "same fraction of itself; below 1 charges the small claims a larger fraction, so the "
                 "starters keep more and the bench pays; 0 charges everyone the same absolute amount.",
        )
        lg["use_context_factors"] = st.toggle("use context factors", value=lg["use_context_factors"])
        lg["schedule_renormalise"] = st.toggle(
            "schedule renormalise", value=lg["schedule_renormalise"],
            help="On, each team's 17 context factors are rescaled to average 1, so the schedule only "
                 "redistributes within the season instead of moving the season total.",
        )
        fitted_market = st.toggle("market weight: use the fitted value",
                                  value=lg["market_weight"] is None)
        lg["market_weight"] = None if fitted_market else st.slider(
            "market weight", 0.0, 1.0, float(lg["market_weight"] or 0.0), 0.05,
            help="How much of a game's scoring level comes from the posted line rather than our "
                 "estimate.",
        )
    with mid:
        st.markdown("**How much history to trust**")
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
        k_scale = st.slider(
            "k scale — every shrinkage constant", 0.0, 5.0, float(sc.k_scale), 0.25,
            help="0 is a player's own history at face value; large is his depth-slot prior alone. The "
                 "two ablations the backtest scored, both of which lost to the fitted value.",
        )
    with right:
        st.markdown("**Shape of the season and the board**")
        lg["games_projected"] = st.number_input("games projected", 1, 17, int(lg["games_projected"]))
        lg["tier_size"] = st.number_input("tier size", 1, 24, int(lg["tier_size"]))
        rec = list(lg["recency"]) + [0.0, 0.0, 0.0]
        weights = st.columns(3)
        lg["recency"] = tuple(
            weights[i].number_input(f"recency {i + 1}", 0.0, 20.0, float(rec[i]), 0.5,
                                    label_visibility="visible")
            for i in range(3)
        )
        ui.note("`recency` weights the seasons of a player's own record, most recent first.")

    wanted = sc.patch_league(k_scale=k_scale, **lg)
    if wanted.league != sc.league or wanted.k_scale != sc.k_scale:
        ui.set_live(wanted)
        st.rerun()

# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def diff_tiles(summary: pl.DataFrame) -> list[dict]:
    """`overrides.summary` is one row per position; the tiles above it are the whole board.

    Adding the positions up here rather than showing the first row: the table under the tiles is the
    per-position reading, and a headline that silently meant *the receivers* would be the wrong number
    in the largest type on the page.
    """
    if summary.is_empty():
        return [{"name": "players moved", "value": 0, "digits": 0}]
    tot = summary.select(
        pl.col("players_moved").sum().alias("players_moved"),
        pl.col("abs_points_moved").sum().alias("abs_points_moved"),
        pl.col("net_points_moved").sum().alias("net_points_moved"),
        pl.col("biggest_gain").max().alias("biggest_gain"),
        pl.col("biggest_loss").min().alias("biggest_loss"),
    ).row(0, named=True)
    return [
        {"name": "players moved", "value": tot["players_moved"], "digits": 0, "highlight": True,
         "sub": f"across {summary.height} position(s)"},
        {"name": "points moved", "value": tot["abs_points_moved"],
         "sub": "added up regardless of direction"},
        {"name": "net", "value": tot["net_points_moved"], "signed": True,
         "sub": "near zero is production moved, not created"},
        {"name": "biggest gain", "value": tot["biggest_gain"], "signed": True},
        {"name": "biggest loss", "value": tot["biggest_loss"], "signed": True},
    ]


with compare_tab:
    ui.section(
        "Against the baseline",
        "`net` against `abs` is the reading: a team edit that adds volume moves both together, while a "
        "share edit inside one receiving room shows a large `abs` and a `net` near zero, because "
        "normalisation took from the teammates what it gave the player.",
    )
    if sc.is_baseline:
        st.info("This *is* the baseline.")
    else:
        summary = ui.board_diff_summary(view)
        ui.tiles(diff_tiles(summary))
        ui.table(summary, config=ui.fixed(1, "abs_points_moved", "net_points_moved", "biggest_gain",
                                          "biggest_loss"))
        moved = ui.board_diff(view)
        ui.rank_bars(moved.sort("d_fantasy_points", descending=True), "d_fantasy_points", height=440,
                     top=24)
        ui.focus_table(
            moved.select("player", "position", "team", "fantasy_points", "new_fantasy_points",
                         "d_fantasy_points", "d_targets", "d_carries", "position_rank",
                         "new_position_rank").head(200),
            ["player", "position", "team", "fantasy_points", "new_fantasy_points", "d_fantasy_points",
             "position_rank", "new_position_rank"],
            key="compare:moved",
            config={**ui.fixed(1, "fantasy_points", "new_fantasy_points", "d_fantasy_points"),
                    **ui.fixed(2, "d_targets", "d_carries")},
            height=460,
            note_text=f"{moved.height} players moved by more than 0.05 points. The teammates are in "
                      "here on purpose: a share taken is a share given.",
        )

    st.divider()
    ui.section("Two scenarios side by side")
    choices = ["(live)", "(baseline)", *saved]
    ccols = st.columns(2)
    a_name = ccols[0].selectbox("A", choices, index=1)
    b_name = ccols[1].selectbox("B", choices, index=0)

    def resolve(labelled: str) -> Scenario:
        if labelled == "(live)":
            return sc
        if labelled == "(baseline)":
            return Scenario(scoring=sc.scoring)
        return overrides.load(labelled)

    a, b = resolve(a_name), resolve(b_name)
    if a.content_json() == b.content_json():
        st.info("A and B are the same scenario.")
    else:
        ra, rb = ui.projection_of(a, view.season), ui.projection_of(b, view.season)
        pair = overrides.summary(ra.board, rb.board)
        ui.tiles(diff_tiles(pair))
        ui.table(pair, config=ui.fixed(1, "abs_points_moved", "net_points_moved", "biggest_gain",
                                       "biggest_loss"))
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
            height=460,
        )
        ui.note(
            f"`A` is {a_name} and `B` is {b_name}; `d_points` and `rank_gain` are both B minus A, so a "
            "positive number is a player B likes better. A null on either side is a player one board "
            "has and the other does not."
        )
