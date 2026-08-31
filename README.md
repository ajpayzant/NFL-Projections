# NFL season projections

Season-long projections for every player on an NFL roster, built game by game from observed data. This
is a Streamlit app that replaces an Excel workbook — the same structure, without the formula engine, and
with the numbers coming from a model that has been scored against seasons it never saw.

The engine is one identity, applied 17 times per player:

```
volume(week)  = participation x share x team volume(week)
stat(week)    = volume(week) x efficiency
season        = sum over the 17 scheduled games
```

Everything else is how each of those four terms is estimated. `participation` is availability times
snap or route share; `share` is the player's slice of a team pool that sums to one; `team volume` comes
from a per-game team environment (pace, pass rate, opponent, weather, market); `efficiency` is a rate
shrunk toward the prior for the player's job. A player's probability of playing is inside every count,
never applied afterwards, so a projection is never a healthy-season number with a discount stapled on.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # or: pip install polars streamlit pyarrow xlsxwriter
.venv/Scripts/python -m src.data.refresh          # rosters, depth charts, schedules, injuries
.venv/Scripts/streamlit run app/Home.py                # serves on http://localhost:8611
```

For day-to-day use start it from **`run_app.bat`** (double-click, or point a Desktop shortcut at it)
rather than from a terminal you intend to close. A server started inside a shell — or inside an agent
session — is a child of that shell and is killed when it ends, which the browser reports mid-edit as a
"connection error" with nothing in the log. `run_app.bat` gives the process to the desktop instead.

The app serves on **port 8611**, pinned in `.streamlit/config.toml`. Don't run it on Streamlit's
default 8501 — the PLL Pricing Tool launcher probes that port, and an app sitting on it will hijack
that shortcut. See `~/PORTS.md` for the machine-wide port registry.

Python 3.13+. On Windows set `PYTHONIOENCODING=utf-8` for the command-line tools — several of them print
tables with box-drawing characters.

## Data

Two sources, and the split matters:

- **The shared lake** — the heavy `processed/` tables (team games, player usage, passer games) are built
  by the play-by-play pipeline in `~/nfl-projection-system`. This project **reads them and never writes
  them.** Point `NFLSP_DATA_DIR` at a copy if the lake lives elsewhere.
- **Our own `data/`** — the four light tables this repo refreshes itself (`data/raw/`), and everything
  the model fits (`data/fitted/`: priors, shrinkage constants, rookie curves, dispersion, the backtest).
  A season present in `data/raw/` wins over the same season in the lake, so a Tuesday roster refresh
  takes effect here without waiting on the other repo's pipeline.

`data/raw/` and `data/cache/` are not committed — `python -m src.data.refresh` rebuilds them in about a
minute, and a committed roster snapshot is stale the moment somebody signs. `data/fitted/` **is**
committed: it is small, it is what the backtest scored, and a clone should project without a refit.

**History → ⚖️ Share sums** reports the age of every table, and separately whether `processed/` has kept up
with the week the season is actually in — the question an age in days cannot answer, since a table
refreshed this morning can still be missing last Sunday's games.

## Is it any good

Two held-out seasons (2024, 2025), full PPR, scored over every player the projection named. Season
fantasy points, players who played:

| variant | MAE | RMSE | Spearman |
|---|---|---|---|
| **full** (the engine) | **36.9** | **52.7** | **0.750** |
| workbook (what this replaces) | 38.6 | 56.1 | 0.739 |
| prior_only (no player history) | 42.0 | 58.2 | 0.707 |
| ewma (player's own recency-weighted average) | 44.4 | 67.4 | 0.599 |

The ablations are in the same table on **History → 📉 Backtest**: each turns off one node, and every node
that pays for itself is kept on that evidence. `no_normalise` has a *lower* MAE than `full` and is still
off, because it wins by under-projecting into a population that is four-tenths zeroes while losing on
RMSE and on rank order — the two things a draft board is actually read for. A node is only reverted on a
clean sweep of MAE, RMSE and Spearman.

Ranges are calibrated, not assumed: 91.8% of held-out seasons landed inside their advertised P5–P95
(against 90%), 4.6% below the floor and 3.7% above the ceiling. The width is fitted on the
interquartile range of the PIT, which is invariant to a shift and therefore scores only the width — a
distance-from-uniform score would widen an interval to absorb a level error.

**One known residual, documented rather than corrected.** Across the 5–11 projected-games band the mean
is over-projected by roughly 6–12 points, offset by a small under-projection across the deep bench; pool
normalisation conserves the total, so this is a distributional error along the games axis and not a level
error. Every correction tested either had the wrong sign in aggregate or traded the mid-band for the
tail. The Calibration tab shows it, and shows why the negative `median_err` in every band is *not* a
defect: season points are right-skewed, so a projection correctly aimed at the mean sits above the
median outcome by construction.

## The pages

| page | what it is for |
|---|---|
| **Home / Rankings** | the board, in three column sets: **Fantasy** (points, per game, above the average starter, the drop to the next man), **Stat line** (the projected box score for the positions on screen), **Everything**. Filters, tiers, positional ranks, a stat-leaders tab, volatility when ranges are on. |
| **League** | the projection as thirty-two records, in four tabs: **Standings** (eight division tables, because nine wins is a title in one division and third place in another), **All 32 offences** (a team a row, ranked on any column), **Every game** (all 272 fixtures with both implied points and the line), **How far through the league** (which teams you have actually reviewed). A projected record is a sum of seventeen win probabilities rather than a count of games won, and the σ that turns a margin into a probability is fitted on played games rather than assumed. |
| **Team** | one offence as a whole, in six tabs in the order the engine multiplies and the order somebody disagrees in: **Depth chart** (each room in slot order with every man's share of the pool and his projected line — and the chart itself editable, type a slot to move a man, and a control to take a man off the roster entirely so nothing counts him), **Availability** (the games for the whole roster, or a whole room at once, with each man's attendance record beside the knob), **Rooms** (the inputs per position, checkable one metric at a time against the league and the room), **What if** (below), **Team volume** (what they have run per game against what is projected, editable per week), **Does it add up** (the pool audits). |
| **Player** | one player end to end: his projected stat line as a box score and as cards, the weekly path, and **his ratings** — one row per estimate with `obs` / `prior` / `used` / `applied`, the sample size behind it, and a knob to override any of them where it is shown — pick a number and his own record, his job's average, his league percentile and his room come up with it. **✎ Adjust** is the same edits in a batch: every rating he has as a typed list, for the season or for chosen weeks, applied in one go. **👥 Competition** is the men he is dividing the same pool with, and **⚖️ Side by side** is him against one to three others on every estimate they have in common with the sample size beside each — the layer that answers *why*, since the same target share on 900 routes and on 90 are not the same claim. Ranges on the same switch as everywhere else, because a man with the higher median and the lower floor is a different pick rather than a better one. |
| **Week** | the season a week at a time, which is how the engine computed it — one week picker and one fixture picker over four tabs: **The week's board** (this week's board with the matchup, the market line and the rank at the position *in that week*; the slate itself with the roof, the rest and the line; a row-per-player, column-per-week grid with the season drawn as a line; and the bye calendar, where a bye is an empty cell rather than a zero), **One game** (the fixture at the scale the engine is actually built at: the projected score, both offences' stat lines, every man's line under them, and the pool audit for that afternoon), **Adjust this game** (below — the knobs for one afternoon, and the season that follows them), **Opponent & market** (the schedule as an input: every team × week as a matrix, what kind of games each team plays, last season's defence by position as *history* rather than as a projection, and the softest and hardest weeks with the game's own scoring level beside the opponent's record). |
| **Availability** | the one term the projection states rather than fits, on one surface, in two tabs: **Who needs a decision** (everybody not being treated as a full season, with his injury record beside the knob), **Back in week N** (a return date converted into the games it is worth off that team's own schedule), with the whole board under it as a sheet, each roster status's factor with the population it is currently deciding, and the league's injury reports under that. |
| **History** | what actually happened, at the three levels a projection is argued about, and the model scored against all three — seven tabs: **The league** (the scoring environment season by season, so the level every team number is read against is explicit rather than remembered), **Teams** (a team a row, its five seasons against the league mean and against what is projected next), **Players** (a man's record season by season on the live scoring, with his projection as the last bar of it), **Projection against record** (the whole board against what each of them did last season — the risers, the fallers, the size of each claim — and, under it, the season being played as it fills in), **Backtest** (held-out seasons against two baselines: the player's own recency-weighted average and the workbook this replaces), **Calibration** (slope against the 1.00 it claims, per position and per projected-games band, with the intervals checked against the one season in twenty they advertise), **Share sums** (the pool audit, the age of every table read, and whether `processed/` has kept up with the week the season is in). The first four tabs are measured rather than modelled, and the one modelled number on them is labelled as the projection every time; the backtest is read from what the script last wrote rather than run in the browser. |
| **Edits** | the ledger of the override layer, and the three things that have no home beside a number: **Every edit** (one row each, the base it was recorded against beside the engine's own number now — *stale* is those two having come apart — and the ↺ that drops one and leaves the rest), **Many at once** (filter a population — every QB at slot 1, every rookie receiver — preview what it would do per player, then write it as one override each so any single row can still be dropped), **League knobs** (the constants behind the model, `k_scale` among them), **Compare scenarios** ("my rankings" against "consensus", as one table of who moved and by how much). The editors themselves live beside the numbers they change: a team's volume on **Team → 📈 Team volume**, a room's ratings on **Team → 🎛️ The rooms**, one man's on **Player → ✎ Adjust**, one afternoon's on **Week → 🎛️ Adjust this game**. Naming, opening and starting a scenario are in the sidebar of every page, and every edit is written to disk as it is made. |
| **Exports** | any table, current filters and adjustments applied, at full precision — as a formatted workbook, a CSV, or a live Google Sheet. What leaves is what is on screen. |

No page computes a projected number. `app/ui.py` is the only door from a page to the engine, which is
what keeps two pages from quietly disagreeing. Formatting goes through the same door: every table names
and rounds its columns from one dictionary — a season total to one decimal because the second is noise
the model does not have, a share as a percentage, a yard as a whole number — so two pages cannot show
the same number to different precision, and a column called `own_weight` is headed *weight on his own
record* wherever it appears. An override is always visible where the number it changed
is: ✏️ on the depth chart, an `edits` column on any frame, and a `base → now` line with a reset beside it
at the top of the player it touches — with the engine's *current* number beside it, so a scenario typed in
August says so when the estimator has since come round on its own.

### How it is read: a list, a panel, and the ✎ beside the number

The first version of this app put everything in tables, some of them forty-four columns wide. Every
number was there and none of it was legible: the right-hand half of a wide table is a half nobody has
ever read, and comparing two players meant scrolling sideways and holding a figure in your head. The
whole interface is now built from six shared components in `app/ui.py`, so a page composes rather than
formats, and two pages cannot disagree about what a share looks like:

- **`pick_from` + a panel.** The default shape everywhere: a list narrow enough to read down — six or
  eight columns — and everything else about the row you picked in a bordered panel beside it. The list
  returns the whole row, so the panel is the same object the engine produced.
- **`tiles`.** The headline numbers as a wrapping row of bordered cards: what it is, what it is, and what
  it is *against*. They replaced `st.metric` rows, which reserve a column's width whatever they hold.
- **`meters`.** The component the redesign turns on. A number drawn inside the range it lives in, with
  the comparisons that justify it — his own record, his job's prior, the position's median, the league
  mean, `1.00` where 1.00 is the claim — as ticks on the same track. *Where an estimate landed between
  the man and the role* becomes a picture instead of a subtraction.
- **`chips`.** The things that are true about a row (rookie, changed team, questionable, carries an
  override), drawn only when true rather than as a grey column of falses.
- **`section` + `note`.** Roughly 130 paragraphs of reasoning used to print as captions under the
  tables, which made the app an essay with data below the fold. Not a word was cut: every one is folded
  behind the small grey ⓘ beside its heading, one click from the number it explains.
- **`focus_table`.** Where a table is still the right answer, it shows the columns that answer the
  page's question and puts the rest one click away under *⋯ every column*. Nothing is hidden and the
  export still writes the whole frame.

Four surfaces are **still grids, on purpose**, each with a comment in the source saying why: the Week
page's week × player matrix and its team × week grid (seventeen columns across is the shape of a
season, and no panel says *three of their first five are away* as plainly as reading across a row), and
the roster and per-game typing sheets (a session that is really about typing wants a spreadsheet).

Overrides moved with the reading. Instead of a separate form, there is a ✎ beside the number itself —
a popover carrying the engine's own estimate, the record behind it, the one-click named values (*his own
record*, *his job's prior*, *position median*, *every game*) and an exact box. The number you are
arguing with never leaves the screen while you argue with it.

### Several of his numbers at once, from wherever you are

The ✎ popover is the right tool for *one* number, and the wrong one for four. Each write is a new
scenario, so it is an engine run and a full redraw; the popover closes on the write it makes; and one
popover edits one metric. Disagreeing with four things about one player was therefore four popovers, four
runs and a lost scroll position — and the per-game version of the same disagreement lived on a different
page again, so mostly it did not get made.

So there is a second editor, and it is **the same one everywhere**. `✎ Adjust` sits in every player panel
in the app — Home, Player, Team rooms, Availability, History, Edits — and week-scoped on the Week page's
board and box-score panels, where it opens already pointed at the week you were reading. It is also in
the sidebar (*✎ Adjust a player*), so a name you think of is one field away from wherever you happen to be.
Every one of them raises the same dialog over the page you were on, so nothing is lost by editing. On the
Player page it is a tab of its own, since that is the place you go *to* edit rather than stumble into it.

What it is: one row per rating, the engine's number and what the projection is running on beside each,
his own record and his job's prior beside that, and three empty columns on the end — `set to`, `x by`,
`drop`. Type into as many rows as are wrong and apply them together. Four things follow from that shape:

- **Nothing is computed until Apply.** A pending edit is a typed cell, not a scenario, so the count and
  the summary line — *3 changes: target share =0.28 · yards per target x1.05 · drop catch rate* — are
  visible and reversible before anything runs. Then one run instead of four, and **Discard** forgets it
  all without touching the scenario. Every sheet in the app now batches this way, the roster and per-game
  ones included; they used to write on each cell you left.
- **Season and weeks are one control, not two pages.** *Applies to* switches between the season and any
  set of weeks. Pick three weeks and one typed row becomes three overrides, each recorded against **its
  own week's** engine number — which is what makes "he is on a pitch count in December" a single pass.
  The week scope offers only what a week can carry: `p_play` is there, the depth slot and the games count
  are not, because neither means anything about one afternoon.
- **`x by` is kept as a multiplier.** Baked into a number it would stop tracking the engine; as a factor
  a re-estimate still moves under it. A factor of exactly 1.0, and a value equal to the number already
  there, are dropped rather than written — an override that changes nothing is a row in the edit list
  that will be mistaken for one that does.
- **Undo is in the same pass.** `drop` per row, so unpicking four edits is also one Apply. Dropping a
  season edit leaves the week edits on that same rating alone — they are separate claims, and
  `Scenario.clear` reads a missing week as a wildcard, which is right for *clear this player* and wrong
  for one row of a sheet.

It is deliberately **not** new evidence: every column in it is one `ratings` already produced. When the
question is *should* this number move rather than *make* it move, the ✎ popover with the record and the
league percentile beside it, and the 🔬 What if tab that runs the season with a candidate in it, are both
still there — and both one click from this list.

### A whole roster, one row a man

Most disagreements are found while reading down a depth chart rather than while looking at one player:
*he is not the third receiver, and the man above him is not playing seventeen games* is two edits to two
players in the same glance. **Team → 🎛️ The rooms** is that shape — one room at a time, players down the
side in slot order, the numbers that decide their projection across the top, with his projected total and
points per game sitting between his name and his ratings so an edit is read against the thing it is
changing. **Week → 🎛️ Adjust this game** is the same surface for one afternoon.

Thirty-nine editable fields do not fit on a screen and would not be read if they did, so the columns come
in sets. The default set is position-aware — the same shortlist the What-if tab offers, ordered by how
much each number moves a season, unioned across the positions on screen — and the named sets follow the
engine's own stages, which is the order a disagreement is actually settled in: is he playing, how much of
the offence is he on the field for, what slice does he take, how well does he do with it. The slot and the
games are in every set, because they are what a roster review reaches for first.

Three things it is careful about, all of which come from a grid being a different surface from a knob:

- **The base is the engine's number, not the number on screen.** On a roster grid the value in front of
  you can be the consequence of somebody else's edit — promote a receiver and the man he passed is
  showing a slot nobody typed. So `base` is read off the baseline run, and a receiver pushed to 2 never
  has 2 recorded as what the published chart said.
- **A blank cell is a column that does not apply, not one to fill in.** A completion percentage in a
  receiving room is dropped entirely when only receivers are on screen, and left blank when a
  quarterback is beside them. Typing in a blank one is ignored rather than recorded, because the engine
  has nothing to change and would only report the edit unapplied.
- **The slot is still an ordering.** It is editable here — and only here, among the grids — because the
  room is on screen in slot order, so the renumbering is something you watch happen rather than
  something that happens to you. It is the same override as the depth chart's, re-priced the same way.

Since it writes ordinary overrides, everything else already holds: the pencil column says who carries
one, the edit list drops any of them, a share stays zero-sum so the teammates' totals move with it on the
next rerun, and 🔬 What if is still the place to see what one number costs before writing it down.

### Off the roster entirely

Our sources are a published depth chart and nflverse's seasonal roster file, and neither of them ever
collapses to a final 53: for a few weeks after cutdown day both still list men who are on another team or
on nobody's. Zeroing such a man's games is the wrong edit — it leaves him standing in his slot with the
room's pool still divided around him, which is right for an injury and wrong for a player who is not
there. So **Team → 🧭 Depth chart → Take somebody off this roster** writes one override, `on_roster = 0`,
and it is the only edit in the app that removes a *row* rather than changing a number in one.

It is applied first, before the chart is settled and before anything is estimated off it, which is what
makes the rest follow by itself: the room renumbers around the hole, the men who moved up are re-priced
off the slot they moved *to* rather than handed a promotion at backup prices, normalisation divides his
share of every pool among the players who are left, and nothing downstream — board, weekly, exports,
simulations — has a row to count him in. Putting him back is dropping the edit, so the chart follows the
published order again. Two things it refuses rather than guessing at: a roster spot cannot be multiplied,
and the last man in a room cannot be released, because a pool with nobody to divide it is not a
projection. Both refusals are reported with their reason in the provenance's `roster` stage, and the same
button is on a player's own page (**Player → ✎ Adjust**) for the case where you notice while reading him.

### One afternoon, not the season

Most disagreements about a projection are about one game. He is banged up this week; the starter is being
rested with the division wrapped up; this is the week that pass rush eats them. None of those is a
season-long claim, and typing one in as a season-long claim is how a projection ends up wrong in sixteen
games in order to be right in one. So **Week → 🎛️ Adjust this game** takes an override with a week on it,
for the offence's own volume and for any per-game player field — `p_play`, every share, every rate. The
one player field that is not per-game is `expected_games`, which has a single value for the season by
definition; a week named on it is reported at a `season grain` stage rather than silently obeyed.

Where the edit lands is the whole point. It is applied *inside* `opportunity()`, before the team's pool is
divided rather than after it, so ruling a starter out for one week hands his carries to the room, promotes
the next man in the quarterback queue for that week alone, and leaves the team running the plays it was
projected to run. A rate edit lands between the join and the stat line, so the used rates are recomputed
from it. Nothing is typed over a total.

The season then follows on its own, because the season is the sum of the seventeen games and one of them
has moved — the same board diff every other page reads, so an edit worth a tenth of a point over a year
reads as one. The pool audit on the box-score tab is the check that this stayed true: a counted pool has
to reconcile to rounding whatever has been typed in, and it would not if an edit were ever applied after
the division. Yards and completions are a count times a rate and are deliberately not forced to match.

### Moving a man on the chart

Every other override sets a number. A depth slot is an *ordering*, so it is applied differently and
before anything else: the room is rebuilt around the move — the men asked for at the slots asked for,
everybody else falling into what is left in the order the published chart had them — and `slot_bucket`
and `avail_slot` are re-derived, because those are the rows every prior, the survival curve and the
quarterback queue are read off. So a promoted backup is *re-priced as a starter*, his expected games and
his share of the dropbacks both re-estimated off the new slot, rather than being handed a starter's share
while still being priced as a backup. It is reported as its own `depth` stage in the provenance, with two
slots where the other rows have two values. `multiply` is refused rather than reinterpreted: there is no
sensible reading of "twice as deep on the chart".

### Every override is made on the record

Wherever a number can be changed, what is known about it is on the same screen: the engine's estimate and
what the projection actually ran on, the player's own record season by season with the opportunity behind
it, what the job is worth to everybody who has held that position and depth slot, how much of the estimate
is the man rather than the role (`own_weight`), where he sits in the position's quartiles, and the rest of
his room on the same metric — because a share is zero-sum and raising his lowers theirs. Named one-click
sets (*his own record*, *his job's prior*, *position median*, *every game*) sit beside the exact-value
knob, so the fast way to override is also the grounded one. `estimate.season_history` and
`roster.attendance_history` are the two engine-side sources this rests on: the record, unweighted and
unshrunk, as opposed to the estimate of it.

### One number, and everything it moves

Typing 0.28 over 0.24 is easy; knowing what it costs the four men behind him is not. **Team → 🔬 What if**
is that question answered before the edit is written down, and it is the surface to reach for rather than
a grid when the disagreement is about a *rating* rather than a total.

It opens on the offence itself: pick a pool — targets, carries, dropbacks, air yards, red-zone work,
touchdowns, snaps, routes, any of the sixteen the engine divides — and see who is claiming it, as a bar
per man and as a table with each one's share of the team beside his projected line. Above it, five numbers
say what there is to divide: per game, over the season, how many men claim it, how much of it the roster
claimed *before* normalisation, and the factor it was scaled by. A pool nobody over-claimed and a pool
scaled by 0.87 are different rooms to argue about.

Then one number, moved. Each position offers the dozen knobs that actually move a projection rather than
all thirty-odd — and each is labelled with what *kind* of number it is, because that decides who else
moves: ⚖️ a slice of a pool that sums to one (zero-sum: his room pays for what he gains), 🕒 how much of
the offence he is on the field for (not divided, so nobody else moves), 📅 availability (scales every count
he has), 🎯 a rate (his own, and nobody else's line rides on it). Move the percentage slider — or type the
exact number — and the season is **run again with that one number changed**, which is the only honest way
to answer it: normalisation, the quarterback queue, the touchdown pools and the per-game chain all sit
between a share and a projected point, so a number worked out on the page would be a different model from
the one the board came from.

What comes back is four cards (him, his room, the offence, his rank at the position), a sentence that
reads the three of them together, his own line before and after stat by stat, every teammate who moved
with the size of the move, the pool drawn twice as grouped bars — what each man has now against what he
would have — and the offence's own totals. Nothing is committed until **Apply**, which writes exactly one
override with the engine's number as its base; moving the control back to 100% throws the candidate away.
Two runs is about a second warm, which is what makes this a live preview rather than a report.

It is also where three things the engine does become visible instead of surprising: a share edit conserves
the pool to about 1e-14, so the honest reading is *whether the losses are the ones you meant to cause*; a
rate edit moves nobody at all; and **touchdowns are their own pool**, so handing a man targets does not
hand him touchdowns — `rec_td_share` does.

## Exports

CSV and a formatted multi-sheet workbook need nothing. The workbook carries a README sheet naming the
scenario and the moment, one sheet per table, frozen headers and a number format per column family. All
three destinations write **full precision** — the app rounds for display, a file is for computing with.

Google Sheets needs a service-account JSON. It is looked for at `$NFLSP_GSPREAD_JSON`, else
`~/.config/nflsp/service-account.json` — both deliberately outside this tree, so no plausible `git add`
can commit a private key. Without one the page says exactly what is missing and offers the other two
formats; it degrades rather than failing.

```bash
pip install gspread google-auth
export NFLSP_GSPREAD_JSON=~/.config/nflsp/service-account.json
```

The service account owns nothing a human can see, so put your own address in "share with" or the
spreadsheet will exist and be invisible.

## Command line

Each model module runs on its own and reports what it fitted:

```bash
python -m src.data.refresh --check              # what is on disk, and how old
python -m src.model.priors --fit                # refit priors, shrinkage, rookie curves
python -m src.model.team                        # team environment and the 2026 per-game chain
python -m src.model.roster                      # participation and expected games
python -m src.model.compose                     # the full board, composed and normalised
python -m src.model.simulate --fit              # fit dispersion; --calibrate to score the ranges
python -m src.model.backtest                    # held-out seasons, ablations, calibration, intervals
```

`backtest` writes `data/fitted/backtest_summary.parquet` and `backtest_players.parquet`, which is what
**History → 📉 Backtest** reads. It is minutes per variant, so the page reports the last run and how old it
is rather than recomputing.

## Layout

```
app/            Home.py, ui.py, pages/1_League, 2_Team, 3_Player, 4_Week,
                5_Availability, 6_History, 7_Edits, 8_Exports
src/config.py   every path, season and tunable; `Settings` is what the adjustment layer edits
src/data/       lake reader, refresh, depth charts, history aggregates
src/model/      priors, estimate, blend, team, roster, opportunity, efficiency,
                compose, overrides, simulate, backtest
src/export/     CSV, workbook and Google Sheets writers
scripts/        smoke_app.py (every page in a real session), diag_dispersion.py
tests/          pytest, including the leakage guards on the backtest
```

## Development

`main` is what runs; work happens on `develop`.

```bash
python -m pytest -q            # ~300 tests, a few minutes: several fit real models
python scripts/smoke_app.py    # every page in a real Streamlit session, sixteen passes
```

The smoke harness is the one that matters before a commit that touches the app: it runs each page with a
genuine session, then with a scenario live, then clicks a reset and types a multiplier, then turns the
Monte Carlo on, then switches the board to a stat line, opens up a depth chart and turns the knob beside
a rating, then drives a one-click override and checks the evidence behind it is on screen, and finally
moves a man on a depth chart and checks the projection re-prices him, and finally drives the two edits
that are not one number at one point — a filtered bulk edit written across a whole position, and a return
date turned into a games count — then switches the team sheet through every column set and filters it to
one position, and finally drives the What-if tab and asserts the sign structure a share
edit must have, so the override layer, the ranges and the reading surfaces are proved wired rather than
merely importable. The depth pass exists because a slot is an ordering: a broken version of it is not an
exception, it is a chart that still reads 1, 2, 3 with the projection underneath unchanged. The population
pass exists for the same reason one level up: a filter that catches nobody still renders a preview and an
Apply button. The team-sheet pass exists because each column set is a different frame with a different
`column_config`, so a set naming a column the frame does not carry fails on a tab the baseline pass never
opened. And the What-if pass exists because a preview that quietly did arithmetic on the page
instead of running the season again would render perfectly while being wrong about the only thing it is
there to say: him up, his room down, the offence barely moved.
