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
.venv/Scripts/streamlit run app/Home.py
```

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

The **Diagnostics** page reports the age of every table, and separately whether `processed/` has kept up
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

The ablations are in the same table on the Diagnostics page: each turns off one node, and every node
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
| **Home / Rankings** | the board. Filters, tiers, positional ranks, volatility when ranges are on. |
| **Player** | one player end to end: weekly path, the shares and rates behind it, and what each estimate was built from. |
| **Team** | one offence as a whole: the pools, who they were divided among, and whether they add up. |
| **Matchups** | the 2026 schedule through the projection — who each team faces, and when the environment helps or hurts. |
| **Adjustments** | the override layer. Edit a team pool, a player's share, a rate or a league setting, for one week or all of them; every page downstream shows the edited number and the diff against the engine's own answer. |
| **Exports** | any table, current filters and adjustments applied, at full precision. |
| **Diagnostics** | the evidence: backtest, calibration, interval coverage, share-sum audits, data freshness. |

No page computes a projected number. `app/ui.py` is the only door from a page to the engine, which is
what keeps two pages from quietly disagreeing.

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
the Diagnostics page reads. It is minutes per variant, so the page reports the last run and how old it
is rather than recomputing.

## Layout

```
app/            Home.py, ui.py, pages/1_Player … 6_Diagnostics
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
python -m pytest -q            # ~250 tests, a few minutes: several fit real models
python scripts/smoke_app.py    # every page in a real Streamlit session, four passes
```

The smoke harness is the one that matters before a commit that touches the app: it runs each page with a
genuine session, then with a scenario live, then clicks a reset and types a multiplier, then turns the
Monte Carlo on — so the override layer and the ranges are proved wired rather than merely importable.
