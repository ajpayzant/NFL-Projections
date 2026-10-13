# NFL Player Game Projections

Weekly projections for every NFL QB, RB, WR and TE (passing, rushing and receiving lines, touchdowns,
fantasy points) and market-free game lines, in the format of the NHL and NBA game-projection tools:
Slate, Game, Depth charts, Player, Edges, Fantasy, Results, How it works.

The player projections come from the projection engine (`~/nfl-projection-system`, a private repo):
its football chain is run on this tool's expected lineup and corrected here. See HOW_IT_WORKS.md.

## Run

The weekly refresh runs in the engine's environment (it imports the engine):

```
refresh.bat                        # live.py (refresh, who plays, project, price) -> tracking snapshot -> publish
%USERPROFILE%\nfl-projection-system\.venv\Scripts\python live.py --no-refresh   # re-project only
run_app.bat                        # the app, http://localhost:8690
```

Fitting (once a season, or after an engine change):

```
python availability.py     # who plays (walk-forward evaluation, then fit on everything)
python correct.py          # direct models, blend weights, distributions (needs research/audit_engine_pred.parquet)
python gamelines.py        # market-free game lines
```

## Files

| File | Role |
|---|---|
| `availability.py` | P(plays) for every rostered QB / RB / WR / TE and the QB starter (early-week and report models) |
| `correct.py` | direct LightGBM per statistic, blended with the engine (weights chosen walk-forward) |
| `dist.py` | outcome distributions for props, touchdowns, fantasy points and game margins / totals |
| `gamelines.py` | market-free margin and total (ridge on team strength) |
| `live.py` | the weekly pipeline |
| `adjust.py` | the Depth charts edits: storage (gist or file) and their effect on the projections |
| `odds.py`, `edges.py`, `tracking.py` | sportsbook odds (manual refresh only), edges, Results grading |
| `cloud.py`, `publish.py` | the data release the Streamlit Cloud app reads |
| `app.py` | the Streamlit app |
| `research/` | the audit, blend, distribution and game-line studies |

## Streamlit Cloud

The app runs on Streamlit Cloud from this repo with only `requirements.txt` (no engine). The refresh
runs on the machine with the engine and uploads its outputs to this repo's release tagged `data`
(`publish.py`); the app checks for newer files every 10 minutes (`cloud.py`).

| Setting | Value |
|---|---|
| Repository / branch / main file | `ajpayzant/NFL-Game-Projections` / `main` / `app.py` |
| Python version | 3.13 or 3.14 |

Secrets (TOML):

```toml
odds_api_key = "..."     # the Edges tab's Refresh odds button
gist_id      = "..."     # optional: shared Depth charts edits that survive restarts (file nfl_lineups.json)
github_token = "..."     # optional: fine-grained token with Gists read and write
```
