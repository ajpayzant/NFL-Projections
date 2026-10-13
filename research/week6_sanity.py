import pickle, sys
import polars as pl
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
import availability as A
C.engine_on_path()
from src.data import storage
from src.projections import week
models = pickle.load(open(A.MODEL_FILE, "rb"))
f = A.frame([2026])
pred = A.predict(models, f.filter(pl.col("week") == 6))
lu = A.lineup(pred)
print("lineup per team:", round(lu.height / lu["team"].n_unique(), 1), "| QBs:", lu.join(pred.filter(pl.col("pos") == "QB"), on=["game_id", "team", "player_id"], how="semi").height)
fc = week.project(2026, 6, lineup=lu.select("team", "player_id"))
avg = storage.load("player_games", layer="processed", seasons=[2026]).group_by("player_id").agg(pl.len().alias("gp"), *[pl.col(c).mean().alias(f"a_{c}") for c in ("passing_yards", "rushing_yards", "carries", "targets", "receiving_yards", "fantasy_points_ppr")])
for pop, cols in (("quarterback", ["passing_yards", "rushing_yards"]), ("running_back", ["carries", "rushing_yards", "targets"]), ("receiver", ["targets", "receiving_yards"])):
    r = fc.players[pop].join(avg, on="player_id").filter(pl.col("gp") >= 3)
    print(pop, {c: (round(r[f"p_{c}_chain"].mean(), 1), round(r[f"a_{c}"].mean(), 1)) for c in cols}, "n", r.height)
q = fc.players["quarterback"].sort("p_passing_yards_chain", descending=True)
print(q.select("player_name", "team", "p_dropback_share", "p_passing_yards_chain", "p_passing_tds_chain").head(8).to_pandas().round(2).to_string(index=False))
rb = fc.players["running_back"].sort("p_rushing_yards_chain", descending=True)
print(rb.select("player_name", "team", "p_carries_chain", "p_rushing_yards_chain", "p_targets_chain").head(8).to_pandas().round(1).to_string(index=False))
wr = fc.players["receiver"].sort("p_receiving_yards_chain", descending=True)
print(wr.select("player_name", "team", "p_targets_chain", "p_receptions_chain", "p_receiving_yards_chain").head(8).to_pandas().round(1).to_string(index=False))
out = pred.filter((pl.col("p_play") < 0.5) & (pl.col("pos") != "QB") & (pl.col("prev_played") == 1)).sort("p_play").select("player_name", "team", "pos", "p_play", "inj_out", "inj_questionable", "depth", "prev_snap")
print("left out though they played last game:", out.height); print(out.head(10).to_pandas().round(2).to_string(index=False))
print("\n--- team sums, week 6")
sk = pl.concat([fc.players["running_back"].select("team", "player_id", "p_targets_chain", "p_carries_chain"),
                fc.players["receiver"].select("team", "player_id", "p_targets_chain").with_columns(pl.lit(0.0).alias("p_carries_chain"))], how="diagonal")
ts = sk.group_by("team").agg(pl.col("p_targets_chain").sum().alias("tgt"), pl.col("p_carries_chain").sum().alias("car"), pl.len().alias("players"))
tv = fc.team.select("team", "p_pass_attempts", "p_designed_rushes")
print(ts.join(tv, on="team").select(pl.col("tgt").mean(), pl.col("p_pass_attempts").mean(), pl.col("car").mean(), pl.col("p_designed_rushes").mean(), pl.col("players").mean()).to_pandas().round(2).to_string(index=False))
tg = storage.load("team_games", layer="processed", seasons=[2026])
print("2026 team avg targets", round(tg["targets"].mean(), 1), "carries", round(tg["carries"].mean(), 1))
u = storage.load("player_usage", layer="processed", seasons=[2026]).filter(pl.col("projected_position").is_in(["RB", "WR", "TE"]))
print("2026 avg RB/WR/TE targets per team-game", round(u.group_by("game_id", "team").agg(pl.col("targets").sum())["targets"].mean(), 1), "RB carries", round(u.filter(pl.col("projected_position") == "RB").group_by("game_id", "team").agg(pl.col("carries").sum())["carries"].mean(), 1))
