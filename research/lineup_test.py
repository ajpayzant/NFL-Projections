import polars as pl
from src.data import storage
from src.projections import week
pu = storage.load("player_usage", layer="processed", seasons=[2026])
pg = storage.load("passer_games", layer="processed", seasons=[2026])
skill = pu.filter((pl.col("week") == 5) & pl.col("projected_position").is_in(["RB", "WR", "TE"]) & (pl.col("offense_snaps") > 0)).select("team", "player_id")
qb = pg.filter(pl.col("week") == 5).sort("dropbacks", descending=True).group_by("team").first().select("team", "player_id")
lineup = pl.concat([skill, qb]).unique()
f = week.project(2026, 6, lineup=lineup)
q = f.players["quarterback"]
print("QB cols:", [c for c in q.columns if "yard" in c or "attempt" in c])
avg = storage.load("player_games", layer="processed", seasons=[2026]).group_by("player_id").agg(pl.len().alias("gp"), pl.col("passing_yards").mean().alias("pa"), pl.col("receiving_yards").mean().alias("ra"), pl.col("targets").mean().alias("ta"), pl.col("rushing_yards").mean().alias("rua"))
py = next(c for c in q.columns if c.startswith("p_passing_yards"))
qa = q.join(avg, on="player_id").filter(pl.col("gp") >= 3)
print(f"QB rows {q.height}, dropback share {q['p_dropback_share'].mean():.3f}; passing yds proj {qa[py].mean():.1f} vs 2026 avg {qa['pa'].mean():.1f}")
for pop, col, a in (("receiver", "p_receiving_yards_chain", "ra"), ("receiver", "p_targets_chain", "ta"), ("running_back", "p_rushing_yards_chain", "rua")):
    r = f.players[pop].join(avg, on="player_id").filter(pl.col("gp") >= 3)
    print(f"{pop} {col}: proj {r[col].mean():.2f} vs avg {r[a].mean():.2f} (n={r.height})")
print(q.sort(py, descending=True).select("player_name", "team", "p_dropback_share", py).head(6).to_pandas().round(2).to_string(index=False))
