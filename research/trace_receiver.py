import polars as pl
from src.data import storage
from src.projections import week
pu = storage.load("player_usage", layer="processed", seasons=[2026])
pg = storage.load("passer_games", layer="processed", seasons=[2026])
skill = pu.filter((pl.col("week") == 5) & pl.col("projected_position").is_in(["RB", "WR", "TE"]) & (pl.col("offense_snaps") > 0)).select("team", "player_id")
qb = pg.filter(pl.col("week") == 5).sort("dropbacks", descending=True).group_by("team").first().select("team", "player_id")
f = week.project(2026, 6, lineup=pl.concat([skill, qb]).unique())
r = f.players["receiver"]
p_cols = [c for c in r.columns if c.startswith("p_") and not c.endswith("_raw")]
for name in [n for n in ("CeeDee Lamb", "Jaxon Smith-Njigba", "Trey McBride") if n in r["player_name"].to_list()]:
    row = r.filter(pl.col("player_name") == name)
    print(name, {c: round(row[c][0], 3) for c in p_cols if row[c][0] is not None})
    act = pu.filter(pl.col("player_name") == name).select("week", "snap_share", "targets", "target_share", "receiving_yards", "team_targets" if "team_targets" in pu.columns else "targets")
    print(act.to_pandas().round(3).to_string(index=False))
tv = f.team.filter(pl.col("team") == "LA").select([c for c in f.team.columns if c.startswith("p_")])
print("LA team:", {c: round(tv[c][0], 2) for c in tv.columns})
