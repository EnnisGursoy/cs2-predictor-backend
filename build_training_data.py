import json
import pandas as pd
import numpy as np

# Load team stats
with open("team_stats_cache.json", "r") as f:
    team_stats = json.load(f)

# Load matchups
with open("matchups.json", "r") as f:
    matchups = json.load(f)

data = []
skipped = 0

def get_stat(team, key):
    return team_stats.get(team, {}).get(key, np.nan)

def create_feature_row(t1, t2, winner):
    row = {}

    # Define keys we want to use
    keys = ["win_rate", "avg_score", "games_played"]
    for k in keys:
        v1 = get_stat(t1, k)
        v2 = get_stat(t2, k)
        row[f"{k}_diff"] = v1 - v2
        row[f"{k}_avg"] = (v1 + v2) / 2
        row[f"{k}_abs_diff"] = abs(v1 - v2)

    row["winrate_ratio"] = get_stat(t1, "win_rate") / (get_stat(t2, "win_rate") + 1e-5)
    row["avgscore_ratio"] = get_stat(t1, "avg_score") / (get_stat(t2, "avg_score") + 1e-5)
    row["team1_higher_winrate"] = int(get_stat(t1, "win_rate") > get_stat(t2, "win_rate"))

    row["match_winner"] = 0 if winner == t1 else 1
    return row

used = 0
for match in matchups:
    t1 = match.get("team1")
    t2 = match.get("team2")
    winner = match.get("winner")

    # Skip invalid match records
    if not t1 or not t2 or not winner or winner not in [t1, t2]:
        skipped += 1
        continue

    # Only skip if both teams are unknown
    if t1 not in team_stats and t2 not in team_stats:
        skipped += 1
        continue

    r1 = create_feature_row(t1, t2, winner)
   # r2 = create_feature_row(t2, t1, winner)

    data.append(r1)
    #data.append(r2)
    used += 1

# Convert to DataFrame and handle missing data in training script
df = pd.DataFrame(data)

print(f"✅ Matches used (doubled): {used * 2}")
print(f"⛔ Skipped matches: {skipped}")
print("🏷️ Class distribution:")
print(df['match_winner'].value_counts(normalize=True))
print("📊 Feature columns:", list(df.columns))

df.to_csv("training_data.csv", index=False)
print("✅ Saved: training_data.csv")
