# ============================
# ✅ Flask: app.py (backend)
# ============================
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "ensemble_model.pkl"
SCALER_PATH = "scaler.pkl"
SELECTOR_PATH = "feature_selector.pkl"
FEATURE_ORDER_PATH = "feature_order.pkl"
CACHE_PATH = "team_stats_cache.json"

# Load model and tools
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)
feature_order = joblib.load(FEATURE_ORDER_PATH)

# Load team stats
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        team_stats_cache = json.load(f)
else:
    team_stats_cache = {}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        team1_name = data["team1"]
        team2_name = data["team2"]

        team1_stats = team_stats_cache.get(team1_name)
        team2_stats = team_stats_cache.get(team2_name)

        if not team1_stats or not team2_stats:
            return jsonify({"error": "Missing stats for one or both teams"}), 400

        def diff(a, b): return a - b
        def avg(a, b): return (a + b) / 2
        def abs_diff(a, b): return abs(a - b)

        features = {
            "win_rate_diff": diff(team1_stats["win_rate"], team2_stats["win_rate"]),
            "win_rate_avg": avg(team1_stats["win_rate"], team2_stats["win_rate"]),
            "win_rate_abs_diff": abs_diff(team1_stats["win_rate"], team2_stats["win_rate"]),
            "avg_score_diff": diff(team1_stats["avg_score"], team2_stats["avg_score"]),
            "avg_score_avg": avg(team1_stats["avg_score"], team2_stats["avg_score"]),
            "avg_score_abs_diff": abs_diff(team1_stats["avg_score"], team2_stats["avg_score"]),
            "games_played_diff": diff(team1_stats["games_played"], team2_stats["games_played"]),
            "games_played_avg": avg(team1_stats["games_played"], team2_stats["games_played"]),
            "games_played_abs_diff": abs_diff(team1_stats["games_played"], team2_stats["games_played"]),
            "team1_higher_winrate": 1 if team1_stats["win_rate"] > team2_stats["win_rate"] else 0,
            "winrate_ratio": team1_stats["win_rate"] / (team2_stats["win_rate"] + 1e-5),
            "avgscore_ratio": team1_stats["avg_score"] / (team2_stats["avg_score"] + 1e-5)
        }

        df = pd.DataFrame([features])[feature_order]
        X_scaled = scaler.transform(df)
        X_selected = selector.transform(X_scaled)

        pred = model.predict(X_selected)[0]
        proba = model.predict_proba(X_selected)[0]
        confidence = float(max(proba))

        predicted_winner = team1_name if pred == 0 else team2_name

        return jsonify({
            "predicted_winner": predicted_winner,
            "confidence": round(confidence, 4),
            "team1_stats": team1_stats,
            "team2_stats": team2_stats
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 400

@app.route("/teams", methods=["GET"])
def get_teams():
    print("Loaded teams from cache:", len(team_stats_cache))
    filtered = sorted(
        [team for team, stats in team_stats_cache.items() if stats.get("games_played", 0) > 5],
        key=lambda t: team_stats_cache[t]["games_played"],
        reverse=True
    )
    return jsonify(filtered)

@app.route("/team_stats/<team>")
def get_team_stats(team):
    stats = team_stats_cache.get(team)
    if not stats:
        return jsonify({"error": "Team not found"}), 404
    return jsonify({"team": team, **stats})

@app.route("/recent_matches/<team>")
def recent_matches(team):
    # Optional route — return empty if unsupported
    return jsonify([])

@app.route("/last_match/<team1>/<team2>")
def last_match(team1, team2):
    # Optional route — return blank if unsupported
    return jsonify({"team1": team1, "team2": team2, "winner": team1})

if __name__ == "__main__":
    print("Launching Flask on http://localhost:5000")
    app.run(debug=True)
