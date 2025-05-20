import requests
import json
import time
from datetime import datetime, timedelta

API_TOKEN = "hYOCrQiS5sskBAgdNNFR27ORwxWCIZTmdk4k9J_-xeBZcsC87zE"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

MATCHES_URL = "https://api.pandascore.co/csgo/matches"
STATS_URL = "https://api.pandascore.co/csgo/teams/{team_id}/stats"

CACHE_FILE = "team_stats_cache.json"
MATCHES_FILE = "matchups.json"

# Load existing cache or initialize
try:
    with open(CACHE_FILE, "r") as f:
        team_stats_cache = json.load(f)
except FileNotFoundError:
    team_stats_cache = {}

# Dynamically use the list of cached teams as "top-tier"
TOP_TIER_TEAMS = list(team_stats_cache.keys())
print(f"Using {len(TOP_TIER_TEAMS)} cached top-tier teams.")

# Helper to check if at least one team is top-tier
def has_top_tier_team(team1, team2):
    return team1 in TOP_TIER_TEAMS or team2 in TOP_TIER_TEAMS

# Fetch and cache team stats
def get_team_stats(team_id, name, api_calls, max_api_calls):
    if name in team_stats_cache:
        return team_stats_cache[name], api_calls

    if api_calls >= max_api_calls:
        print("⚠️ API call limit reached while fetching team stats.")
        return None, api_calls

    url = STATS_URL.format(team_id=team_id)
    r = requests.get(url, headers=HEADERS)
    api_calls += 1
    if r.status_code == 200:
        data = r.json()
        team_stats_cache[name] = data
        return data, api_calls
    return None, api_calls

# Fetch recent matches
def fetch_recent_matches(pages=1100):
    all_matches = []
    now = datetime.utcnow()
    cutoff_date = now - timedelta(days=365)
    api_calls = 0
    max_api_calls = 995

    for page in range(1, pages + 1):
        print(f"Fetching page {page}")
        if api_calls >= max_api_calls:
            print("⚠️ API call limit reached. Stopping early.")
            break

        params = {
            "per_page": 50,
            "page": page,
            "sort": "-begin_at"
        }
        r = requests.get(MATCHES_URL, headers=HEADERS, params=params)
        api_calls += 1
        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text}")
            break
        matches = r.json()

        for match in matches:
            try:
                if not match["opponents"] or len(match["opponents"]) < 2:
                    continue

                begin_at = match.get("begin_at")
                if not begin_at:
                    continue
                date = datetime.fromisoformat(begin_at.replace("Z", "+00:00"))
                if date < cutoff_date:
                    continue

                team1 = match["opponents"][0]["opponent"]["name"]
                team2 = match["opponents"][1]["opponent"]["name"]
                if not has_top_tier_team(team1, team2):
                    continue

                team1_id = match["opponents"][0]["opponent"]["id"]
                team2_id = match["opponents"][1]["opponent"]["id"]

                team1_stats, api_calls = get_team_stats(team1_id, team1, api_calls, max_api_calls)
                team2_stats, api_calls = get_team_stats(team2_id, team2, api_calls, max_api_calls)

                if not team1_stats or not team2_stats:
                    continue

                match_entry = {
                    "team_1": team1,
                    "team_2": team2,
                    "team_1_stats": team1_stats,
                    "team_2_stats": team2_stats,
                    "winner": match["winner"]["name"] if match["winner"] else None,
                    "date": match["begin_at"]
                }
                all_matches.append(match_entry)
                print(f"✅ {team1} vs {team2} @ {match['begin_at']}")
                time.sleep(1)
            except Exception as e:
                print(f"Error processing match: {e}")
                continue

    with open(MATCHES_FILE, "w") as f:
        json.dump(all_matches, f, indent=2)

    with open(CACHE_FILE, "w") as f:
        json.dump(team_stats_cache, f, indent=2)

    print(f"✅ Saved {len(all_matches)} matches to {MATCHES_FILE}")
    print(f"📦 Cached {len(team_stats_cache)} team stats to {CACHE_FILE}")

if __name__ == "__main__":
    fetch_recent_matches(pages=1100)
