import streamlit as st
import os
import requests
import datetime

st.set_page_config(page_title="NHL Predictor", page_icon="🏒")

# Environment variables (configured in Railway)
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST")
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}
BASE_URL = f"https://{RAPIDAPI_HOST}"

# Generic API fetcher
def fetch(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"API error for {endpoint}: {e}")
        return {}

# Schedule for today
def get_today_schedule():
    today = datetime.datetime.today()
    return fetch("/nhlschedule", {"year": today.year, "month": today.month, "day": today.day})

# Stats & data endpoints
def get_team_stats(team_id):
    return fetch("/team-statistic", {"teamId": team_id})

def get_players(team_id):
    return fetch("/players/id", {"teamId": team_id})

def get_player_stats(player_id):
    return fetch("/player-statistic", {"playerId": player_id})

def get_injuries():
    return fetch("/injuries")

def get_head_to_head(team_id_1, team_id_2):
    season = datetime.datetime.today().year
    games_1 = fetch("/schedule-team", {"season": season, "teamId": team_id_1})
    games_2 = fetch("/schedule-team", {"season": season, "teamId": team_id_2})
    return games_1, games_2

# Score calculation
def score_team(team_id, injuries, recent_weight=0.15, h2h_weight=0.15):
    stats = get_team_stats(team_id)
    players = get_players(team_id)
    active_players = []
    all_points = 0

    if players.get("data"):
        for p in players["data"]:
            if not any(i["playerId"] == p["id"] for i in injuries.get("data", [])):
                stats_p = get_player_stats(p["id"])
                all_points += stats_p.get("data", {}).get("points", 0)
                active_players.append(p["name"])

    score = 0
    score += (stats.get("data", {}).get("goalsPerGame", 0) or 0) * 0.25
    score += (all_points / max(len(active_players), 1)) * 0.20

    # Recent games factor
    recent_games = fetch("/nhlscoreboard", {
        "year": datetime.datetime.today().year,
        "month": datetime.datetime.today().month,
        "day": datetime.datetime.today().day,
        "limit": 5
    })
    if recent_games.get("data"):
        recent_count = len([g for g in recent_games["data"] if g["home"]["id"] == team_id or g["away"]["id"] == team_id])
        score += recent_count * recent_weight

    return score

# ---------- Streamlit UI ----------
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

schedule = get_today_schedule()
injuries = get_injuries()

if schedule.get("data"):
    matchups = schedule["data"]
    matchup_options = [f"{m['away']['name']} @ {m['home']['name']}" for m in matchups]
    selected = st.selectbox("Choose today's game:", matchup_options)

    if st.button("🔮 Predict Winner"):
        game = next(g for g in matchups if f"{g['away']['name']} @ {g['home']['name']}" == selected)
        away_score = score_team(game["away"]["id"], injuries)
        home_score = score_team(game["home"]["id"], injuries)
        winner = game["home"]["name"] if home_score > away_score else game["away"]["name"]

        st.success(f"🏆 Predicted Winner: {winner}")
        st.write(f"🔢 Home score: {home_score:.2f} | Away score: {away_score:.2f}")
else:
    st.info("No games found for today.")

