import streamlit as st
import requests
import datetime
import os

# Page config
st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")

# Environment variables from Railway
API_KEY = os.environ.get("RAPIDAPI_KEY")
API_HOST = os.environ.get("RAPIDAPI_HOST")

HEADERS = {
    "x-rapidapi-host": API_HOST,
    "x-rapidapi-key": API_KEY
}

BASE_URL = f"https://{API_HOST}"

# Date setup: we force April 6, 2025 as user test target
target_date = datetime.datetime(2025, 4, 6)
year = target_date.strftime('%Y')
month = target_date.strftime('%m')
day = target_date.strftime('%d')

# Functions to fetch data
def get_schedule():
    url = f"{BASE_URL}/nhlschedule?year={year}&month={month}&day={day}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_team_ids():
    url = f"{BASE_URL}/team/id"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("teams", [])

def get_team_stats(team_id):
    url = f"{BASE_URL}/team-statistic?teamId={team_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_recent_performance(team_id):
    url = f"{BASE_URL}/schedule-team?season={year}&teamId={team_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_injuries():
    url = f"{BASE_URL}/injuries"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("injuries", [])

def get_head_to_head_summary(game_id):
    url = f"{BASE_URL}/nhlsummary?id={game_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

# Weighted AI matchup logic
def predict_game(home_id, away_id, game_id):
    try:
        home_stats = get_team_stats(home_id)
        away_stats = get_team_stats(away_id)

        injuries = get_injuries()
        head_to_head = get_head_to_head_summary(game_id)

        home_score = 0
        away_score = 0

        # Recent performance (15%)
        home_recent = get_recent_performance(home_id)
        away_recent = get_recent_performance(away_id)
        home_score += 15 * len(home_recent.get("schedule", []))
        away_score += 15 * len(away_recent.get("schedule", []))

        # Injuries (5%) if star players missing
        key_players = ["Connor McDavid", "Sidney Crosby", "Nathan MacKinnon"]
        for inj in injuries:
            if inj.get("player", "") in key_players:
                if str(home_id) in inj.get("teamId", ""):
                    home_score -= 5
                if str(away_id) in inj.get("teamId", ""):
                    away_score -= 5

        # Head-to-head (15%)
        if "previousGames" in head_to_head:
            for game in head_to_head["previousGames"]:
                winner = game.get("winner", "")
                if str(home_id) in winner:
                    home_score += 15
                elif str(away_id) in winner:
                    away_score += 15

        # Generic team strength (65%) — simple example
        home_score += 65 * float(home_stats.get("wins", 0))
        away_score += 65 * float(away_stats.get("wins", 0))

        if home_score > away_score:
            return f"Prediction: 🏠 Home team ({home_id}) wins!"
        else:
            return f"Prediction: 🚨 Away team ({away_id}) wins!"

    except Exception as e:
        return f"Error in prediction: {e}"

# UI
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

try:
    schedule = get_schedule()
    games = schedule.get("games", [])

    if not games:
        st.info("📅 No games found for today.")
    else:
        team_ids = get_team_ids()
        options = [f"{g['awayTeam']} @ {g['homeTeam']}" for g in games]
        selected_game = st.selectbox("Choose a matchup:", options)
        if st.button("🔮 Predict Result"):
            idx = options.index(selected_game)
            game = games[idx]
            home_id = game["homeId"]
            away_id = game["awayId"]
            game_id = game["gameId"]
            result = predict_game(home_id, away_id, game_id)
            st.success(result)

except Exception as e:
    st.error(f"❌ Error: {e}")

