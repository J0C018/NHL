import streamlit as st
import pandas as pd
import requests
import datetime
import os

st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

# ✅ Environment-based secrets (for Railway)
API_KEY = os.environ.get("RAPIDAPI_KEY")
API_HOST = os.environ.get("RAPIDAPI_HOST")
API_BASE = f"https://{API_HOST}"

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST
}

# 🔄 Get today's schedule
def get_today_schedule():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"{API_BASE}/schedule?date={today}"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        return data.get("games", [])
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return []

# Get all teams and map ID → name
def get_teams():
    try:
        url = f"{API_BASE}/teams"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        return {team['id']: team['name'] for team in data.get("teams", [])}
    except Exception as e:
        st.error(f"❌ Failed to load team names: {e}")
        return {}

# 🔮 Simple prediction logic
def predict_matchup(home_team, away_team):
    if len(home_team) > len(away_team):
        return f"Prediction: {home_team} is slightly favored."
    else:
        return f"Prediction: {away_team} is slightly favored."

# ------------------ UI -------------------
st.title("🏒 NHL Matchup Predictor (RapidAPI)")

teams = get_teams()
games = get_today_schedule()

if games and teams:
    matchup_options = []
    for g in games:
        home_id = g.get("homeTeam")
        away_id = g.get("awayTeam")
        home = teams.get(home_id, f"Team {home_id}")
        away = teams.get(away_id, f"Team {away_id}")
        matchup_options.append((f"{away} @ {home}", home, away))

    selected = st.selectbox("Select a matchup:", [x[0] for x in matchup_options])
    
    if st.button("🔮 Predict Result"):
        match = next((m for m in matchup_options if m[0] == selected), None)
        if match:
            result = predict_matchup(match[1], match[2])
            st.success(result)
else:
    st.info("📅 No games found or team data missing.")
