import streamlit as st
import pandas as pd
import requests
import datetime
import os

st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

API_HOST = "nhl-api5.p.rapidapi.com"
API_KEY = os.environ.get("X_RAPIDAPI_KEY")

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

def get_teams():
    try:
        url = f"https://{API_HOST}/teams"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        teams_data = r.json()
        return {team['abbreviation']: team['id'] for team in teams_data['teams']}
    except Exception as e:
        st.error(f"❌ Failed to load teams: {e}")
        return {}

def get_today_schedule():
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        url = f"https://{API_HOST}/schedule?date={today}"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        games = data.get("dates", [])[0].get("games", [])
        return games
    except IndexError:
        st.warning("📅 No NHL games scheduled for today.")
        return []
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return []

def predict_matchup(home_team, away_team):
    return f"Prediction: {home_team} is slightly favored over {away_team}."

# Streamlit UI
st.title("🏒 NHL Matchup Predictor")

teams = get_teams()
games = get_today_schedule()

if teams and games:
    options = [f"{g['teams']['away']['team']['name']} @ {g['teams']['home']['team']['name']}" for g in games]
    selected = st.selectbox("Select a game to predict from today's matchups:", options)
    if st.button("🔮 Predict Result"):
        away, home = selected.split(" @ ")
        result = predict_matchup(home, away)
        st.success(result)
elif not games:
    st.info("📅 No NHL games found for today.")
