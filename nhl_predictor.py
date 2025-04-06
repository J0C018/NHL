import streamlit as st
import pandas as pd
import requests
import datetime

st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

PROXY_URL = "https://web-production-7027.up.railway.app"

def get_teams():
    try:
        r = requests.get(f"{PROXY_URL}/api/v1/teams")
        r.raise_for_status()
        teams_data = r.json()
        return {team['abbreviation']: team['id'] for team in teams_data['teams']}
    except Exception as e:
        st.error(f"❌ Failed to load teams: {e}")
        return {}

def get_today_schedule():
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        r = requests.get(f"{PROXY_URL}/api/v1/schedule?date={today}")
        r.raise_for_status()
        schedule_data = r.json()
        games = schedule_data.get("dates", [])[0].get("games", [])
        return games
    except IndexError:
        st.warning("No NHL games scheduled for today.")
        return []
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return []

def predict_matchup(home_team, away_team):
    # Placeholder: basic logic
    return f"Prediction: {home_team} is slightly favored over {away_team}."

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

