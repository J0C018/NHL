import streamlit as st
import requests
import datetime
import os

st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")

# Get API details from environment variables (Railway secrets)
API_KEY = os.environ.get("RAPIDAPI_KEY")
API_HOST = os.environ.get("RAPIDAPI_HOST")

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

BASE_URL = f"https://{API_HOST}"

# Set today’s date
today = datetime.datetime.now().strftime("%Y-%m-%d")
year = datetime.datetime.now().year
month = f"{datetime.datetime.now().month:02d}"
day = f"{datetime.datetime.now().day:02d}"

st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

def get_schedule_data():
    try:
        url = f"{BASE_URL}/nhlscoreboard?year={year}&month={month}&day={day}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return None

# Load schedule data
scoreboard_data = get_schedule_data()

# Show raw response for debugging
if scoreboard_data:
    st.subheader("📦 Raw API Response:")
    st.json(scoreboard_data)

def extract_games(data):
    try:
        games = data.get("games", [])
        matchups = []
        for game in games:
            # Try multiple formats depending on API structure
            try:
                home = game['homeTeam']['abbreviation']
                away = game['awayTeam']['abbreviation']
            except KeyError:
                try:
                    home = game['teams']['home']['abbreviation']
                    away = game['teams']['away']['abbreviation']
                except KeyError:
                    continue  # Skip if format not found
            matchups.append(f"{away} @ {home}")
        return matchups
    except Exception as e:
        st.error(f"Error extracting games: {e}")
        return []

if scoreboard_data:
    matchups = extract_games(scoreboard_data)

    if matchups:
        selected = st.selectbox("Select a game to predict:", matchups)
        if st.button("🔮 Predict Winner"):
            st.success(f"Prediction: {selected.split('@')[1].strip()} is favored!")
    else:
        st.info("📅 No NHL games found or could not parse matchups.")
else:
    st.info("📅 Failed to retrieve any game data.")

