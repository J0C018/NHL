import streamlit as st
import requests
import datetime
import os

# App title and config
st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

# Pulling environment variables
API_KEY = os.environ.get("RAPIDAPI_KEY")
API_HOST = os.environ.get("RAPIDAPI_HOST")

# Validate secrets
if not API_KEY or not API_HOST:
    st.error("❌ Missing API credentials. Make sure RAPIDAPI_KEY and RAPIDAPI_HOST are set in Railway.")
    st.stop()

# Define headers for RapidAPI
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST,
}

# Get today's date in UTC
today = datetime.datetime.now(datetime.UTC)
year = today.strftime("%Y")
month = today.strftime("%m")
day = today.strftime("%d")
formatted_date = today.strftime("%B %d, %Y")

st.info(f"📅 Showing games for **{formatted_date}** (UTC)")

# API request to get today's games
def get_todays_games():
    url = f"https://{API_HOST}/nhlschedule"
    params = {"year": year, "month": month, "day": day}

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch schedule: {e}")
        return {}

# Parse matchups from JSON
def parse_games(data):
    try:
        games = data.get("games", [])
        parsed = []
        for g in games:
            name = g.get("name")
            date = g.get("date")
            if name and date:
                parsed.append(f"{name} at {date}")
        return parsed
    except Exception as e:
        st.error(f"⚠️ Error parsing games: {e}")
        return []

# Main execution
data = get_todays_games()
games = parse_games(data)

if games:
    selected = st.selectbox("Select a game to predict:", games)
    if st.button("🔮 Predict Winner"):
        st.success(f"Prediction: {selected.split(' at ')[0]} is slightly favored.")
else:
    st.info("📭 No NHL games found or could not parse matchups.")

