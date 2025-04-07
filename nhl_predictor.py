import streamlit as st
import requests
import os
from datetime import datetime

# Page setup
st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

# Get today's date (UTC)
today = datetime.utcnow()
year = today.strftime("%Y")
month = today.strftime("%m")
day = today.strftime("%d")
st.caption(f"📅 Showing games for **{today.strftime('%B %d, %Y')}** (UTC)")

# Get environment variables from Railway
API_KEY = os.environ.get("RAPIDAPI_KEY")
API_HOST = os.environ.get("RAPIDAPI_HOST")
BASE_URL = f"https://{API_HOST}"

# Function to get NHL schedule for today
def get_today_schedule():
    try:
        url = f"{BASE_URL}/nhlschedule?year={year}&month={month}&day={day}"
        headers = {
            "X-RapidAPI-Key": API_KEY,
            "X-RapidAPI-Host": API_HOST
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        st.error(f"❌ Failed to fetch schedule: {e}")
        return None

# Get and display today's games
schedule = get_today_schedule()

if schedule and "games" in schedule:
    games = schedule["games"]
    if games:
        options = [f"{game['name']} ({game['date']})" for game in games]
        selected_game = st.selectbox("Select a matchup to analyze:", options)
        if st.button("🔍 Predict Winner"):
            st.success(f"Prediction coming soon for: {selected_game}")
    else:
        st.info("📭 No valid matchups found.")
else:
    st.info("📭 No NHL games found or could not parse matchups.")
