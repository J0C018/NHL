import streamlit as st
import requests
import datetime
import os

st.set_page_config(page_title="🏒 NHL Matchup Predictor (AI-Enhanced)", layout="centered")

# Load API credentials from Railway environment variables
API_KEY = os.environ.get("RAPIDAPI_KEY")
API_HOST = os.environ.get("RAPIDAPI_HOST")

if not API_KEY or not API_HOST:
    st.error("❌ Missing API credentials. Please set RAPIDAPI_KEY and RAPIDAPI_HOST as environment variables.")
    st.stop()

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# Get today's date in UTC
today = datetime.datetime.utcnow()
year = today.year
month = today.month
day = today.day

# Title
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")
st.caption(f"🔄 Showing games for **{today.strftime('%B %d, %Y')}** (UTC)")

# Build API URL
url = f"https://{API_HOST}/nhlschedule?year={year}&month={month:02d}&day={day:02d}"

# Request data
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    st.error(f"❌ Failed to fetch schedule: {e}")
    st.stop()

# Parse and display matchups
games = data.get("games", [])

if not games:
    st.info("📅 No NHL games found or could not parse matchups.")
else:
    st.subheader("Today's Matchups")
    for game in games:
        try:
            matchup = game.get("name", "Matchup")
            start_time = game.get("date", "").replace("T", " ").replace("Z", " UTC")
            st.markdown(f"### {matchup}")
            st.markdown(f"🕒 Start Time: `{start_time}`")
            st.markdown("---")
        except Exception as parse_err:
            st.warning(f"⚠️ Couldn't parse a game: {parse_err}")
