import streamlit as st
import requests
import datetime
import os

st.set_page_config(page_title="NHL Schedule Viewer", page_icon="🏒")

# Load from environment (Railway Secrets)
API_KEY = os.getenv("RAPIDAPI_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST")

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}
BASE_URL = f"https://{API_HOST}"

# Set a fixed date for testing — April 6, 2025
target_date = datetime.datetime(2025, 4, 6)
year = target_date.strftime('%Y')
month = target_date.strftime('%m')
day = target_date.strftime('%d')

st.title("🏒 NHL Schedule - RapidAPI Live Test")

def get_schedule():
    url = f"{BASE_URL}/nhlschedule"
    params = {
        "year": year,
        "month": month,
        "day": day
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch schedule: {e}")
        return {}

schedule_data = get_schedule()

st.subheader("📦 Raw API Response")
st.json(schedule_data)

def extract_matchups(data):
    games = data.get("games", [])
    matchups = []
    for game in games:
        try:
            home = game.get("homeTeam", {}).get("abbreviation") or game.get("homeTeam", {}).get("name")
            away = game.get("awayTeam", {}).get("abbreviation") or game.get("awayTeam", {}).get("name")
            if home and away:
                matchups.append(f"{away} @ {home}")
        except Exception as e:
            st.warning(f"Skipping game due to error: {e}")
    return matchups

matchups = extract_matchups(schedule_data)

if matchups:
    st.subheader("✅ Today's Matchups")
    for matchup in matchups:
        st.markdown(f"- {matchup}")
else:
    st.info("📅 No valid matchups found.")

