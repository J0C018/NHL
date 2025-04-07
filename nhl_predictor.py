import streamlit as st
import requests
import datetime
import os

st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")

# RapidAPI credentials from Railway environment variables
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

BASE_URL = f"https://{RAPIDAPI_HOST}"

def get_schedule(date):
    """Fetches the NHL schedule for a specific date."""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    url = f"{BASE_URL}/nhlschedule?year={year}&month={month}&day={day}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return {}

def extract_matchups(schedule_json):
    """Extracts matchups from the /nhlschedule response."""
    games = schedule_json.get("20250406", {}).get("calendar", [])
    matchups = []
    for game in games:
        try:
            away = game["awayTeam"]["teamName"]
            home = game["homeTeam"]["teamName"]
            game_id = game["gameId"]
            matchups.append({
                "label": f"{away} @ {home}",
                "away": away,
                "home": home,
                "id": game_id
            })
        except Exception as e:
            continue  # Skip malformed entries
    return matchups

def predict_matchup(matchup):
    """Dummy prediction logic placeholder."""
    # TODO: Replace with AI-enhanced logic
    return f"Prediction: {matchup['home']} has a slight edge over {matchup['away']}."

# ---------- Streamlit UI ----------
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

today = datetime.datetime(2025, 4, 6)  # Test date for known 10PM EST game
schedule_data = get_schedule(today)

if schedule_data:
    matchups = extract_matchups(schedule_data)
    if matchups:
        options = [m["label"] for m in matchups]
        selected = st.selectbox("Select a game:", options)
        if st.button("🔮 Predict Result"):
            selected_matchup = next((m for m in matchups if m["label"] == selected), None)
            if selected_matchup:
                result = predict_matchup(selected_matchup)
                st.success(result)
    else:
        st.info("📅 No NHL games found or could not parse matchups.")
else:
    st.warning("⚠️ No schedule data available.")

