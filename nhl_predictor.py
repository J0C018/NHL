import streamlit as st
import requests
import datetime
import os

st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")

# Get environment variables
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

BASE_URL = f"https://{RAPIDAPI_HOST}"

def get_scoreboard(date):
    """Fetch scoreboard from /nhlscoreboard with team matchups."""
    url = f"{BASE_URL}/nhlscoreboard"
    params = {
        "year": date.strftime("%Y"),
        "month": date.strftime("%m"),
        "day": date.strftime("%d")
    }
    try:
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Failed to load scoreboard: {e}")
        return {}

def extract_matchups(scoreboard_json):
    """Extract valid team matchups from the scoreboard."""
    games = scoreboard_json.get("games", [])
    matchups = []
    for game in games:
        try:
            home = game["homeTeam"]["teamName"]
            away = game["awayTeam"]["teamName"]
            game_id = game["gameId"]
            matchups.append({
                "label": f"{away} @ {home}",
                "home": home,
                "away": away,
                "id": game_id
            })
        except KeyError:
            continue
    return matchups

def predict_matchup(matchup):
    """Placeholder prediction function."""
    return f"Prediction: {matchup['home']} has a slight edge over {matchup['away']}."

# ---------- UI ----------
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

today = datetime.datetime(2025, 4, 6)
scoreboard_data = get_scoreboard(today)

if scoreboard_data:
    matchups = extract_matchups(scoreboard_data)
    if matchups:
        option_labels = [m["label"] for m in matchups]
        selected = st.selectbox("Select a matchup to predict:", option_labels)
        if st.button("🔮 Predict Result"):
            selected_game = next((m for m in matchups if m["label"] == selected), None)
            if selected_game:
                result = predict_matchup(selected_game)
                st.success(result)
    else:
        st.info("📅 No valid matchups found for today.")
else:
    st.warning("⚠️ Could not retrieve game data.")
