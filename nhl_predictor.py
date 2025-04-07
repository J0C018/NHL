import streamlit as st
import requests
import datetime
import os

st.set_page_config(page_title="NHL Matchup Predictor (AI-Enhanced)", page_icon="🏒")

# Get env vars
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}
BASE_URL = f"https://{RAPIDAPI_HOST}"

def get_scoreboard(date):
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
    games = scoreboard_json.get("games", [])
    matchups = []
    for game in games:
        st.write("📦 Game Object:", game)  # <--- DEBUG
        try:
            home = game["homeTeam"].get("teamName") or game["homeTeam"].get("abbreviation")
            away = game["awayTeam"].get("teamName") or game["awayTeam"].get("abbreviation")
            game_id = game.get("gameId") or game.get("id")
            if home and away:
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
    return f"Prediction: {matchup['home']} is favored over {matchup['away']}."

# ---- App ----
st.title("🏒 NHL Matchup Predictor (AI-Enhanced)")

today = datetime.datetime(2025, 4, 6)
scoreboard_data = get_scoreboard(today)

if scoreboard_data:
    st.subheader("🧾 Raw Scoreboard Response")
    st.json(scoreboard_data)  # <--- DEBUG

    matchups = extract_matchups(scoreboard_data)
    if matchups:
        labels = [m["label"] for m in matchups]
        selected = st.selectbox("Select a matchup to predict:", labels)
        if st.button("🔮 Predict Result"):
            match = next((m for m in matchups if m["label"] == selected), None)
            if match:
                st.success(predict_matchup(match))
    else:
        st.info("📅 No valid matchups found for today.")
else:
    st.warning("⚠️ Could not retrieve game data.")

