import streamlit as st
import requests
import datetime
import os

# Set page configuration
st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

# Retrieve API credentials from environment variables
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

BASE_URL = f"https://{RAPIDAPI_HOST}"

# Function to fetch the NHL scoreboard data
def get_scoreboard(date):
    url = f"{BASE_URL}/nhlscoreboard"
    params = {
        "year": date.strftime("%Y"),
        "month": date.strftime("%m"),
        "day": date.strftime("%d")
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to retrieve data: {e}")
        return {}

# Function to extract matchups from the API response
def extract_matchups(scoreboard_json):
    games = scoreboard_json.get("games", [])
    matchups = []
    for game in games:
        st.write("Game Data:", game)  # Debug: Display each game's data
        try:
            home_team = game.get("homeTeam", {}).get("abbreviation")
            away_team = game.get("awayTeam", {}).get("abbreviation")
            game_id = game.get("gameId")
            if home_team and away_team:
                matchups.append({
                    "label": f"{away_team} @ {home_team}",
                    "home": home_team,
                    "away": away_team,
                    "id": game_id
                })
        except KeyError as e:
            st.warning(f"Key error: {e} in game data: {game}")
    return matchups

# Main application logic
st.title("🏒 NHL Matchup Predictor")

# Set the date for which to fetch games
today = datetime.datetime(2025, 4, 6)
scoreboard_data = get_scoreboard(today)

if scoreboard_data:
    st.subheader("Raw API Response:")
    st.json(scoreboard_data)  # Debug: Display the raw API response

    matchups = extract_matchups(scoreboard_data)
    if matchups:
        matchup_labels = [m["label"] for m in matchups]
        selected_matchup = st.selectbox("Select a matchup to predict:", matchup_labels)
        if st.button("Predict Result"):
            selected_game = next((m for m in matchups if m["label"] == selected_matchup), None)
            if selected_game:
                st.success(f"Prediction: {selected_game['home']} is favored over {selected_game['away']}.")
    else:
        st.info("No valid matchups found for the selected date.")
else:
    st.warning("Failed to retrieve game data.")

