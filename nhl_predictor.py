import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

# Set environment variable in Railway as X_RAPIDAPI_KEY
API_KEY = st.secrets["X_RAPIDAPI_KEY"]
API_HOST = "nhl-api5.p.rapidapi.com"

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

def get_team_players(team_id):
    try:
        url = f"https://{API_HOST}/players/id?teamId={team_id}"
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch team players: {e}")
        return {}

st.title("🏒 NHL Matchup Predictor (RapidAPI Edition)")

# Example: Fetch player data for Washington Capitals (ID: 16)
team_id = st.text_input("Enter NHL Team ID (e.g. 16 for Capitals)", "16")

if st.button("Fetch Team Players"):
    data = get_team_players(team_id)
    if "players" in data:
        players = data["players"]
        df = pd.DataFrame(players)
        st.dataframe(df[["name", "position", "age", "nationality"]])
    else:
        st.warning("No player data found.")

