import os
import requests
import datetime
import streamlit as st
import pandas as pd

# --- Configuration & Constants ---
API_HOST = "nhl-api5.p.rapidapi.com"
API_KEY = os.environ.get("RAPIDAPI_KEY")
if not API_KEY:
    st.error("RAPIDAPI_KEY environment variable not set!")
    st.stop()

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# --- API Fetch Functions ---

def fetch_schedule(date_str):
    """
    Fetch NHL schedule data for the specified date.
    Uses the endpoint based on:
    https://rapidapi.com/belchiorarkad-FqvHs2EDOtP/api/nhl-api5/playground/apiendpoint_5516d398-4fc0-4db5-9894-3d86d09516c0
    """
    url = f"https://{API_HOST}/schedule"
    params = {"date": date_str}  # Date must be in YYYY-MM-DD format
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching schedule data! HTTP {response.status_code}")
        return []
    data = response.json()
    # The API returns schedule data under the key "schedule"
    return data.get("schedule", [])

def fetch_teams():
    """
    Fetch NHL team details.
    """
    url = f"https://{API_HOST}/teams"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error(f"Error fetching teams data! HTTP {response.status_code}")
        return {}
    data = response.json()
    teams = {}
    # Expecting a "teams" key with a list of team objects
    for team in data.get("teams", []):
        team_id = team.get("id")
        teams[team_id] = team
    return teams

def fetch_standings(season="2023"):
    """
    Fetch NHL standings for a given season.
    """
    url = f"https://{API_HOST}/standings"
    params = {"season": season}  # Adjust the season parameter as needed
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching standings data! HTTP {response.status_code}")
        return {}
    data = response.json()
    standings = {}
    # Expecting a "standings" key with team statistics
    for team in data.get("standings", []):
        team_id = team.get("teamId")
        standings[team_id] = {
            "winPercentage": team.get("winPercentage", 0),
            "goalDifferential": team.get("goalDifferential", 0),
            "corsiForPercentage": team.get("corsiForPercentage", 0)  # Optional: if available
        }
    return standings

# --- Prediction Logic Functions ---

def calculate_team_score(team_stats, is_home):
    """
    Calculates a score for a team based on weighted factors:
      - Win Percentage: 40%
      - Goal Differential (normalized): 30%
      - Corsi For Percentage: 15%
      - Home Advantage: 15% bonus for home teams
    """
    win_pct = team_stats.get("winPercentage", 0)
    goal_diff = team_stats.get("goalDifferential", 0)
    corsi = team_stats.get("corsiForPercentage", 0)
    normalized_goal_diff = goal_diff / 20.0  # Normalization factor (adjustable)
    score = (win_pct * 0.4) + (normalized_goal_diff * 0.3) + (corsi * 0.15)
    if is_home:
        score += 0.15  # Home advantage bonus
    return score

def predict_game(game, standings):
    """
    Uses standings data to predict the outcome of the game.
    Assumes the game object contains keys 'homeTeam' and 'awayTeam', each with 'id' and 'name'.
    """
    home_team = game.get("homeTeam", {})
    away_team = game.get("awayTeam", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")
    if not home_id or not away_id:
        return None

    home_stats = standings.get(home_id, {})
    away_stats = standings.get(away_id, {})

    home_score = calculate_team_score(home_stats, is_home=True)
    away_score = calculate_team_score(away_stats, is_home=False)

    predicted_winner = home_team.get("name") if home_score >= away_score else away_team.get("name")
    confidence = abs(home_score - away_score)

    return {
        "homeTeam": home_team.get("name", "Unknown"),
        "awayTeam": away_team.get("name", "Unknown"),
        "homeScore": round(home_score, 3),
        "awayScore": round(away_score, 3),
        "predictedWinner": predicted_winner,
        "confidence": round(confidence, 3)
    }

# --- Main Streamlit App ---

def main():
    st.title("NHL Outcome Predictor for Today's Games")
    st.markdown("""
    This application predicts the outcome of NHL games scheduled for today.
    It fetches today's schedule, retrieves current team standings, and applies a weighted model 
    (based on win percentage, goal differential, and Corsi for percentage with a home advantage bonus)
    to predict the outcome.
    """)

    # Use today's date in YYYY-MM-DD format
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    st.subheader(f"Games Scheduled for {today_str}")

    # Fetch today's schedule, teams, and standings
    with st.spinner("Fetching today's schedule..."):
        schedule = fetch_schedule(today_str)
    if not schedule:
        st.warning(f"No games found for {today_str}.")
        return

    with st.spinner("Fetching team details..."):
        teams = fetch_teams()
    season_input = st.text_input("Enter season (e.g., 2023):", "2023")
    with st.spinner("Fetching standings..."):
        standings = fetch_standings(season_input)

    # Create a list of games for the dropdown
    game_options = []
    game_lookup = {}
    for game in schedule:
        # Construct a display string for the game (e.g., "Team A vs. Team B")
        home = game.get("homeTeam", {}).get("name", "Unknown")
        away = game.get("awayTeam", {}).get("name", "Unknown")
        game_str = f"{away} @ {home}"
        game_options.append(game_str)
        game_lookup[game_str] = game

    if not game_options:
        st.warning("No games available for selection.")
        return

    # Dropdown for selecting a game
    selected_game_str = st.selectbox("Select a game to predict its outcome:", game_options)
    selected_game = game_lookup.get(selected_game_str)

    if selected_game:
        prediction = predict_game(selected_game, standings)
        if prediction:
            st.subheader("Prediction Results")
            df = pd.DataFrame([prediction])
            st.dataframe(df)
            st.markdown(f"**Predicted Winner:** {prediction['predictedWinner']}")
            st.markdown(f"**Confidence Score:** {prediction['confidence']}")
        else:
            st.error("Could not generate a prediction for the selected game.")

if __name__ == "__main__":
    main()

