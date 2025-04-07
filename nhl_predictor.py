import os
import requests
import datetime
import streamlit as st
import pandas as pd

# API Configuration
API_HOST = "nhl-api5.p.rapidapi.com"
API_KEY = os.environ.get("RAPIDAPI_KEY")
if not API_KEY:
    st.error("RAPIDAPI_KEY environment variable not set!")
    st.stop()

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# --- Data Fetching Functions ---

def fetch_schedule(date_str):
    """
    Fetch NHL schedule data for a given date.
    Based on the documentation from the NHL schedule endpoint.
    """
    url = f"https://{API_HOST}/schedule"
    params = {
        "date": date_str  # Ensure this matches the expected format (YYYY-MM-DD)
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching schedule data! HTTP {response.status_code}")
        return []
    data = response.json()
    # The API returns a key (e.g., "schedule") with an array of game objects.
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
    # Expecting a key "teams" containing a list of team objects.
    for team in data.get("teams", []):
        team_id = team.get("id")
        teams[team_id] = team
    return teams

def fetch_standings(season="2023"):
    """
    Fetch current NHL standings for a given season.
    """
    url = f"https://{API_HOST}/standings"
    params = {
        "season": season  # Adjust the season parameter as required.
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching standings data! HTTP {response.status_code}")
        return {}
    data = response.json()
    standings = {}
    # Expecting a key "standings" containing team statistics.
    for team in data.get("standings", []):
        team_id = team.get("teamId")
        standings[team_id] = {
            "winPercentage": team.get("winPercentage", 0),
            "goalDifferential": team.get("goalDifferential", 0),
            "corsiForPercentage": team.get("corsiForPercentage", 0)  # If provided
        }
    return standings

# --- Prediction Logic ---

def calculate_team_score(team_stats, is_home):
    """
    Calculates a score for a team based on key metrics.
    Weights (example):
      - Win Percentage: 40%
      - Goal Differential (normalized): 30%
      - Corsi For Percentage: 15%
      - Home Advantage Bonus: 15% for home team
    """
    win_pct = team_stats.get("winPercentage", 0)
    goal_diff = team_stats.get("goalDifferential", 0)
    corsi = team_stats.get("corsiForPercentage", 0)
    normalized_goal_diff = goal_diff / 20.0  # Normalize goal differential by an assumed scale
    score = (win_pct * 0.4) + (normalized_goal_diff * 0.3) + (corsi * 0.15)
    if is_home:
        score += 0.15  # Home advantage bonus
    return score

def predict_game(game, standings):
    """
    Predicts the outcome of a game based on team standings.
    The game object is expected to include team objects for 'homeTeam' and 'awayTeam'
    with fields 'id' and 'name'.
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
    st.title("NHL Outcome Predictor")
    st.markdown("""
    This app predicts NHL game outcomes by fetching today's schedule, team details, 
    and standings data from the RapidAPI NHL API (nhl-api5). The prediction model uses 
    win percentage, goal differential, and Corsi for percentage (if available), along with a home advantage bonus.
    """)

    # Input date and season
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    date_input = st.text_input("Enter game date (YYYY-MM-DD):", today_str)
    season_input = st.text_input("Enter season (e.g., 2023):", "2023")

    # Fetch data using our functions
    with st.spinner("Fetching schedule..."):
        schedule = fetch_schedule(date_input)
    if not schedule:
        st.warning(f"No games found for {date_input}.")
        return

    with st.spinner("Fetching teams..."):
        teams = fetch_teams()
    with st.spinner("Fetching standings..."):
        standings = fetch_standings(season_input)

    # Generate predictions for each game
    predictions = []
    for game in schedule:
        pred = predict_game(game, standings)
        if pred:
            predictions.append(pred)

    if predictions:
        df = pd.DataFrame(predictions)
        st.subheader("Predicted Outcomes")
        st.dataframe(df)
    else:
        st.warning("No predictions could be generated for the selected date.")

if __name__ == "__main__":
    main()

