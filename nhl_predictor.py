import os
import requests
import datetime
import streamlit as st
import pandas as pd

# --- API Configuration ---
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
    Uses the endpoint: GET /schedule?date=<YYYY-MM-DD>
    Assumes the API returns a JSON object with a "games" key.
    """
    url = f"https://{API_HOST}/schedule"
    params = {"date": date_str}
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code == 404:
        # No data found for this date
        return []
    elif response.status_code != 200:
        st.error(f"Error fetching schedule data! HTTP {response.status_code}")
        return []
    
    data = response.json()
    return data.get("games", [])

def fetch_teams():
    """
    Fetch NHL team details.
    Uses: GET /teams
    """
    url = f"https://{API_HOST}/teams"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error(f"Error fetching teams data! HTTP {response.status_code}")
        return {}
    
    data = response.json()
    teams = {}
    for team in data.get("teams", []):
        team_id = team.get("id")
        teams[team_id] = team
    return teams

def fetch_standings(season="2024"):
    """
    Fetch NHL standings for the specified season.
    Uses: GET /standings?season=<year>
    """
    url = f"https://{API_HOST}/standings"
    params = {"season": season}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching standings data! HTTP {response.status_code}")
        return {}
    
    data = response.json()
    standings = {}
    for team in data.get("standings", []):
        team_id = team.get("teamId")
        standings[team_id] = {
            "winPercentage": team.get("winPercentage", 0),
            "goalDifferential": team.get("goalDifferential", 0),
            "corsiForPercentage": team.get("corsiForPercentage", 0)
        }
    return standings

# --- Prediction Logic Functions ---

def calculate_team_score(team_stats, is_home):
    """
    Calculates a score for a team using weighted historical metrics:
      - Win Percentage: 40%
      - Goal Differential (normalized): 30%
      - Corsi For Percentage: 15%
      - Home Advantage: +15% for home team
    """
    win_pct = team_stats.get("winPercentage", 0)
    goal_diff = team_stats.get("goalDifferential", 0)
    corsi = team_stats.get("corsiForPercentage", 0)
    
    # Normalize goal differential (adjust factor as needed)
    normalized_goal_diff = goal_diff / 20.0
    score = (win_pct * 0.4) + (normalized_goal_diff * 0.3) + (corsi * 0.15)
    if is_home:
        score += 0.15
    return score

def predict_game(game, standings):
    """
    Predicts the outcome of a game using historical standings data.
    The game object is expected to have a nested structure:
      game["teams"]["home"]["team"] and game["teams"]["away"]["team"]
    """
    teams_info = game.get("teams", {})
    home_team = teams_info.get("home", {}).get("team", {})
    away_team = teams_info.get("away", {}).get("team", {})
    
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
    This application predicts the outcomes of NHL games by using historical team statistics.
    Select a specific date (from 2020 to 2025) to fetch games and see predictions.
    The prediction model is based on win percentage, goal differential, and Corsi for percentage with a home advantage bonus.
    """)
    
    # Single date selection
    selected_date = st.date_input(
        "Select a date to fetch games:",
        value=datetime.date.today(),
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )
    date_str = selected_date.strftime("%Y-%m-%d")
    st.subheader(f"Games Scheduled for {date_str}")
    
    with st.spinner(f"Fetching schedule for {date_str}..."):
        schedule = fetch_schedule(date_str)
    
    if not schedule:
        st.warning("No games found for the selected date. Verify that the API has up-to-date schedule data for that date.")
        return
    
    with st.spinner("Fetching team details..."):
        teams = fetch_teams()
    
    season = st.text_input("Enter season for standings (e.g., 2023):", "2023")
    with st.spinner("Fetching standings..."):
        standings = fetch_standings(season)
    
    # Build dropdown list of games
    game_options = []
    game_map = {}
    for game in schedule:
        teams_info = game.get("teams", {})
        home_team = teams_info.get("home", {}).get("team", {})
        away_team = teams_info.get("away", {}).get("team", {})
        home_name = home_team.get("name", "Unknown")
        away_name = away_team.get("name", "Unknown")
        display_str = f"{date_str} - {away_name} @ {home_name}"
        game_options.append(display_str)
        game_map[display_str] = game
    
    selected_game_str = st.selectbox("Select a game to predict its outcome:", game_options)
    selected_game = game_map[selected_game_str]
    
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

