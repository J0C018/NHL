import os
import requests
import datetime
import streamlit as st
import pandas as pd

# Constants for the RapidAPI NHL API
API_HOST = "nhl-api5.p.rapidapi.com"
API_KEY = os.environ.get("RAPIDAPI_KEY")
if not API_KEY:
    st.error("RAPIDAPI_KEY environment variable not set!")
    st.stop()

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# --- API FETCH FUNCTIONS ---

def fetch_games(date_str):
    """
    Fetches games scheduled for a given date.
    Expected endpoint: GET https://nhl-api5.p.rapidapi.com/games?date=<YYYY-MM-DD>
    """
    url = f"https://{API_HOST}/games"
    params = {"date": date_str}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error("Error fetching games data!")
        return []
    data = response.json()
    # Assuming the response JSON has a key 'games' that is a list of game objects
    return data.get("games", [])

def fetch_teams():
    """
    Fetches team details.
    Expected endpoint: GET https://nhl-api5.p.rapidapi.com/teams
    """
    url = f"https://{API_HOST}/teams"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error("Error fetching teams data!")
        return {}
    data = response.json()
    # Build a dictionary mapping team IDs to team details
    teams = {}
    for team in data.get("teams", []):
        teams[team.get("id")] = team
    return teams

def fetch_standings():
    """
    Fetches current team standings.
    Expected endpoint: GET https://nhl-api5.p.rapidapi.com/standings
    """
    url = f"https://{API_HOST}/standings"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error("Error fetching standings data!")
        return {}
    data = response.json()
    # Build a dictionary mapping team IDs to standings/stats
    standings = {}
    # We assume the response includes a list of team standings under "standings"
    for team in data.get("standings", []):
        # Example expected keys (may differ based on API): 'teamId', 'winPercentage', 'goalDifferential'
        team_id = team.get("teamId")
        standings[team_id] = {
            "winPercentage": team.get("winPercentage", 0),
            "goalDifferential": team.get("goalDifferential", 0),
            # If available, the API might provide possession metrics; if not, these remain None.
            "corsiForPercentage": team.get("corsiForPercentage")
        }
    return standings

# --- PREDICTION LOGIC FUNCTIONS ---

def calculate_team_score(team_stats, is_home):
    """
    Calculates a simple score for a team using available metrics.
    We use:
      - winPercentage (assumed to be 0 to 1)
      - goalDifferential (normalized by an assumed scale)
      - corsiForPercentage (if available, as a fraction)
      - Bonus for home advantage.
    Weights are chosen to sum approximately to 1.
    """
    win_pct = team_stats.get("winPercentage", 0)
    goal_diff = team_stats.get("goalDifferential", 0)
    # Normalize goal differential (assuming a scale; adjust denominator as needed)
    normalized_goal_diff = goal_diff / 20.0  
    corsi = team_stats.get("corsiForPercentage")
    corsi_weight = 0.15
    corsi_score = (corsi if corsi is not None else 0) * corsi_weight

    # Weighted components: winPct (0.4), goal_diff (0.3), corsi (0.15)
    score = (win_pct * 0.4) + (normalized_goal_diff * 0.3) + corsi_score

    # Home advantage bonus
    if is_home:
        score += 0.15

    return score

def predict_game(game, standings):
    """
    Given a game object and standings dictionary,
    compute a prediction for the winner.
    """
    # Assuming game object contains keys "homeTeam" and "awayTeam" with "id" and "name"
    home_team = game.get("homeTeam")
    away_team = game.get("awayTeam")
    if not home_team or not away_team:
        return None

    home_id = home_team.get("id")
    away_id = away_team.get("id")

    home_stats = standings.get(home_id, {"winPercentage": 0, "goalDifferential": 0})
    away_stats = standings.get(away_id, {"winPercentage": 0, "goalDifferential": 0})

    home_score = calculate_team_score(home_stats, is_home=True)
    away_score = calculate_team_score(away_stats, is_home=False)

    prediction = {
        "homeTeam": home_team.get("name", "Unknown"),
        "awayTeam": away_team.get("name", "Unknown"),
        "homeScore": home_score,
        "awayScore": away_score,
        "predictedWinner": home_team.get("name") if home_score >= away_score else away_team.get("name"),
        "confidence": abs(home_score - away_score)
    }
    return prediction

# --- MAIN STREAMLIT APP ---

def main():
    st.title("NHL Outcome Predictor")
    st.markdown("""
    This application predicts the outcome of NHL games using team standings and available performance metrics 
    from the RapidAPI NHL API (nhl-api5). It pulls today’s games, retrieves current team standings, and then applies 
    a weighted model to predict winners.
    """)

    # Get today's date in YYYY-MM-DD format
    today_date = datetime.date.today().strftime("%Y-%m-%d")
    st.header(f"Games for {today_date}")

    # Fetch data from API endpoints
    with st.spinner("Fetching games..."):
        games = fetch_games(today_date)
    with st.spinner("Fetching teams..."):
        teams = fetch_teams()
    with st.spinner("Fetching standings..."):
        standings = fetch_standings()

    if not games:
        st.info("No games found for today.")
        return

    predictions = []
    for game in games:
        pred = predict_game(game, standings)
        if pred:
            predictions.append(pred)

    if not predictions:
        st.error("Unable to generate predictions for the games today.")
        return

    # Convert predictions to DataFrame for display
    df_predictions = pd.DataFrame(predictions)
    # Order columns for clarity
    df_predictions = df_predictions[["homeTeam", "awayTeam", "homeScore", "awayScore", "predictedWinner", "confidence"]]
    st.subheader("Predicted Outcomes")
    st.dataframe(df_predictions)

    st.markdown("""
    **Scoring Breakdown:**  
    - **Win Percentage (40%)**: Reflects overall success.
    - **Goal Differential (30%)**: Normalized by an assumed scale.
    - **Corsi For % (15%)**: Included if available.
    - **Home Advantage (15%)**: A bonus for the home team.
    """)

if __name__ == "__main__":
    main()

