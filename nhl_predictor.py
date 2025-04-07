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

def fetch_schedule(year, month, day):
    """
    Fetch NHL schedule data for the specified date using:
    GET /nhlschedule?year=YYYY&month=MM&day=DD
    """
    url = f"https://{API_HOST}/nhlschedule"
    params = {
        "year": str(year),
        "month": str(month).zfill(2),
        "day": str(day).zfill(2)
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 404:
        return []
    elif response.status_code != 200:
        st.error(f"Error fetching schedule data! HTTP {response.status_code}")
        return []
    data = response.json()
    # Adjust the key if necessary based on the API's JSON structure.
    return data.get("games", [])

def fetch_teams():
    """
    Fetch NHL team list using:
    GET /nhlteamlist HTTP/1.1
    """
    url = f"https://{API_HOST}/nhlteamlist"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error(f"Error fetching teams data! HTTP {response.status_code}")
        return {}
    data = response.json()
    teams = {}
    # Assuming the JSON returns a key "teams"
    for team in data.get("teams", []):
        team_id = team.get("id")
        teams[team_id] = team
    return teams

def fetch_standings(season="2022"):
    """
    Fetch NHL standings for the specified season using:
    GET /nhlstandings?year=YYYY HTTP/1.1
    """
    url = f"https://{API_HOST}/nhlstandings"
    params = {"year": season}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching standings data! HTTP {response.status_code}")
        return {}
    data = response.json()
    standings = {}
    # Assuming the JSON returns a key "standings" as a list
    for team in data.get("standings", []):
        team_id = team.get("teamId")
        standings[team_id] = {
            "winPercentage": team.get("winPercentage", 0),
            "goalDifferential": team.get("goalDifferential", 0),
            "corsiForPercentage": team.get("corsiForPercentage", 0)
        }
    return standings

def fetch_team_stats(team_id):
    """
    Fetch detailed team statistics using:
    GET /team-statistic?teamId=TEAM_ID
    """
    url = f"https://{API_HOST}/team-statistic"
    params = {"teamId": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching team stats for team {team_id}! HTTP {response.status_code}")
        return {}
    return response.json()

def fetch_team_players(team_id):
    """
    Fetch team players using:
    GET /nhlteamplayers?teamid=TEAM_ID HTTP/1.1
    """
    url = f"https://{API_HOST}/nhlteamplayers"
    params = {"teamid": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching team players for team {team_id}! HTTP {response.status_code}")
        return []
    data = response.json()
    # Assuming the JSON returns a key "players"
    return data.get("players", [])

def fetch_player_stats(player_id):
    """
    Fetch detailed player statistics using:
    GET /player-statistic?playerId=PLAYER_ID
    """
    url = f"https://{API_HOST}/player-statistic"
    params = {"playerId": player_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching player stats for player {player_id}! HTTP {response.status_code}")
        return {}
    return response.json()

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
    
    normalized_goal_diff = goal_diff / 20.0  # Adjust normalization factor if needed.
    score = (win_pct * 0.4) + (normalized_goal_diff * 0.3) + (corsi * 0.15)
    if is_home:
        score += 0.15
    return score

def predict_game(game, standings):
    """
    Predicts the outcome of a game using historical standings data.
    Expects the game object to have the structure:
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
    This application predicts the outcomes of NHL games using historical team statistics.
    
    **Endpoints Used:**
    - **Schedule:** `/nhlschedule?year=YYYY&month=MM&day=DD` (Returns games for a specific day)
    - **Team List:** `/nhlteamlist`
    - **Standings:** `/nhlstandings?year=YYYY`
    - **Team Stats:** `/team-statistic?teamId=...`
    - **Team Players:** `/nhlteamplayers?teamid=...`
    - **Player Stats:** `/player-statistic?playerId=...`
    
    *Note:* The example parameters (e.g., 2022, a specific day) are just examples. You can change them to fetch data for any day or season.
    """)
    
    # Single date selection (fetch schedule for one day)
    selected_date = st.date_input(
        "Select a date to fetch games:",
        value=datetime.date.today(),
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )
    year = selected_date.year
    month = selected_date.month
    day = selected_date.day
    date_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    st.subheader(f"Games Scheduled for {date_str}")
    
    with st.spinner(f"Fetching schedule for {date_str}..."):
        schedule = fetch_schedule(year, month, day)
    
    if not schedule:
        st.warning("No games found for the selected date. Verify that the API has up-to-date schedule data for that date.")
        return
    
    with st.spinner("Fetching team list..."):
        teams = fetch_teams()
    
    season = st.text_input("Enter season for standings (e.g., 2022):", "2022")
    with st.spinner("Fetching standings..."):
        standings = fetch_standings(season)
    
    # Build dropdown list of games from the schedule
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
        return
    
    # Display additional information: Team Stats & Team Players
    teams_info = selected_game.get("teams", {})
    home_team = teams_info.get("home", {}).get("team", {})
    away_team = teams_info.get("away", {}).get("team", {})
    home_team_id = home_team.get("id")
    away_team_id = away_team.get("id")
    
    st.subheader("Team Statistics")
    if home_team_id:
        with st.expander(f"{home_team.get('name')} Stats"):
            home_team_stats = fetch_team_stats(home_team_id)
            st.json(home_team_stats)
    if away_team_id:
        with st.expander(f"{away_team.get('name')} Stats"):
            away_team_stats = fetch_team_stats(away_team_id)
            st.json(away_team_stats)
    
    st.subheader("Team Players and Player Statistics")
    if home_team_id:
        with st.expander(f"{home_team.get('name')} Players"):
            home_players = fetch_team_players(home_team_id)
            if home_players:
                # Display as a table
                home_players_df = pd.DataFrame(home_players)
                st.dataframe(home_players_df)
                # Optionally, let the user select a player to view detailed stats
                selected_home_player = st.selectbox("Select a Home Team Player for stats", 
                                                      options=home_players_df["id"].tolist(),
                                                      format_func=lambda pid: home_players_df.loc[home_players_df["id"] == pid, "name"].iloc[0])
                if selected_home_player:
                    st.markdown("**Home Player Stats:**")
                    player_stats = fetch_player_stats(selected_home_player)
                    st.json(player_stats)
    if away_team_id:
        with st.expander(f"{away_team.get('name')} Players"):
            away_players = fetch_team_players(away_team_id)
            if away_players:
                away_players_df = pd.DataFrame(away_players)
                st.dataframe(away_players_df)
                selected_away_player = st.selectbox("Select an Away Team Player for stats", 
                                                      options=away_players_df["id"].tolist(),
                                                      format_func=lambda pid: away_players_df.loc[away_players_df["id"] == pid, "name"].iloc[0])
                if selected_away_player:
                    st.markdown("**Away Player Stats:**")
                    player_stats = fetch_player_stats(selected_away_player)
                    st.json(player_stats)

if __name__ == "__main__":
    main()


