import os
import requests
import datetime
import time
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# -------------------------------
# RAPIDAPI CONFIGURATION (for stats)
# -------------------------------
API_HOST = "nhl-api5.p.rapidapi.com"
API_KEY = os.environ.get("RAPIDAPI_KEY")
if not API_KEY:
    st.error("RAPIDAPI_KEY environment variable not set!")
    st.stop()

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# -------------------------------
# SCRAPE SCHEDULE FROM NHL WEBSITE USING SELENIUM
# -------------------------------
def scrape_schedule_nhl(date_str):
    """
    Scrape the NHL schedule for the specified date from the NHL website.
    
    We use Selenium (with headless Chrome) to load the page
    "https://www.nhl.com/schedule?date=YYYY-MM-DD" and then parse the HTML.
    
    Adjust the URL and selectors as needed if the NHL website changes.
    """
    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    # Add any additional options if necessary
    driver = webdriver.Chrome(options=options)
    
    url = f"https://www.nhl.com/schedule?date={date_str}"
    driver.get(url)
    # Wait for the page to fully load (adjust the sleep time if necessary)
    time.sleep(5)
    html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html, "lxml")
    games = []
    
    # The selectors below are based on the NHL schedule page as inspected.
    # In this example, we assume each game is within a <div> element with a class containing "ScheduleGame".
    game_elements = soup.find_all("div", class_="ScheduleGame")
    for element in game_elements:
        try:
            # Assume team names are in <span> elements with class "teamName"
            team_spans = element.find_all("span", class_="teamName")
            if len(team_spans) < 2:
                continue
            away_team = team_spans[0].get_text(strip=True)
            home_team = team_spans[1].get_text(strip=True)
            # Assume game time is in a <span> with class "gameTime"
            time_elem = element.find("span", class_="gameTime")
            game_time = time_elem.get_text(strip=True) if time_elem else "TBD"
            
            games.append({
                "away": away_team,
                "home": home_team,
                "time": game_time,
                "date": date_str,
                # Build a minimal structure expected by our predictor.
                "teams": {
                    "away": {"team": {"name": away_team, "id": None}},
                    "home": {"team": {"name": home_team, "id": None}}
                }
            })
        except Exception as e:
            st.write(f"Error parsing a game element: {e}")
            continue
    return games

# -------------------------------
# RAPIDAPI ENDPOINT FUNCTIONS (Team, Standings, Stats)
# -------------------------------
def fetch_teams():
    """
    Fetch the NHL team list using RapidAPI.
    Endpoint: GET /nhlteamlist
    """
    url = f"https://{API_HOST}/nhlteamlist"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error(f"Error fetching team list: HTTP {response.status_code}")
        return {}
    data = response.json()
    teams = {}
    for team in data.get("teams", []):
        team_id = team.get("id")
        teams[team_id] = team
    return teams

def fetch_standings(season="2024"):
    """
    Fetch NHL standings for the given season using RapidAPI.
    Endpoint: GET /nhlstandings?year=YYYY
    """
    url = f"https://{API_HOST}/nhlstandings"
    params = {"year": season}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching standings: HTTP {response.status_code}")
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

def fetch_team_stats(team_id):
    """
    Fetch detailed statistics for a team using RapidAPI.
    Endpoint: GET /team-statistic?teamId=TEAM_ID
    """
    url = f"https://{API_HOST}/team-statistic"
    params = {"teamId": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching team stats for team {team_id}: HTTP {response.status_code}")
        return {}
    return response.json()

def fetch_team_players(team_id):
    """
    Fetch players for a team using RapidAPI.
    Endpoint: GET /nhlteamplayers?teamid=TEAM_ID
    """
    url = f"https://{API_HOST}/nhlteamplayers"
    params = {"teamid": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching team players for team {team_id}: HTTP {response.status_code}")
        return []
    data = response.json()
    return data.get("players", [])

def fetch_player_stats(player_id):
    """
    Fetch detailed statistics for a player using RapidAPI.
    Endpoint: GET /player-statistic?playerId=PLAYER_ID
    """
    url = f"https://{API_HOST}/player-statistic"
    params = {"playerId": player_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching player stats for player {player_id}: HTTP {response.status_code}")
        return {}
    return response.json()

# -------------------------------
# PREDICTION LOGIC FUNCTIONS
# -------------------------------
def calculate_team_score(team_stats, is_home):
    """
    Calculate a weighted score for a team based on:
      - Win Percentage: 40%
      - Goal Differential (normalized): 30%
      - Corsi For Percentage: 15%
      - Home Advantage: +15% for home team
    """
    win_pct = team_stats.get("winPercentage", 0)
    goal_diff = team_stats.get("goalDifferential", 0)
    corsi = team_stats.get("corsiForPercentage", 0)
    normalized_goal_diff = goal_diff / 20.0
    score = (win_pct * 0.4) + (normalized_goal_diff * 0.3) + (corsi * 0.15)
    if is_home:
        score += 0.15
    return score

def predict_game(game, standings):
    """
    Predict the outcome of a game using team standings.
    Expects the game object to have a structure:
      game["teams"]["home"]["team"] and game["teams"]["away"]["team"]
    Note: If team IDs are missing (from the scraped schedule), you may need to map team names to IDs.
    """
    teams_info = game.get("teams", {})
    home_team = teams_info.get("home", {}).get("team", {})
    away_team = teams_info.get("away", {}).get("team", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")
    if not home_id or not away_id:
        return None  # Cannot predict without team IDs
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

# -------------------------------
# MAIN STREAMLIT APPLICATION
# -------------------------------
def main():
    st.title("NHL Outcome Predictor")
    st.markdown("""
    This application predicts NHL game outcomes using the schedule scraped directly from the NHL website,
    and combines that with team and player statistics (retrieved via RapidAPI).
    """)
    
    selected_date = st.date_input(
        "Select a date to fetch games:",
        value=datetime.date(2024, 10, 1),
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )
    date_str = selected_date.strftime("%Y-%m-%d")
    st.subheader(f"Games Scheduled for {date_str}")
    
    with st.spinner(f"Scraping schedule for {date_str}..."):
        schedule = scrape_schedule_nhl(date_str)
    if not schedule:
        st.warning("No games found for the selected date. Check the scraping logic and NHL website structure.")
        return
    
    # Fetch RapidAPI data
    with st.spinner("Fetching team list..."):
        teams = fetch_teams()
    season = st.text_input("Enter season for standings (e.g., 2024):", "2024")
    with st.spinner("Fetching standings..."):
        standings = fetch_standings(season)
    
    # Build a dropdown of games based on the scraped schedule
    game_options = []
    game_map = {}
    for game in schedule:
        away_team = game.get("away", "Unknown")
        home_team = game.get("home", "Unknown")
        display_str = f"{date_str} - {away_team} @ {home_team}"
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
        st.error("Could not generate a prediction for the selected game. (Team IDs may be missing.)")
        st.info("If the scraped schedule lacks team IDs, consider mapping team names to their corresponding IDs from the team list.")
        return
    
    # (Optional) Display additional details for teams/players via RapidAPI here.
    st.subheader("Additional Team and Player Data")
    st.write("Further integration (e.g., team stats, player stats) can be added below as needed.")

if __name__ == "__main__":
    main()


