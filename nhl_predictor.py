import os
import datetime
import time
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests

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
    Scrape the NHL schedule from the official NHL schedule page.
    URL is assumed to be: https://www.nhl.com/schedule?date=YYYY-MM-DD
    IMPORTANT: You must verify and adjust the CSS selectors below by inspecting the page.
    """
    # Set up headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    # If needed, specify the chromedriver path:
    # driver = webdriver.Chrome(executable_path="/path/to/chromedriver", options=chrome_options)
    driver = webdriver.Chrome(options=chrome_options)
    
    url = f"https://www.nhl.com/schedule?date={date_str}"
    driver.get(url)
    # Allow time for page to load; adjust if necessary
    time.sleep(5)
    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "lxml")
    games = []
    
    # YOU NEED TO UPDATE THESE SELECTORS BASED ON THE PAGE'S CURRENT HTML
    # Example selectors (these are placeholders):
    game_cards = soup.select("div.gameCard")  # Assume each game is in a div with class "gameCard"
    for card in game_cards:
        try:
            away_elem = card.select_one("span.awayTeamName")
            home_elem = card.select_one("span.homeTeamName")
            time_elem = card.select_one("span.gameTime")
            if away_elem and home_elem:
                away_team = away_elem.get_text(strip=True)
                home_team = home_elem.get_text(strip=True)
                game_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                games.append({
                    "away": away_team,
                    "home": home_team,
                    "time": game_time,
                    "date": date_str,
                    # For prediction, we need a structure like this; team IDs must be mapped later:
                    "teams": {
                        "away": {"team": {"name": away_team, "id": None}},
                        "home": {"team": {"name": home_team, "id": None}}
                    }
                })
        except Exception as e:
            st.write(f"Error processing a game card: {e}")
            continue
    return games

# -------------------------------
# RAPIDAPI ENDPOINT FUNCTIONS FOR STATS
# -------------------------------
def fetch_teams():
    """Fetch the NHL team list using RapidAPI."""
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
    """Fetch NHL standings using RapidAPI."""
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
    """Fetch detailed statistics for a team using RapidAPI."""
    url = f"https://{API_HOST}/team-statistic"
    params = {"teamId": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching stats for team {team_id}: HTTP {response.status_code}")
        return {}
    return response.json()

def fetch_team_players(team_id):
    """Fetch players for a team using RapidAPI."""
    url = f"https://{API_HOST}/nhlteamplayers"
    params = {"teamid": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching players for team {team_id}: HTTP {response.status_code}")
        return []
    data = response.json()
    return data.get("players", [])

def fetch_player_stats(player_id):
    """Fetch detailed statistics for a player using RapidAPI."""
    url = f"https://{API_HOST}/player-statistic"
    params = {"playerId": player_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching stats for player {player_id}: HTTP {response.status_code}")
        return {}
    return response.json()

# -------------------------------
# PREDICTION LOGIC
# -------------------------------
def calculate_team_score(team_stats, is_home):
    """
    Calculate a weighted score for a team based on:
      - Win Percentage (40%)
      - Normalized Goal Differential (30%)
      - Corsi For Percentage (15%)
      - Home Advantage (15% bonus for home teams)
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
    Predict a game's outcome based on team statistics.
    NOTE: If the scraped schedule doesn't provide team IDs, you need to map team names to IDs.
    """
    teams_info = game.get("teams", {})
    home_team = teams_info.get("home", {}).get("team", {})
    away_team = teams_info.get("away", {}).get("team", {})
    home_id = home_team.get("id")
    away_id = away_team.get("id")
    
    if not home_id or not away_id:
        return None  # Without team IDs from the schedule, prediction won't work.
    
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
    This application predicts NHL game outcomes by scraping today's (or a selected date's) schedule from the NHL website
    and combining it with team and player statistics from RapidAPI.
    
    **Important:** You must update the CSS selectors in the scraping function to match the current NHL schedule page.
    """)
    
    # Let the user select a date (for example, today)
    selected_date = st.date_input(
        "Select a date to fetch games:",
        value=datetime.date.today(),
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2025, 12, 31)
    )
    date_str = selected_date.strftime("%Y-%m-%d")
    st.subheader(f"Games Scheduled for {date_str}")
    
    # Scrape the schedule from the NHL website
    with st.spinner(f"Scraping schedule for {date_str}..."):
        schedule = scrape_schedule_nhl(date_str)
    if not schedule:
        st.warning("No games found for the selected date. Please verify your selectors and URL.")
        return
    
    # Fetch additional data via RapidAPI
    with st.spinner("Fetching team list..."):
        teams = fetch_teams()
    season = st.text_input("Enter season for standings (e.g., 2024):", "2024")
    with st.spinner("Fetching standings..."):
        standings = fetch_standings(season)
    
    # Build a dropdown of games from the scraped schedule
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
    
    # Attempt prediction
    prediction = predict_game(selected_game, standings)
    if prediction:
        st.subheader("Prediction Results")
        df = pd.DataFrame([prediction])
        st.dataframe(df)
        st.markdown(f"**Predicted Winner:** {prediction['predictedWinner']}")
        st.markdown(f"**Confidence Score:** {prediction['confidence']}")
    else:
        st.error("Prediction could not be generated. It may be due to missing team IDs from the scraped data.")
        st.info("If team IDs are missing, please map scraped team names to RapidAPI team IDs manually.")
        return
    
    # You can then integrate additional details (team stats, player stats) as needed.
    st.subheader("Additional Data")
    st.write("Additional integration for detailed stats can be added here.")

if __name__ == "__main__":
    main()


