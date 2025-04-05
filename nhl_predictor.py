import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load API key
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# Fetch full season games
def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Today's schedule
def get_schedule():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.sportsdata.io/v3/nhl/scores/json/GamesByDate/{today}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Get player stats
def get_player_stats_by_team(team):
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Get injuries
def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Aggregate stats for training
def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    if df.empty or 'Team' not in df.columns:
        return {"goals_avg": 0, "shots_avg": 0, "points_avg": 0, "top_scorers": [], "scratched": []}
    injured_players = injury_df[injury_df['Team'] == team]['Name'].tolist()
    scratched = df[df['Name'].isin(injured_players)][['Name', 'Position']].to_dict('records')
    df_active = df[~df['Name'].isin(injured_players)]
    top_scorers = df_active.sort_values('Points', ascending=False).head(3)[['Name', 'Points']].to_dict('records')
    return {
        "goals_avg": df_active['Goals'].mean(),
        "shots_avg": df_active['ShotsOnGoal'].mean(),
        "points_avg": df_active['Points'].mean(),
        "top_scorers": top_scorers,
        "scratched": scratched
    }

# (Remaining code stays unchanged)







