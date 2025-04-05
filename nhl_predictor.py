import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
import time
import os
import json
from sklearn.ensemble import RandomForestClassifier

SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]
PREDICTION_LOG_PATH = "prediction_log.json"

TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

def get_schedule():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.sportsdata.io/v3/nhl/scores/json/GamesByDate/{today}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    df = pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    return df[df['Status'] != 'Final'] if 'Status' in df.columns else df

@st.cache_data
def get_player_stats_by_team(team):
    team_fixed = TEAM_ABBREVIATION_FIX.get(team, team)
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team_fixed}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    time.sleep(0.5)
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    try:
        data = r.json()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    if df.empty or not {'Name', 'Team', 'Goals', 'ShotsOnGoal', 'Points'}.issubset(df.columns):
        return {
            "goals_avg": 0, "shots_avg": 0, "points_avg": 0,
            "top_scorers": [], "scratched": []
        }





