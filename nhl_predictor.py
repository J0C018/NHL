import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Load SportsData.io API key securely
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# ✅ Fetch game schedules by date
def get_schedule(date=None):
    if not date:
        date = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.sportsdata.io/v3/nhl/scores/json/GamesByDate/{date}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        return pd.DataFrame()
    games = response.json()
    return pd.DataFrame([{
        "date": game.get("Day"),
        "homeTeam": game.get("HomeTeam"),
        "awayTeam": game.get("AwayTeam"),
        "homeScore": game.get("HomeTeamScore"),
        "awayScore": game.get("AwayTeamScore")
    } for game in games])

# ✅ Optional: Fetch player stats by team
def get_player_stats_by_team(team):
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.warning(f"Could not fetch player stats for {team}. API Error: {response.status_code}")
        return pd.DataFrame()
    return pd.DataFrame(response.json())

# 🚀 Feature Engineering
def create_features(df):
    df['home_win'] = df['homeScore'] > df['awayScore']
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['homeTeam'] = df['homeTeam'].astype('category')
    df['awayTeam'] = df['awayTeam'].astype('category')
    df['homeTeam_code'] = df['homeTeam'].cat.codes
    df['awayTeam_code'] = df['awayTeam'].cat.codes
    return df

# 🚀 Model Training
def train_model(df):
    df = df.dropna(subset=['homeScore', 'awayScore'])
    df = create_features(df)
    X = df[['homeTeam_code', 'awayTeam_code', 'dayofweek']]
    y = df['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, df[['homeTeam', 'awayTeam', 'homeTeam_code', 'awayTeam_code']]), 'nhl_model.pkl')

# 🚀 Prediction
def predict_game(home, away):
    model, mapping_df = joblib.load('nhl_model.pkl')
    home_code = mapping_df[mapping_df['homeTeam'] == home]['homeTeam_code'].values[0]
    away_code = mapping_df[mapping_df['awayTeam'] == away]['awayTeam_code'].values[0]
    day = datetime.datetime.today().weekday()
    features = pd.DataFrame([[home_code, away_code, day]], columns=['homeTeam_code', 'awayTeam_code', 'dayofweek'])
    result = model.predict(features)
    return "Home Win" if result[0] else "Away Win"

# 🖥️ Streamlit UI
st.title("🏒 NHL Predictor (Powered by SportsData.io)")

home = st.text_input("Home Team Abbreviation", "BOS")
away = st.text_input("Away Team Abbreviation", "TOR")

if st.button("Train & Predict"):
    df = get_schedule()
    if df.empty:
        st.error("No games available today.")
    else:
        train_model(df)
        prediction = predict_game(home, away)
        st.success(f"Prediction: {prediction}")

if st.button("Show Player Stats for Home Team"):
    stats_df = get_player_stats_by_team(home)
    if not stats_df.empty:
        st.dataframe(stats_df[['Name', 'Position', 'Goals', 'Assists', 'Points', 'ShotsOnGoal']])
    else:
        st.info("No player stats found.")

