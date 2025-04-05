# NHL Predictor App (Updated with Error Handling, SportsData.io API, and Date Visibility)

import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load SportsData.io API Key (for local or Streamlit Cloud use)
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"] if "SPORTSDATA_API_KEY" in st.secrets else "YOUR_API_KEY_HERE"

# 2. Get Game Schedule Data by Date from SportsData.io
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
    game_data = []
    for game in games:
        game_data.append({
            "date": game.get("Day"),
            "homeTeam": game.get("HomeTeam"),
            "awayTeam": game.get("AwayTeam"),
            "homeScore": game.get("HomeTeamScore"),
            "awayScore": game.get("AwayTeamScore")
        })

    return pd.DataFrame(game_data)

# 3. (Optional) Get Player Season Stats By Team from SportsData.io
def get_player_stats_by_team(team):
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.warning(f"Could not fetch player stats for {team}. API Error: {response.status_code}")
        return pd.DataFrame()

    players = response.json()
    return pd.DataFrame(players)

# 4. Feature Engineering
def create_features(df):
    df['home_win'] = df['homeScore'] > df['awayScore']
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['homeTeam'] = df['homeTeam'].astype('category')
    df['awayTeam'] = df['awayTeam'].astype('category')
    df['homeTeam_code'] = df['homeTeam'].cat.codes
    df['awayTeam_code'] = df['awayTeam'].cat.codes
    return df

# 5. Train Model
def train_model(df):
    df = df.dropna(subset=['homeScore', 'awayScore'])
    df = create_features(df)
    X = df[['homeTeam_code', 'awayTeam_code', 'dayofweek']]
    y = df['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, df[['homeTeam', 'awayTeam', 'homeTeam_code', 'awayTeam_code']]), 'nhl_model.pkl')

# 6. Prediction Function with Validation
def predict_game(home, away):
    model, mapping_df = joblib.load('nhl_model.pkl')

    if home not in mapping_df['homeTeam'].values:
        raise ValueError(f"Home team '{home}' not found in training data.")
    if away not in mapping_df['awayTeam'].values:
        raise ValueError(f"Away team '{away}' not found in training data.")

    home_code = mapping_df[mapping_df['homeTeam'] == home]['homeTeam_code'].values[0]
    away_code = mapping_df[mapping_df['awayTeam'] == away]['awayTeam_code'].values[0]
    day = datetime.datetime.today().weekday()
    features = pd.DataFrame([[home_code, away_code, day]], columns=['homeTeam_code', 'awayTeam_code', 'dayofweek'])
    result = model.predict(features)
    return "Home Win" if result[0] else "Away Win"

# 7. Streamlit UI
st.title("🏒 NHL Predictor App (Powered by SportsData.io)")
st.markdown("""
This app predicts NHL game outcomes using machine learning and live data from SportsData.io.
Enter a home and away team abbreviation (e.g., BOS, TOR) and click **Predict**.
""")

home = st.text_input("Home Team Abbreviation", "BOS").upper()
away = st.text_input("Away Team Abbreviation", "TOR").upper()

if st.button("Train & Predict"):
    df = get_schedule()
    if df.empty:
        st.error("No games found to train on.")
    else:
        st.write(f"📅 Date range of training data: {df['date'].min().date()} to {df['date'].max().date()}")
        try:
            train_model(df)
            prediction = predict_game(home, away)
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if st.button("Show Player Stats for Home Team"):
    stats_df = get_player_stats_by_team(home)
    if not stats_df.empty:
        st.dataframe(stats_df[['Name', 'Position', 'Goals', 'Assists', 'Points', 'ShotsOnGoal']])
    else:
        st.info("No player stats found.")

if st.button("Show Available Teams"):
    df = get_schedule()
    if not df.empty:
        st.write("Available home teams from today's schedule:")
        st.write(df['homeTeam'].unique())

if st.checkbox("Show full schedule data"):
    df = get_schedule()
    if not df.empty:
        st.dataframe(df)


