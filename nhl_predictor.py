import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
import time
import os
import json
from sklearn.ensemble import RandomForestClassifier

# Load API key
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# Team name fixer
TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

PREDICTION_LOG_PATH = "prediction_log.json"

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
    injured = injury_df[injury_df['Team'] == team]['Name'].tolist()
    df_active = df[~df['Name'].isin(injured)]
    scratched = df[df['Name'].isin(injured)][['Name', 'Position']].to_dict('records')
    top_scorers = df_active.sort_values('Points', ascending=False).head(3)[['Name', 'Points']].to_dict('records')
    return {
        "goals_avg": df_active['Goals'].mean(),
        "shots_avg": df_active['ShotsOnGoal'].mean(),
        "points_avg": df_active['Points'].mean(),
        "top_scorers": top_scorers,
        "scratched": scratched
    }

def train_model(df, injury_df):
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore'])
    df['home_win'] = df['HomeTeamScore'] > df['AwayTeamScore']
    df['date'] = pd.to_datetime(df['Day'], errors='coerce')
    df['dayofweek'] = df['date'].dt.dayofweek

    rows = []
    for _, row in df.iterrows():
        h_stats = aggregate_team_stats(row['HomeTeam'], injury_df)
        a_stats = aggregate_team_stats(row['AwayTeam'], injury_df)
        rows.append({
            "homeTeam": row['HomeTeam'],
            "awayTeam": row['AwayTeam'],
            "dayofweek": row['dayofweek'],
            "home_goals_avg": h_stats['goals_avg'],
            "away_goals_avg": a_stats['goals_avg'],
            "home_points_avg": h_stats['points_avg'],
            "away_points_avg": a_stats['points_avg'],
            "home_win": row['home_win']
        })

    model_df = pd.DataFrame(rows)
    X = model_df.drop(columns=['homeTeam', 'awayTeam', 'home_win'])
    y = model_df['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, model_df), 'nhl_model.pkl')

def predict_game(home, away, injury_df):
    model, _ = joblib.load('nhl_model.pkl')
    day = datetime.datetime.today().weekday()
    h_stats = aggregate_team_stats(home, injury_df)
    a_stats = aggregate_team_stats(away, injury_df)
    X_pred = pd.DataFrame([{
        "dayofweek": day,
        "home_goals_avg": h_stats['goals_avg'],
        "away_goals_avg": a_stats['goals_avg'],
        "home_points_avg": h_stats['points_avg'],
        "away_points_avg": a_stats['points_avg']
    }])
    proba = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)
    return {
        "result": "Home Win" if pred[0] else "Away Win",
        "prob_home_win": proba[1],
        "prob_away_win": proba[0],
        "home_top_scorers": h_stats['top_scorers'],
        "away_top_scorers": a_stats['top_scorers'],
        "home_scratched": h_stats['scratched'],
        "away_scratched": a_stats['scratched']
    }

def log_prediction(matchup, predicted_result, actual_result=None):
    log = {"matchup": matchup, "predicted": predicted_result, "actual": actual_result,
           "timestamp": datetime.datetime.now().isoformat()}
    history = []
    if os.path.exists(PREDICTION_LOG_PATH):
        with open(PREDICTION_LOG_PATH) as f:
            history = json.load(f)
    history.append(log)
    with open(PREDICTION_LOG_PATH, 'w') as f:
        json.dump(history, f, indent=2)

def show_prediction_history():
    if not os.path.exists(PREDICTION_LOG_PATH):
        st.info("No prediction history yet.")
        return
    with open(PREDICTION_LOG_PATH) as f:
        history = json.load(f)
    df = pd.DataFrame(history)
    st.subheader("📜 Prediction History")
    st.dataframe(df[::-1])

# ---------------------- UI ----------------------

st.title("🏒 NHL Matchup Predictor")
schedule_df = get_schedule()

if not schedule_df.empty:
    schedule_df = schedule_df.dropna(subset=['HomeTeam', 'AwayTeam'])
    matchups = schedule_df[['HomeTeam', 'AwayTeam']].drop_duplicates()
    matchup_options = matchups.apply(lambda row: f"{row['AwayTeam']} @ {row['HomeTeam']}", axis=1).tolist()
    selected_matchup = st.selectbox("Select a game to predict from today's matchups:", matchup_options)
    away, home = selected_matchup.split(" @ ")

    if st.button("Train & Predict Today’s Game"):
        season_df = get_full_season_schedule()
        injuries = get_injuries()
        if season_df.empty:
            st.error("Season data not available.")
        else:
            season_df['Day'] = pd.to_datetime(season_df['Day'], errors='coerce')
            st.write(f"📅 Date range of training data: {season_df['Day'].min()} to {season_df['Day'].max()}")
            try:
                train_model(season_df, injuries)
                result = predict_game(home, away, injuries)

                log_prediction(f"{away} @ {home}", result['result'])

                st.success(f"Prediction: {result['result']}")
                st.info(f"📊 Probability - Home Win: {result['prob_home_win']:.2%} | Away Win: {result['prob_away_win']:.2%}")

                st.subheader("🏒 Top Scorers (Home)")
                st.json(result['home_top_scorers'] or "No data available")

                st.subheader("🏒 Top Scorers (Away)")
                st.json(result['away_top_scorers'] or "No data available")

                st.subheader("🚑 Scratched Players (Home)")
                st.json(result['home_scratched'] or "None listed")

                st.subheader("🚑 Scratched Players (Away)")
                st.json(result['away_scratched'] or "None listed")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No games scheduled today.")

if st.checkbox("Show full season schedule data"):
    st.dataframe(get_full_season_schedule())

if st.sidebar.checkbox("📊 Show Prediction History"):
    show_prediction_history()





