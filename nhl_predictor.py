import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- Constants ---
BASE_URL = "https://statsapi.web.nhl.com/api/v1"
CURRENT_SEASON = "20232024"

# --- Helpers ---
def get_today_schedule():
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/schedule?date={today}"
    try:
        r = requests.get(url)
        data = r.json()
        games = data.get("dates", [{}])[0].get("games", [])
        return games
    except Exception as e:
        st.warning(f"NHL API unreachable: {e}")
        return []

def get_team_stats(team_id):
    url = f"{BASE_URL}/teams/{team_id}/stats"
    try:
        r = requests.get(url)
        stats = r.json().get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
        return stats
    except:
        return {}

def get_teams():
    url = f"{BASE_URL}/teams"
    r = requests.get(url)
    data = r.json()
    return {team["name"]: team["id"] for team in data["teams"]}

def get_past_games():
    url = f"{BASE_URL}/schedule?season={CURRENT_SEASON}"
    try:
        r = requests.get(url)
        all_games = []
        for day in r.json().get("dates", []):
            all_games.extend(day.get("games", []))
        return all_games
    except:
        return []

def prepare_training_data(games, team_id_map):
    rows = []
    for game in games:
        if game['status']['abstractGameState'] != 'Final':
            continue
        try:
            home = game['teams']['home']['team']['name']
            away = game['teams']['away']['team']['name']
            home_id = team_id_map[home]
            away_id = team_id_map[away]
            home_stats = get_team_stats(home_id)
            away_stats = get_team_stats(away_id)
            rows.append({
                'homeTeam': home,
                'awayTeam': away,
                'home_win': game['teams']['home']['score'] > game['teams']['away']['score'],
                'home_goals_for': home_stats.get('goalsPerGame', 0),
                'away_goals_for': away_stats.get('goalsPerGame', 0),
                'home_goals_against': home_stats.get('goalsAgainstPerGame', 0),
                'away_goals_against': away_stats.get('goalsAgainstPerGame', 0),
                'home_win_pct': home_stats.get('winPct', 0),
                'away_win_pct': away_stats.get('winPct', 0),
            })
        except:
            continue
    return pd.DataFrame(rows)

def train_model(data):
    X = data.drop(columns=['homeTeam', 'awayTeam', 'home_win'])
    y = data['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, data), 'nhl_model.pkl')

def predict_match(home, away, team_id_map):
    model, _ = joblib.load('nhl_model.pkl')
    home_stats = get_team_stats(team_id_map[home])
    away_stats = get_team_stats(team_id_map[away])
    X = pd.DataFrame([{
        'home_goals_for': home_stats.get('goalsPerGame', 0),
        'away_goals_for': away_stats.get('goalsPerGame', 0),
        'home_goals_against': home_stats.get('goalsAgainstPerGame', 0),
        'away_goals_against': away_stats.get('goalsAgainstPerGame', 0),
        'home_win_pct': home_stats.get('winPct', 0),
        'away_win_pct': away_stats.get('winPct', 0),
    }])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return pred, proba

# --- UI ---
st.title("🏒 NHL Matchup Predictor")
team_id_map = get_teams()
today_games = get_today_schedule()

if today_games:
    matchups = [f"{g['teams']['away']['team']['name']} @ {g['teams']['home']['team']['name']}" for g in today_games]
    game_choice = st.selectbox("Select a game to predict:", matchups)

    if st.button("Train & Predict Today’s Game"):
        past_games = get_past_games()
        st.write(f"Training data from {CURRENT_SEASON}")
        df = prepare_training_data(past_games, team_id_map)
        if df.empty:
            st.error("Not enough data to train the model.")
        else:
            train_model(df)
            away, home = game_choice.split(" @ ")
            pred, proba = predict_match(home, away, team_id_map)
            outcome = "Home Win" if pred else "Away Win"
            st.success(f"Prediction: {outcome}")
            st.info(f"Probability - Home Win: {proba[1]:.2%}, Away Win: {proba[0]:.2%}")
else:
    st.warning("No NHL games scheduled today.")

