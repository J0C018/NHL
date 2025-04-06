import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.set_page_config(page_title="NHL Matchup Predictor", layout="wide")

BASE_URL = "https://api-web.nhle.com"

@st.cache_data(ttl=3600)
def get_teams():
    url = f"{BASE_URL}/v1/teams"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to fetch team list.")
        return {}
    teams = r.json()["teams"]
    return {team["abbrev"]: team["id"] for team in teams}

@st.cache_data(ttl=300)
def get_today_schedule():
    today = datetime.today().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/v1/schedule/{today}"
    r = requests.get(url)
    if r.status_code != 200:
        st.warning("No live schedule found. Using fallback data.")
        return []
    return r.json().get("games", [])

@st.cache_data(ttl=3600)
def get_team_stats(team_id):
    url = f"{BASE_URL}/v1/team-stats/{team_id}/now"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    stats = r.json()
    try:
        data = stats["teamStats"][0]["splits"][0]["stat"]
        return {
            "goalsPerGame": float(data.get("goalsPerGame", 0)),
            "shotsPerGame": float(data.get("shotsPerGame", 0)),
            "powerPlayPercentage": float(data.get("powerPlayPercentage", 0)),
        }
    except:
        return None

@st.cache_data(ttl=86400)
def get_game_results(team_id):
    url = f"{BASE_URL}/v1/club-schedule-season/{team_id}/now"
    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()
    games = r.json()["games"]
    rows = []
    for game in games:
        if not game.get("gameOutcome"):
            continue
        is_home = game["homeTeamId"] == team_id
        result = game["gameOutcome"]["lastPeriodType"]
        rows.append({
            "homeTeam": game["homeTeamAbbrev"],
            "awayTeam": game["awayTeamAbbrev"],
            "homeWin": 1 if (is_home and game["homeTeamResult"] == "W") or
                             (not is_home and game["awayTeamResult"] == "L") else 0
        })
    return pd.DataFrame(rows)

def train_model(team_map):
    all_games = []
    for abbrev, team_id in team_map.items():
        df = get_game_results(team_id)
        all_games.append(df)
    full_df = pd.concat(all_games).dropna().drop_duplicates()

    feature_rows = []
    for _, row in full_df.iterrows():
        home_id = team_map.get(row["homeTeam"])
        away_id = team_map.get(row["awayTeam"])
        home_stats = get_team_stats(home_id)
        away_stats = get_team_stats(away_id)
        if not home_stats or not away_stats:
            continue
        feature_rows.append({
            "home_goals": home_stats["goalsPerGame"],
            "away_goals": away_stats["goalsPerGame"],
            "home_shots": home_stats["shotsPerGame"],
            "away_shots": away_stats["shotsPerGame"],
            "home_pp": home_stats["powerPlayPercentage"],
            "away_pp": away_stats["powerPlayPercentage"],
            "homeWin": row["homeWin"]
        })

    model_df = pd.DataFrame(feature_rows)
    if model_df.empty:
        st.error("Not enough historical data to train model.")
        return None
    X = model_df.drop(columns=["homeWin"])
    y = model_df["homeWin"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

def predict_game(home_abbrev, away_abbrev, team_map):
    model = joblib.load("model.pkl")
    home_id = team_map[home_abbrev]
    away_id = team_map[away_abbrev]
    home_stats = get_team_stats(home_id)
    away_stats = get_team_stats(away_id)
    if not home_stats or not away_stats:
        return None
    X_pred = pd.DataFrame([{
        "home_goals": home_stats["goalsPerGame"],
        "away_goals": away_stats["goalsPerGame"],
        "home_shots": home_stats["shotsPerGame"],
        "away_shots": away_stats["shotsPerGame"],
        "home_pp": home_stats["powerPlayPercentage"],
        "away_pp": away_stats["powerPlayPercentage"]
    }])
    prob = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)[0]
    return {
        "prediction": "Home Win" if pred == 1 else "Away Win",
        "prob_home": prob[1],
        "prob_away": prob[0],
        "home_stats": home_stats,
        "away_stats": away_stats
    }

# ===================== Streamlit UI =====================

st.title("🏒 NHL Matchup Predictor")
teams = get_teams()
today_games = get_today_schedule()

if today_games:
    matchups = [f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}" for g in today_games]
    selected_game = st.selectbox("Select today's matchup:", matchups)
    away_abbrev, home_abbrev = selected_game.split(" @ ")
    
    if st.button("🔮 Predict Outcome"):
        with st.spinner("Training model and fetching stats..."):
            model = train_model(teams)
            if model:
                result = predict_game(home_abbrev, away_abbrev, teams)
                if result:
                    st.success(f"🏆 Predicted Winner: **{result['prediction']}**")
                    st.info(f"📊 Probability: Home Win: {result['prob_home']:.2%} | Away Win: {result['prob_away']:.2%}")
                    st.markdown("### 🔎 Why this prediction?")
                    st.write(f"**Home Team ({home_abbrev}) Stats:** {result['home_stats']}")
                    st.write(f"**Away Team ({away_abbrev}) Stats:** {result['away_stats']}")
                else:
                    st.error("❌ Could not fetch stats for prediction.")
else:
    st.warning("No NHL games found for today.")

