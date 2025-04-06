import streamlit as st
import requests
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="🏒 NHL Matchup Predictor", layout="centered")

# ✅ Your Proxy URL (via Railway)
PROXY_BASE = "https://web-production-7027.up.railway.app/api"

# 🔄 Get today's games from NHL schedule
@st.cache_data
def get_todays_schedule():
    today = datetime.date.today().isoformat()
    url = f"{PROXY_BASE}/v1/schedule"
    try:
        r = requests.get(url)
        games = r.json().get("dates", [])[0].get("games", [])
        return games
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return []

# 📋 Get team list
@st.cache_data
def get_teams():
    url = f"{PROXY_BASE}/v1/teams"
    try:
        r = requests.get(url)
        teams = {t["id"]: t["name"] for t in r.json()["teams"]}
        return teams
    except Exception as e:
        st.error(f"❌ Failed to load teams: {e}")
        return {}

# 🔧 Train a simple model with dummy features
def train_dummy_model():
    X = pd.DataFrame({
        "home_avg_goals": [3.1, 2.7, 3.4, 2.9],
        "away_avg_goals": [2.8, 3.0, 2.5, 3.2]
    })
    y = [1, 0, 1, 0]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "dummy_model.pkl")

# 📈 Predict match outcome
def predict_game(home_team, away_team):
    try:
        model = joblib.load("dummy_model.pkl")
    except:
        train_dummy_model()
        model = joblib.load("dummy_model.pkl")

    # Dummy averages (replace with real stats if added later)
    home_goals = 3.2
    away_goals = 2.7

    X_pred = pd.DataFrame([{
        "home_avg_goals": home_goals,
        "away_avg_goals": away_goals
    }])

    prob = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)[0]
    return pred, prob[1], prob[0]

# ----------------------------------------
# 🎛️ Streamlit Interface
# ----------------------------------------

st.title("🏒 NHL Matchup Predictor")
games = get_todays_schedule()
teams = get_teams()

if not games:
    st.warning("No NHL games scheduled for today.")
else:
    matchups = [f"{g['teams']['away']['team']['name']} @ {g['teams']['home']['team']['name']}" for g in games]
    selected_game = st.selectbox("Select a game:", matchups)

    if selected_game and st.button("🔮 Predict Result"):
        away_name, home_name = selected_game.split(" @ ")

        pred, prob_home, prob_away = predict_game(home_name, away_name)

        st.subheader("📢 Prediction Result")
        st.markdown(f"**Winner:** {'🏠 Home' if pred == 1 else '🛫 Away'}")
        st.info(f"Probability → Home Win: {prob_home:.2%}, Away Win: {prob_away:.2%}")

