import streamlit as st
import requests
import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="NHL Matchup Predictor", layout="centered")

# ----------------------------------------
# 🏒 NHL Public API Base URL
# ----------------------------------------
BASE_URL = "https://api-web.nhle.com/v1"

# ----------------------------------------
# 🔁 Helper: Get today’s games
# ----------------------------------------
@st.cache_data
def get_todays_schedule():
    today = datetime.date.today().isoformat()
    url = f"{BASE_URL}/schedule/{today}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data.get("gameWeek", [])[0].get("games", []) if data else []
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return []

# ----------------------------------------
# 🏒 Get all NHL teams and their IDs
# ----------------------------------------
@st.cache_data
def get_teams():
    url = f"{BASE_URL}/teams"
    try:
        r = requests.get(url, timeout=10)
        return {team["id"]: team["abbrev"] for team in r.json()}
    except Exception as e:
        st.error(f"❌ Failed to fetch team list. {e}")
        return {}

# ----------------------------------------
# 📊 Dummy model for illustration
# ----------------------------------------
def train_dummy_model():
    # Dummy training with synthetic features
    X = pd.DataFrame({
        "home_score_avg": [3.1, 2.7, 3.4, 2.9],
        "away_score_avg": [2.8, 3.0, 2.5, 3.2]
    })
    y = [1, 0, 1, 0]  # Home win = 1, Away win = 0
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "dummy_model.pkl")

# ----------------------------------------
# 📈 Predict game outcome
# ----------------------------------------
def predict_match(home, away):
    try:
        model = joblib.load("dummy_model.pkl")
    except:
        train_dummy_model()
        model = joblib.load("dummy_model.pkl")

    # NOTE: Placeholder averages — replace with real stats if you have them
    home_avg = 3.1
    away_avg = 2.8

    X_pred = pd.DataFrame([{
        "home_score_avg": home_avg,
        "away_score_avg": away_avg
    }])
    prob = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)[0]
    return pred, prob[1], prob[0]

# ----------------------------------------
# 🚀 Streamlit UI
# ----------------------------------------
st.title("🏒 NHL Matchup Predictor")

games = get_todays_schedule()
team_id_map = get_teams()

if not games:
    st.warning("No NHL games found for today.")
else:
    game_options = [
        f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}"
        for g in games
    ]
    selected_game = st.selectbox("Select a game to predict from today's matchups:", game_options)

    if selected_game and st.button("🔮 Predict Result"):
        away_abbr, home_abbr = selected_game.split(" @ ")
        pred, prob_home, prob_away = predict_match(home_abbr, away_abbr)
        st.subheader("📢 Prediction Result")
        st.markdown(f"**Winner:** {'🏠 Home' if pred == 1 else '🛫 Away'}")
        st.progress(prob_home if pred == 1 else prob_away)
        st.info(f"**Home Win Probability:** {prob_home:.2%} | **Away Win:** {prob_away:.2%}")
