import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="🏒 NHL Predictor", layout="centered")

API_BASE = "https://api-web.nhle.com"

@st.cache_data(ttl=3600)
def get_teams():
    url = f"{API_BASE}/v1/teams"
    r = requests.get(url)
    return {t["abbrev"]: t["id"] for t in r.json()["teams"]}

@st.cache_data(ttl=300)
def get_today_schedule():
    today = datetime.now().strftime("%Y-%m-%d")
    r = requests.get(f"{API_BASE}/v1/schedule/{today}")
    return r.json().get("games", [])

@st.cache_data(ttl=3600)
def get_team_stats(team_id):
    r = requests.get(f"{API_BASE}/v1/team-stats/{team_id}/now")
    try:
        data = r.json()["teamStats"][0]["splits"][0]["stat"]
        return {
            "goals": float(data.get("goalsPerGame", 0)),
            "shots": float(data.get("shotsPerGame", 0)),
            "pp_pct": float(data.get("powerPlayPercentage", 0))
        }
    except:
        return {}

def train_model(matchups, teams):
    rows = []
    for g in matchups:
        try:
            h = g["homeTeam"]["abbrev"]
            a = g["awayTeam"]["abbrev"]
            hs = get_team_stats(teams[h])
            as_ = get_team_stats(teams[a])
            rows.append({
                "home_goals": hs["goals"],
                "away_goals": as_["goals"],
                "home_shots": hs["shots"],
                "away_shots": as_["shots"],
                "home_pp": hs["pp_pct"],
                "away_pp": as_["pp_pct"],
                "home_win": 1  # Fake training label
            })
        except:
            continue
    df = pd.DataFrame(rows)
    if df.empty: return None
    X = df.drop(columns=["home_win"])
    y = df["home_win"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

def predict_game(h_id, a_id):
    model = joblib.load("model.pkl")
    hs = get_team_stats(h_id)
    as_ = get_team_stats(a_id)
    X = pd.DataFrame([{
        "home_goals": hs["goals"],
        "away_goals": as_["goals"],
        "home_shots": hs["shots"],
        "away_shots": as_["shots"],
        "home_pp": hs["pp_pct"],
        "away_pp": as_["pp_pct"]
    }])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return pred, prob, hs, as_

# UI
st.title("🏒 NHL Matchup Predictor")
teams = get_teams()
games = get_today_schedule()

if not games:
    st.warning("No games today or unable to load schedule.")
else:
    options = [f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}" for g in games]
    pick = st.selectbox("Choose a game", options)
    away, home = pick.split(" @ ")

    if st.button("Predict"):
        with st.spinner("Training model..."):
            model = train_model(games, teams)
            if not model:
                st.error("Not enough data to train.")
            else:
                pred, prob, hs, as_ = predict_game(teams[home], teams[away])
                winner = "Home Win" if pred == 1 else "Away Win"
                st.success(f"Prediction: {winner}")
                st.write(f"🏠 **{home}** - Goals: {hs['goals']}, Shots: {hs['shots']}, PP%: {hs['pp_pct']}")
                st.write(f"🛫 **{away}** - Goals: {as_['goals']}, Shots: {as_['shots']}, PP%: {as_['pp_pct']}")
                st.info(f"Probabilities → Home: {prob[1]:.2%}, Away: {prob[0]:.2%}")

