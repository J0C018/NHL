import streamlit as st
import pandas as pd
import datetime
import requests
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- UTILITIES ---

def get_today_schedule():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={today}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to fetch today's games.")
        return []
    data = r.json()
    games = data.get("dates", [{}])[0].get("games", [])
    return games

def get_team_roster(team_id):
    url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/roster"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    return r.json().get("roster", [])

def get_player_stats(player_id):
    url = f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=statsSingleSeason&season=20232024"
    r = requests.get(url)
    if r.status_code != 200:
        return {}
    stats = r.json().get("stats", [])
    if stats and "splits" in stats[0] and stats[0]["splits"]:
        return stats[0]["splits"][0].get("stat", {})
    return {}

def aggregate_team_stats(team_id):
    roster = get_team_roster(team_id)
    players_stats = []
    for player in roster:
        stat = get_player_stats(player["person"]["id"])
        if "points" in stat:
            players_stats.append({
                "name": player["person"]["fullName"],
                "points": stat.get("points", 0),
                "goals": stat.get("goals", 0),
                "shots": stat.get("shots", 0)
            })
    df = pd.DataFrame(players_stats)
    if df.empty:
        return {"avg_points": 0, "avg_goals": 0, "avg_shots": 0, "top_scorers": []}
    top_scorers = df.sort_values("points", ascending=False).head(3).to_dict("records")
    return {
        "avg_points": df["points"].mean(),
        "avg_goals": df["goals"].mean(),
        "avg_shots": df["shots"].mean(),
        "top_scorers": top_scorers
    }

def train_model(schedule_games):
    rows = []
    for game in schedule_games:
        home_id = game["teams"]["home"]["team"]["id"]
        away_id = game["teams"]["away"]["team"]["id"]
        home_stats = aggregate_team_stats(home_id)
        away_stats = aggregate_team_stats(away_id)

        rows.append({
            "home_points": home_stats["avg_points"],
            "away_points": away_stats["avg_points"],
            "home_goals": home_stats["avg_goals"],
            "away_goals": away_stats["avg_goals"],
            "home_win": 1  # placeholder since we're not using historical outcomes
        })

    df = pd.DataFrame(rows)
    X = df.drop(columns=["home_win"])
    y = df["home_win"]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, "nhl_model.pkl")

def predict_matchup(home_id, away_id):
    model = joblib.load("nhl_model.pkl")
    home_stats = aggregate_team_stats(home_id)
    away_stats = aggregate_team_stats(away_id)

    X = pd.DataFrame([{
        "home_points": home_stats["avg_points"],
        "away_points": away_stats["avg_points"],
        "home_goals": home_stats["avg_goals"],
        "away_goals": away_stats["avg_goals"]
    }])
    proba = model.predict_proba(X)[0]
    return {
        "prediction": "Home Win" if proba[1] > proba[0] else "Away Win",
        "home_prob": proba[1],
        "away_prob": proba[0],
        "home_stats": home_stats,
        "away_stats": away_stats
    }

# --- UI ---

st.title("🏒 NHL Matchup Predictor (Using Official NHL API)")

games = get_today_schedule()
if not games:
    st.warning("No games found today.")
else:
    matchup_options = [f"{g['teams']['away']['team']['name']} @ {g['teams']['home']['team']['name']}" for g in games]
    selected = st.selectbox("Select a matchup to analyze", matchup_options)
    selected_game = games[matchup_options.index(selected)]

    home_id = selected_game["teams"]["home"]["team"]["id"]
    away_id = selected_game["teams"]["away"]["team"]["id"]
    home_name = selected_game["teams"]["home"]["team"]["name"]
    away_name = selected_game["teams"]["away"]["team"]["name"]

    if st.button("Train & Predict"):
        train_model(games)
        result = predict_matchup(home_id, away_id)

        st.success(f"📊 Predicted Outcome: {result['prediction']}")
        st.write(f"**{home_name} Win Probability**: {result['home_prob']:.2%}")
        st.write(f"**{away_name} Win Probability**: {result['away_prob']:.2%}")

        st.markdown("### 🔍 Prediction Explanation")
        st.markdown(f"**Top Home Scorers ({home_name}):**")
        st.json(result["home_stats"]["top_scorers"])
        st.markdown(f"**Top Away Scorers ({away_name}):**")
        st.json(result["away_stats"]["top_scorers"])

