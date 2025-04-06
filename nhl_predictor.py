import streamlit as st
import pandas as pd
import requests
import datetime

st.set_page_config(page_title="🏒 NHL Matchup Predictor", page_icon="🏒")

PROXY_URL = "https://web-production-7027.up.railway.app"

def get_teams():
    try:
        r = requests.get(f"{PROXY_URL}/api/v1/teams")
        r.raise_for_status()
        teams = r.json()
        return {team['id']: team for team in teams}
    except Exception as e:
        st.error(f"❌ Failed to load teams: {e}")
        return {}

def get_today_schedule():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"{PROXY_URL}/api/v1/schedule/{today}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json().get("games", [])
    except Exception as e:
        st.error(f"❌ Failed to fetch schedule: {e}")
        return []

def get_team_stats(team_id):
    try:
        url = f"{PROXY_URL}/api/v1/club-stats-season/{team_id}/20232024"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        stats = data.get("regularSeason", {}).get("teamStats", {})
        return {
            "goalsPerGame": stats.get("goalsPerGame"),
            "powerPlayPct": stats.get("powerPlayPct"),
            "penaltyKillPct": stats.get("penaltyKillPct"),
        }
    except Exception as e:
        return {
            "goalsPerGame": 0,
            "powerPlayPct": 0,
            "penaltyKillPct": 0,
            "error": str(e)
        }

def predict_winner(home_stats, away_stats):
    if home_stats["goalsPerGame"] > away_stats["goalsPerGame"]:
        return "🏠 Home team is favored based on scoring."
    elif away_stats["goalsPerGame"] > home_stats["goalsPerGame"]:
        return "🛫 Away team has stronger scoring and is slightly favored."
    else:
        return "🤝 It’s a very close matchup!"

# Streamlit UI
st.title("🏒 NHL Matchup Predictor")

teams = get_teams()
games = get_today_schedule()

if teams and games:
    matchups = []
    for game in games:
        home_id = game["homeTeam"]["id"]
        away_id = game["awayTeam"]["id"]
        home_name = teams[home_id]["fullName"]
        away_name = teams[away_id]["fullName"]
        matchups.append((f"{away_name} @ {home_name}", home_id, away_id, home_name, away_name))

    options = [m[0] for m in matchups]
    selected = st.selectbox("Select a game from today's matchups:", options)

    if st.button("🔮 Predict Result"):
        selected_match = next(m for m in matchups if m[0] == selected)
        _, home_id, away_id, home_name, away_name = selected_match

        st.subheader("📊 Team Stats")

        home_stats = get_team_stats(home_id)
        away_stats = get_team_stats(away_id)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🏠 {home_name}")
            st.write(home_stats)
        with col2:
            st.markdown(f"### 🛫 {away_name}")
            st.write(away_stats)

        prediction = predict_winner(home_stats, away_stats)
        st.success(prediction)

else:
    st.warning("No games or team data available.")

