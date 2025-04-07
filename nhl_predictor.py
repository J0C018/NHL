import streamlit as st
import requests
import datetime

st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

API_KEY = st.secrets["RAPIDAPI_KEY"]
API_HOST = st.secrets["RAPIDAPI_HOST"]
API_BASE = f"https://{API_HOST}"

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST
}

def get_today_schedule():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"{API_BASE}/schedule?date={today}"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        return data.get("games", [])
    except Exception as e:
        st.error(f"❌ Failed to load schedule: {e}")
        return []

def get_team_name(team_id):
    url = f"{API_BASE}/teams"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        teams = r.json().get("teams", [])
        for team in teams:
            if str(team.get("id")) == str(team_id):
                return team.get("name", f"Team {team_id}")
        return f"Team {team_id}"
    except Exception as e:
        return f"Team {team_id}"

def predict_matchup(home_team, away_team):
    # 🧠 Placeholder logic
    if len(home_team) > len(away_team):
        return f"Prediction: {home_team} is slightly favored."
    else:
        return f"Prediction: {away_team} is slightly favored."

st.title("🏒 NHL Matchup Predictor")

games = get_today_schedule()

if games:
    matchup_options = []
    for g in games:
        home = get_team_name(g["homeTeam"])
        away = get_team_name(g["awayTeam"])
        matchup_options.append((f"{away} @ {home}", home, away))

    selected_label = st.selectbox("Select a game:", [x[0] for x in matchup_options])
    if st.button("🔮 Predict Result"):
        selected_game = next(x for x in matchup_options if x[0] == selected_label)
        home, away = selected_game[1], selected_game[2]
        result = predict_matchup(home, away)
        st.success(result)
else:
    st.info("📅 No NHL games found for today.")

