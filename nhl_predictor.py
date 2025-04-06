import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

# ========== Utility ==========

@st.cache_data
def fetch_schedule():
    url = "https://statsapi.web.nhl.com/api/v1/schedule"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            games = []
            for date in data.get("dates", []):
                for game in date["games"]:
                    games.append({
                        "gamePk": game["gamePk"],
                        "date": date["date"],
                        "home": game["teams"]["home"]["team"]["abbreviation"],
                        "away": game["teams"]["away"]["team"]["abbreviation"]
                    })
            return pd.DataFrame(games)
        else:
            st.error(f"❌ Failed to fetch NHL schedule. Status: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ NHL API unreachable: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_past_results():
    # fallback: simulate training set with made-up results
    today = datetime.datetime.today()
    dates = pd.date_range(end=today - datetime.timedelta(days=1), periods=30)
    teams = ['BOS', 'TOR', 'NYR', 'COL', 'STL', 'EDM', 'VGK', 'WPG']
    data = []
    for date in dates:
        h, a = sorted(np.random.choice(teams, 2, replace=False))
        data.append({
            "date": date,
            "home": h,
            "away": a,
            "home_score": int(np.random.randint(2, 6)),
            "away_score": int(np.random.randint(1, 5))
        })
    return pd.DataFrame(data)

def simulate_team_stats(team):
    return {
        "points_avg": round(25 + 10 * np.random.rand(), 2),
        "goals_avg": round(2 + 1.5 * np.random.rand(), 2),
        "scratches": int(np.random.randint(0, 3)),
        "top_scorers": [
            {"Name": f"Player {i+1}", "Points": round(20 + 10 * np.random.rand(), 1)}
            for i in range(3)
        ]
    }

def aggregate_features(row):
    home_stats = simulate_team_stats(row['home'])
    away_stats = simulate_team_stats(row['away'])
    return {
        "home_points_avg": home_stats["points_avg"],
        "away_points_avg": away_stats["points_avg"],
        "home_goals_avg": home_stats["goals_avg"],
        "away_goals_avg": away_stats["goals_avg"],
        "home_win": row["home_score"] > row["away_score"]
    }

def train_model(df):
    features = df.apply(aggregate_features, axis=1, result_type='expand')
    X = features.drop(columns='home_win')
    y = features['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "nhl_model.pkl")
    return model

def predict(home, away):
    model = joblib.load("nhl_model.pkl")
    h_stats = simulate_team_stats(home)
    a_stats = simulate_team_stats(away)
    X = pd.DataFrame([{
        "home_points_avg": h_stats["points_avg"],
        "away_points_avg": a_stats["points_avg"],
        "home_goals_avg": h_stats["goals_avg"],
        "away_goals_avg": a_stats["goals_avg"],
    }])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    explanation = {
        "🏒 Avg Points (Home)": h_stats["points_avg"],
        "🏒 Avg Points (Away)": a_stats["points_avg"],
        "🥅 Avg Goals (Home)": h_stats["goals_avg"],
        "🥅 Avg Goals (Away)": a_stats["goals_avg"],
        "🚑 Scratches (Home)": h_stats["scratches"],
        "🚑 Scratches (Away)": a_stats["scratches"]
    }

    return {
        "result": "Home Win" if pred else "Away Win",
        "prob_home": proba[1],
        "prob_away": proba[0],
        "explanation": explanation,
        "home_top": h_stats["top_scorers"],
        "away_top": a_stats["top_scorers"]
    }

# ========== UI ==========

st.title("🏒 NHL Matchup Predictor")

schedule_df = fetch_schedule()
if schedule_df.empty:
    st.warning("No live schedule found. Using fallback data.")
    schedule_df = fetch_past_results()
    schedule_df['today'] = schedule_df['date'].dt.strftime("%Y-%m-%d")
else:
    schedule_df['today'] = pd.to_datetime(schedule_df['date']).dt.strftime("%Y-%m-%d")

today_str = datetime.datetime.today().strftime("%Y-%m-%d")
todays_games = schedule_df[schedule_df['today'] == today_str]

if not todays_games.empty:
    options = todays_games.apply(lambda row: f"{row['away']} @ {row['home']}", axis=1).tolist()
    matchup = st.selectbox("Select a game to predict from today's matchups:", options)
    away, home = matchup.split(" @ ")

    if st.button("Train & Predict Today’s Game"):
        with st.spinner("Training model..."):
            df = fetch_past_results()
            model = train_model(df)
            result = predict(home, away)

        st.success(f"Prediction: {result['result']}")
        st.info(f"📊 Probability - Home Win: {result['prob_home']:.2%} | Away Win: {result['prob_away']:.2%}")

        st.subheader("📌 Why This Prediction?")
        for key, val in result["explanation"].items():
            st.markdown(f"- **{key}**: {val}")

        st.subheader("🏒 Top Scorers (Home)")
        st.write(pd.DataFrame(result["home_top"]))
        st.subheader("🏒 Top Scorers (Away)")
        st.write(pd.DataFrame(result["away_top"]))

else:
    st.warning("No games scheduled today or unable to load schedule.")

if st.checkbox("Show full fallback training data"):
    st.dataframe(fetch_past_results())

