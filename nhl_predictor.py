import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
import time

# Load API key from Streamlit secrets
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# Fix known abbreviation mismatches
TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

# Pull full season schedule
def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Pull today’s games
def get_schedule():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.sportsdata.io/v3/nhl/scores/json/GamesByDate/{today}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    if 'Status' in df.columns:
        df = df[df['Status'] != 'Final']
    return df

# Get player stats with retry
def get_player_stats_by_team(team):
    team_fixed = TEAM_ABBREVIATION_FIX.get(team, team)
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team_fixed}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}

    def fetch_stats():
        try:
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                return None
            data = r.json()
            return pd.DataFrame(data) if isinstance(data, list) else None
        except Exception:
            return None

    time.sleep(0.5)
    stats_df = fetch_stats()
    if stats_df is None or stats_df.empty:
        time.sleep(1)
        stats_df = fetch_stats()

    if stats_df is None or stats_df.empty:
        st.warning(f"⚠️ No player data found for team {team_fixed}")
        return pd.DataFrame()
    return stats_df

# Pull injury list
def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Get per-team stats
def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    required = {'Name', 'Team', 'Goals', 'ShotsOnGoal', 'Points'}

    if df.empty or not required.issubset(df.columns):
        return {
            "goals_avg": 0, "shots_avg": 0, "points_avg": 0,
            "top_scorers": "N/A", "scratched": "None listed"
        }

    injured = injury_df[injury_df['Team'] == team]['Name'].tolist()
    df_active = df[~df['Name'].isin(injured)]

    top_scorers = df_active.sort_values('Points', ascending=False).head(3)
    return {
        "goals_avg": df_active['Goals'].mean(),
        "shots_avg": df_active['ShotsOnGoal'].mean(),
        "points_avg": df_active['Points'].mean(),
        "top_scorers": ", ".join(top_scorers['Name'].tolist()) if not top_scorers.empty else "N/A",
        "scratched": len(injured)
    }

# Train model
def train_model(df, injury_df):
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore'])
    df['home_win'] = df['HomeTeamScore'] > df['AwayTeamScore']
    df['date'] = pd.to_datetime(df['Day'], errors='coerce')
    df['dayofweek'] = df['date'].dt.dayofweek

    model_rows = []
    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        home_stats = aggregate_team_stats(home, injury_df)
        away_stats = aggregate_team_stats(away, injury_df)

        model_rows.append({
            "dayofweek": row['dayofweek'],
            "home_goals_avg": home_stats['goals_avg'],
            "away_goals_avg": away_stats['goals_avg'],
            "home_points_avg": home_stats['points_avg'],
            "away_points_avg": away_stats['points_avg'],
            "home_win": row['home_win']
        })

    model_df = pd.DataFrame(model_rows)
    X = model_df.drop(columns=['home_win'])
    y = model_df['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, model_df), 'nhl_model.pkl')

# Predict game outcome
def predict_game(home, away, injury_df):
    model, df = joblib.load('nhl_model.pkl')
    day = datetime.datetime.today().weekday()
    home_stats = aggregate_team_stats(home, injury_df)
    away_stats = aggregate_team_stats(away, injury_df)

    X = pd.DataFrame([{
        "dayofweek": day,
        "home_goals_avg": home_stats['goals_avg'],
        "away_goals_avg": away_stats['goals_avg'],
        "home_points_avg": home_stats['points_avg'],
        "away_points_avg": away_stats['points_avg']
    }])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "result": "Home Win" if pred else "Away Win",
        "prob_home_win": proba[1],
        "prob_away_win": proba[0],
        "explain": {
            "Avg Points (Home)": home_stats['points_avg'],
            "Avg Points (Away)": away_stats['points_avg'],
            "Top Home Scorers": home_stats['top_scorers'],
            "Top Away Scorers": away_stats['top_scorers'],
            "Home Scratches": home_stats['scratched'],
            "Away Scratches": away_stats['scratched']
        }
    }

# ------------------- Streamlit UI -------------------

st.title("🏒 NHL Matchup Predictor")

schedule_df = get_schedule()
if not schedule_df.empty:
    matchups = schedule_df[['HomeTeam', 'AwayTeam']].dropna()
    matchup_options = matchups.apply(lambda row: f"{row['AwayTeam']} @ {row['HomeTeam']}", axis=1).tolist()
    selected = st.selectbox("Select a game to predict from today's matchups:", matchup_options)
    away, home = selected.split(" @ ")

    if st.button("Train & Predict Today’s Game"):
        full_df = get_full_season_schedule()
        injuries = get_injuries()
        if full_df.empty:
            st.error("No season data available.")
        else:
            full_df['Day'] = pd.to_datetime(full_df['Day'], errors='coerce')
            st.write(f"📅 Training data from {full_df['Day'].min()} to {full_df['Day'].max()}")

            try:
                train_model(full_df, injuries)
                result = predict_game(home, away, injuries)
                st.success(f"Prediction: **{result['result']}**")
                st.info(f"📊 Probability - Home Win: {result['prob_home_win']:.2%} | Away Win: {result['prob_away_win']:.2%}")

                st.markdown("### 🔍 Why This Prediction?")
                for k, v in result['explain'].items():
                    st.markdown(f"- **{k}**: {v}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No games scheduled today.")


