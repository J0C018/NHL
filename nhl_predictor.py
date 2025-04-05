import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
import time

# Debug toggle
DEBUG_MODE = st.sidebar.checkbox("🪵 Enable Debug Mode", value=False)

# Load API key
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# Team abbreviation corrections
TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

# Default fallback when stats are missing or broken
def default_team_stats():
    return {
        "goals_avg": 0,
        "shots_avg": 0,
        "points_avg": 0,
        "top_scorers": [],
        "scratched": []
    }

# Get full season schedule
def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Get today's games
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

# Get injury list
@st.cache_data(show_spinner=False)
def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Get player stats for team
@st.cache_data(show_spinner=False)
def get_player_stats_by_team(team):
    team_fixed = TEAM_ABBREVIATION_FIX.get(team, team)
    time.sleep(0.5)
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team_fixed}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        if DEBUG_MODE:
            st.warning(f"❌ Failed to fetch stats for {team_fixed} – Status {r.status_code}")
        return pd.DataFrame()
    try:
        data = r.json()
        if not isinstance(data, list):
            if DEBUG_MODE:
                st.warning(f"⚠️ Unexpected structure for {team_fixed}: {data}")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'Team' not in df.columns:
            if DEBUG_MODE:
                st.warning(f"⚠️ 'Team' column missing in {team_fixed} response")
            return pd.DataFrame()
        return df
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"⚠️ Error parsing stats for {team_fixed}: {e}")
        return pd.DataFrame()

# Compile stats for training or prediction
def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    required_columns = {'Name', 'Team', 'Goals', 'ShotsOnGoal', 'Points'}

    if df.empty:
        if DEBUG_MODE:
            st.warning(f"⚠️ No player stats for team {team}")
        return default_team_stats()

    if not required_columns.issubset(df.columns):
        if DEBUG_MODE:
            st.warning(f"⚠️ Missing expected columns for team {team}: {df.columns.tolist()}")
        return default_team_stats()

    injured_players = injury_df[injury_df['Team'] == team]['Name'].tolist()
    scratched = df[df['Name'].isin(injured_players)][['Name', 'Position']].to_dict('records')
    df_active = df[~df['Name'].isin(injured_players)]
    top_scorers = df_active.sort_values('Points', ascending=False).head(3)[['Name', 'Points']].to_dict('records')
    return {
        "goals_avg": df_active['Goals'].mean(),
        "shots_avg": df_active['ShotsOnGoal'].mean(),
        "points_avg": df_active['Points'].mean(),
        "top_scorers": top_scorers,
        "scratched": scratched
    }

# Train model on full season
def train_model(df, injury_df):
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore']).copy()
    df['home_win'] = df['HomeTeamScore'] > df['AwayTeamScore']
    df['date'] = pd.to_datetime(df['Day'], errors='coerce')
    df['dayofweek'] = df['date'].dt.dayofweek

    stat_rows = []
    for _, row in df.iterrows():
        home_stats = aggregate_team_stats(row['HomeTeam'], injury_df)
        away_stats = aggregate_team_stats(row['AwayTeam'], injury_df)
        stat_rows.append({
            "homeTeam": row['HomeTeam'],
            "awayTeam": row['AwayTeam'],
            "dayofweek": row['dayofweek'],
            "home_goals_avg": home_stats['goals_avg'],
            "away_goals_avg": away_stats['goals_avg'],
            "home_points_avg": home_stats['points_avg'],
            "away_points_avg": away_stats['points_avg'],
            "home_win": row['home_win']
        })

    model_df = pd.DataFrame(stat_rows)
    X = model_df.drop(columns=['homeTeam', 'awayTeam', 'home_win'])
    y = model_df['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, model_df), 'nhl_model.pkl')

# Predict today's game
def predict_game(home, away, injury_df):
    model, df = joblib.load('nhl_model.pkl')
    day = datetime.datetime.today().weekday()
    home_stats = aggregate_team_stats(home, injury_df)
    away_stats = aggregate_team_stats(away, injury_df)
    X_pred = pd.DataFrame([{
        "dayofweek": day,
        "home_goals_avg": home_stats['goals_avg'],
        "away_goals_avg": away_stats['goals_avg'],
        "home_points_avg": home_stats['points_avg'],
        "away_points_avg": away_stats['points_avg']
    }])
    proba = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)
    return {
        "result": "Home Win" if pred[0] else "Away Win",
        "prob_home_win": proba[1],
        "prob_away_win": proba[0],
        "home_top_scorers": home_stats['top_scorers'],
        "away_top_scorers": away_stats['top_scorers'],
        "home_scratched": home_stats['scratched'],
        "away_scratched": away_stats['scratched']
    }

# --- UI ---
schedule_df = get_schedule()
if not schedule_df.empty:
    schedule_df = schedule_df.dropna(subset=['HomeTeam', 'AwayTeam'])
    matchups = schedule_df[['HomeTeam', 'AwayTeam']].drop_duplicates()
    matchup_options = matchups.apply(lambda row: f"{row['AwayTeam']} @ {row['HomeTeam']}", axis=1).tolist()
    selected_matchup = st.selectbox("Select a game to predict from today's matchups:", matchup_options)
    away, home = selected_matchup.split(" @ ")

    if st.button("Train & Predict Today’s Game"):
        full_season_df = get_full_season_schedule()
        injuries = get_injuries()
        if full_season_df.empty:
            st.error("No season data found to train on.")
        else:
            full_season_df['Day'] = pd.to_datetime(full_season_df['Day'], errors='coerce')
            st.write(f"📅 Date range of training data: {full_season_df['Day'].min()} to {full_season_df['Day'].max()}")
            try:
                train_model(full_season_df, injuries)
                result = predict_game(home, away, injuries)
                st.success(f"Prediction: {result['result']}")
                st.info(f"📊 Probability - Home Win: {result['prob_home_win']:.2%} | Away Win: {result['prob_away_win']:.2%}")

                st.subheader("🏒 Top Scorers (Home)")
                st.json(result['home_top_scorers'])
                st.subheader("🏒 Top Scorers (Away)")
                st.json(result['away_top_scorers'])
                st.subheader("🚑 Scratched Players (Home)")
                st.json(result['home_scratched'])
                st.subheader("🚑 Scratched Players (Away)")
                st.json(result['away_scratched'])

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No games scheduled today.")

if st.checkbox("Show full season schedule data"):
    df = get_full_season_schedule()
    st.dataframe(df)




