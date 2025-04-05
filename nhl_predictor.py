import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
import time

# Load API key
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# TEAM ABBREVIATION FIXES
TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

# API CALLS
def get_schedule():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.sportsdata.io/v3/nhl/scores/json/GamesByDate/{today}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    return df[df['Status'] != 'Final'] if 'Status' in df.columns else df

def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()

def get_player_stats_by_team(team):
    team_fixed = TEAM_ABBREVIATION_FIX.get(team, team)
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team_fixed}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    time.sleep(0.4)
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.warning(f"⚠️ No player data found for team {team_fixed}")
        return pd.DataFrame()
    try:
        data = r.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if df.empty:
                st.warning(f"⚠️ No player data returned for team {team_fixed}")
            return df
        else:
            st.warning(f"⚠️ Unexpected response for {team_fixed}: {data}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Parsing error for {team_fixed}: {e}")
        return pd.DataFrame()

def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
        else:
            st.error("❌ Failed to fetch injury data.")
            return pd.DataFrame()
    except:
        st.error("❌ Exception occurred while fetching injury data.")
        return pd.DataFrame()

# FEATURE AGGREGATION
def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    injured_players = injury_df[injury_df['Team'] == team]['Name'].tolist()
    scratched = df[df['Name'].isin(injured_players)][['Name', 'Position']].to_dict('records') if not df.empty else []
    df_active = df[~df['Name'].isin(injured_players)] if not df.empty else pd.DataFrame()
    
    if not df_active.empty and {'Goals', 'ShotsOnGoal', 'Points'}.issubset(df_active.columns):
        top_scorers = df_active.sort_values('Points', ascending=False).head(3)[['Name', 'Points']].to_dict('records')
        return {
            "goals_avg": df_active['Goals'].mean(),
            "shots_avg": df_active['ShotsOnGoal'].mean(),
            "points_avg": df_active['Points'].mean(),
            "top_scorers": top_scorers,
            "scratched": scratched
        }
    else:
        return {
            "goals_avg": 0, "shots_avg": 0, "points_avg": 0,
            "top_scorers": [], "scratched": scratched
        }

# MODELING
def train_model(df, injury_df):
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore'])
    df['home_win'] = df['HomeTeamScore'] > df['AwayTeamScore']
    df['date'] = pd.to_datetime(df['Day'], errors='coerce')
    df['dayofweek'] = df['date'].dt.dayofweek
    features = []
    for _, row in df.iterrows():
        home_stats = aggregate_team_stats(row['HomeTeam'], injury_df)
        away_stats = aggregate_team_stats(row['AwayTeam'], injury_df)
        features.append({
            "homeTeam": row['HomeTeam'],
            "awayTeam": row['AwayTeam'],
            "dayofweek": row['dayofweek'],
            "home_goals_avg": home_stats['goals_avg'],
            "away_goals_avg": away_stats['goals_avg'],
            "home_points_avg": home_stats['points_avg'],
            "away_points_avg": away_stats['points_avg'],
            "home_win": row['home_win']
        })
    model_df = pd.DataFrame(features)
    X = model_df.drop(columns=['homeTeam', 'awayTeam', 'home_win'])
    y = model_df['home_win']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, model_df), 'nhl_model.pkl')

# PREDICT
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
        "away_scratched": away_stats['scratched'],
        "home_points_avg": home_stats['points_avg'],
        "away_points_avg": away_stats['points_avg']
    }

# --- UI ---
st.set_page_config(page_title="NHL Predictor", layout="centered")
st.title("🏒 NHL Matchup Predictor")

schedule_df = get_schedule()
if not schedule_df.empty:
    matchups = schedule_df.dropna(subset=['HomeTeam', 'AwayTeam'])[['HomeTeam', 'AwayTeam']].drop_duplicates()
    matchup_options = matchups.apply(lambda row: f"{row['AwayTeam']} @ {row['HomeTeam']}", axis=1).tolist()
    selected_matchup = st.selectbox("Select a game to predict from today's matchups:", matchup_options)
    away, home = selected_matchup.split(" @ ")

    if st.button("Train & Predict Today’s Game"):
        full_season_df = get_full_season_schedule()
        injuries = get_injuries()

        if full_season_df.empty:
            st.error("No data available for training.")
        else:
            full_season_df['Day'] = pd.to_datetime(full_season_df['Day'], errors='coerce')
            st.write(f"📅 Training data from {full_season_df['Day'].min()} to {full_season_df['Day'].max()}")
            try:
                train_model(full_season_df, injuries)
                result = predict_game(home, away, injuries)

                st.success(f"📊 Prediction: {result['result']}")
                st.info(f"Probability - Home Win: {result['prob_home_win']:.2%} | Away Win: {result['prob_away_win']:.2%}")

                st.markdown("### 🔍 Why This Prediction?")
                st.markdown(f"- **Avg Points (Home)**: {result['home_points_avg']:.2f}")
                st.markdown(f"- **Avg Points (Away)**: {result['away_points_avg']:.2f}")
                st.markdown(f"- **Top Home Scorers**: {', '.join([p['Name'] for p in result['home_top_scorers']]) or 'N/A'}")
                st.markdown(f"- **Top Away Scorers**: {', '.join([p['Name'] for p in result['away_top_scorers']]) or 'N/A'}")
                st.markdown(f"- **Home Scratches**: {len(result['home_scratched'])}")
                st.markdown(f"- **Away Scratches**: {len(result['away_scratched'])}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No games scheduled today.")

if st.checkbox("Show full season schedule data"):
    st.dataframe(get_full_season_schedule())

