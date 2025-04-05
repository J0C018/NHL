import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
import time

# Load API key
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# Abbreviation fixes
TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

# GET today's matchups
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

# GET full season schedule
def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# GET player stats (with retry logic)
def get_player_stats_by_team(team):
    team_fixed = TEAM_ABBREVIATION_FIX.get(team, team)
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team_fixed}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}

    def fetch():
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return pd.DataFrame(data)
        except Exception as e:
            print(f"Retry failed for {team_fixed}: {e}")
        return None

    time.sleep(0.5)
    df = fetch()
    if df is None:
        time.sleep(1)
        df = fetch()
    if df is None or df.empty:
        st.warning(f"⚠️ No player data found for team {team} ({team_fixed}). Skipping.")
        return pd.DataFrame()
    return df

# GET injuries
def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

# Aggregate stats for team
def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    if df.empty or 'Name' not in df:
        return {
            "goals_avg": 0, "shots_avg": 0, "points_avg": 0,
            "top_scorers": [], "scratched": []
        }

    injured_players = injury_df[injury_df['Team'] == team]['Name'].tolist()
    scratched = df[df['Name'].isin(injured_players)][['Name', 'Position']].to_dict('records')
    df_active = df[~df['Name'].isin(injured_players)]

    top_scorers = df_active.sort_values('Points', ascending=False).head(3)
    top_scorers = top_scorers[['Name', 'Points']].to_dict('records')

    return {
        "goals_avg": df_active['Goals'].mean(),
        "shots_avg": df_active['ShotsOnGoal'].mean(),
        "points_avg": df_active['Points'].mean(),
        "top_scorers": top_scorers,
        "scratched": scratched
    }

# Train model
def train_model(df, injury_df):
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore'])
    df = df.copy()
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

# Predict a single game
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

    explanation = {
        "home_points_avg": home_stats['points_avg'],
        "away_points_avg": away_stats['points_avg'],
        "home_scratches": len(home_stats['scratched']),
        "away_scratches": len(away_stats['scratched']),
        "home_top": home_stats['top_scorers'],
        "away_top": away_stats['top_scorers']
    }

    return {
        "result": "Home Win" if pred[0] else "Away Win",
        "prob_home_win": proba[1],
        "prob_away_win": proba[0],
        "home_top_scorers": home_stats['top_scorers'],
        "away_top_scorers": away_stats['top_scorers'],
        "home_scratched": home_stats['scratched'],
        "away_scratched": away_stats['scratched'],
        "explanation": explanation
    }

# UI
st.title("🏒 NHL Matchup Predictor")

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
            st.write(f"📅 Training data from {full_season_df['Day'].min()} to {full_season_df['Day'].max()}")
            try:
                train_model(full_season_df, injuries)
                result = predict_game(home, away, injuries)

                st.success(f"Prediction: {result['result']}")
                st.info(f"📊 Probability - Home Win: {result['prob_home_win']:.2%} | Away Win: {result['prob_away_win']:.2%}")

                st.subheader("📣 Why This Prediction?")
                exp = result["explanation"]
                st.markdown(f"""
                - **Avg Points (Home)**: {exp['home_points_avg']:.2f}  
                - **Avg Points (Away)**: {exp['away_points_avg']:.2f}  
                - **Top Home Scorers**: {', '.join([p['Name'] for p in exp['home_top']]) or 'N/A'}  
                - **Top Away Scorers**: {', '.join([p['Name'] for p in exp['away_top']]) or 'N/A'}  
                - **Home Scratches**: {exp['home_scratches']}  
                - **Away Scratches**: {exp['away_scratches']}  
                """)

                st.subheader("🏒 Top Scorers (Home)")
                st.write(result['home_top_scorers'] or "No data available.")
                st.subheader("🏒 Top Scorers (Away)")
                st.write(result['away_top_scorers'] or "No data available.")
                st.subheader("🚑 Scratched Players (Home)")
                st.write(result['home_scratched'] or "None listed.")
                st.subheader("🚑 Scratched Players (Away)")
                st.write(result['away_scratched'] or "None listed.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No games scheduled today.")

if st.checkbox("Show full season schedule data"):
    df = get_full_season_schedule()
    st.dataframe(df)


