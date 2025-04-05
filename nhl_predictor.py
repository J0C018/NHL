import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
import time
from sklearn.ensemble import RandomForestClassifier

# Load secret
SPORTSDATA_API_KEY = st.secrets["SPORTSDATA_API_KEY"]

# Team name fixer
TEAM_ABBREVIATION_FIX = {
    'NAS': 'NSH', 'VEG': 'VGK', 'PHO': 'ARI', 'TAM': 'TB', 'LA': 'LAK', 'NJ': 'NJD',
    'SJ': 'SJS', 'CLS': 'CBJ', 'MON': 'MTL', 'CHI': 'CHI', 'STL': 'STL', 'COL': 'COL',
    'NYI': 'NYI', 'NYR': 'NYR', 'PIT': 'PIT', 'FLA': 'FLA', 'BUF': 'BUF', 'BOS': 'BOS',
    'CGY': 'CGY', 'CAR': 'CAR', 'EDM': 'EDM', 'VAN': 'VAN', 'WSH': 'WSH', 'SEA': 'SEA',
    'OTT': 'OTT', 'DET': 'DET', 'DAL': 'DAL', 'WPG': 'WPG', 'TOR': 'TOR', 'MIN': 'MIN'
}

# --- API CALLS ---
def get_schedule():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.sportsdata.io/v3/nhl/scores/json/GamesByDate/{today}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.error("❌ Failed to fetch today's schedule")
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    return df[df['Status'] != 'Final'] if 'Status' in df.columns else df

def get_full_season_schedule():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Games/2024"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.error("❌ Failed to fetch full season data.")
        return pd.DataFrame()
    return pd.DataFrame(r.json())

def get_injuries():
    url = "https://api.sportsdata.io/v3/nhl/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        st.error("❌ Failed to fetch injury data.")
        return pd.DataFrame()
    return pd.DataFrame(r.json())

def get_player_stats_by_team(team):
    team_fixed = TEAM_ABBREVIATION_FIX.get(team, team)
    url = f"https://api.sportsdata.io/v3/nhl/stats/json/PlayerSeasonStatsByTeam/2024/{team_fixed}"
    headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
    time.sleep(0.3)
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        st.warning(f"⚠️ Could not fetch stats for team {team_fixed} — {r.status_code}")
        print(f"[Stats API ERROR] {team_fixed} — {r.status_code} — {r.text}")
        return pd.DataFrame()

    try:
        data = r.json()
        if not isinstance(data, list) or not data:
            st.warning(f"⚠️ No player data found for team {team_fixed}")
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"🚨 Error parsing player stats for {team_fixed}: {e}")
        return pd.DataFrame()

# --- STAT AGGREGATION ---
def aggregate_team_stats(team, injury_df):
    df = get_player_stats_by_team(team)
    if df.empty or 'Points' not in df.columns:
        return {"points_avg": 0, "top_scorers": "N/A", "scratched": "N/A"}

    injured_players = injury_df[injury_df['Team'] == team]['Name'].tolist()
    df_active = df[~df['Name'].isin(injured_players)]
    scratched = df[df['Name'].isin(injured_players)][['Name', 'Position']].to_dict('records')

    top_scorers = df_active.sort_values('Points', ascending=False).head(3)
    top_scorers = top_scorers[['Name', 'Points']].to_dict('records') if not top_scorers.empty else "N/A"

    return {
        "points_avg": df_active['Points'].mean(),
        "top_scorers": top_scorers,
        "scratched": scratched
    }

# --- MODEL ---
def train_model(df, injury_df):
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'HomeTeamScore', 'AwayTeamScore'])
    df['home_win'] = df['HomeTeamScore'] > df['AwayTeamScore']
    df['dayofweek'] = pd.to_datetime(df['Day'], errors='coerce').dt.dayofweek

    stat_rows = []
    for _, row in df.iterrows():
        home_stats = aggregate_team_stats(row['HomeTeam'], injury_df)
        away_stats = aggregate_team_stats(row['AwayTeam'], injury_df)
        stat_rows.append({
            "homeTeam": row['HomeTeam'],
            "awayTeam": row['AwayTeam'],
            "dayofweek": row['dayofweek'],
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

# --- PREDICTION ---
def predict_game(home, away, injury_df):
    model, _ = joblib.load('nhl_model.pkl')
    day = datetime.datetime.today().weekday()

    home_stats = aggregate_team_stats(home, injury_df)
    away_stats = aggregate_team_stats(away, injury_df)

    X_pred = pd.DataFrame([{
        "dayofweek": day,
        "home_points_avg": home_stats['points_avg'],
        "away_points_avg": away_stats['points_avg']
    }])
    pred = model.predict(X_pred)
    proba = model.predict_proba(X_pred)[0]

    return {
        "result": "Home Win" if pred[0] else "Away Win",
        "prob_home_win": proba[1],
        "prob_away_win": proba[0],
        "explain": {
            "home_points": home_stats['points_avg'],
            "away_points": away_stats['points_avg'],
            "home_scorers": home_stats['top_scorers'],
            "away_scorers": away_stats['top_scorers'],
            "home_scratches": home_stats['scratched'],
            "away_scratches": away_stats['scratched']
        }
    }

# --- STREAMLIT UI ---
st.set_page_config(page_title="NHL Matchup Predictor", page_icon="🏒")

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
            st.error("No season data found to train.")
        else:
            st.write(f"📅 Training data from {full_season_df['Day'].min()} to {full_season_df['Day'].max()}")
            try:
                train_model(full_season_df, injuries)
                result = predict_game(home, away, injuries)

                st.success(f"🏆 Prediction: {result['result']}")
                st.info(f"📊 Probability - Home Win: {result['prob_home_win']:.2%} | Away Win: {result['prob_away_win']:.2%}")

                st.subheader("🔍 Why This Prediction?")
                st.markdown(f"""
                - **Avg Points (Home)**: {result['explain']['home_points']:.2f}
                - **Avg Points (Away)**: {result['explain']['away_points']:.2f}
                - **Top Home Scorers**: {result['explain']['home_scorers']}
                - **Top Away Scorers**: {result['explain']['away_scorers']}
                - **Home Scratches**: {len(result['explain']['home_scratches'])}
                - **Away Scratches**: {len(result['explain']['away_scratches'])}
                """)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.warning("No games scheduled today.")

if st.checkbox("📊 Show full season schedule data"):
    df = get_full_season_schedule()
    st.dataframe(df)


