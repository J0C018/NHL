# NHL Predictor App (Streamlit-Only Version for GitHub Deployment)

# 1. Import Libraries
import requests
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# 2. Fetch NHL Game Data
def get_schedule(season="20232024"):
    url = f"https://statsapi.web.nhl.com/api/v1/schedule?season={season}&sportId=1"
    response = requests.get(url)
    data = response.json()
    games = []
    for date in data['dates']:
        for game in date['games']:
            games.append({
                'gamePk': game['gamePk'],
                'date': game['gameDate'],
                'homeTeam': game['teams']['home']['team']['name'],
                'awayTeam': game['teams']['away']['team']['name'],
                'homeScore': game['teams']['home'].get('score', None),
                'awayScore': game['teams']['away'].get('score', None)
            })
    return pd.DataFrame(games)

# 3. Feature Engineering
def create_features(df):
    df['home_win'] = df['homeScore'] > df['awayScore']
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['homeTeam'] = df['homeTeam'].astype('category')
    df['awayTeam'] = df['awayTeam'].astype('category')
    df['homeTeam_code'] = df['homeTeam'].cat.codes
    df['awayTeam_code'] = df['awayTeam'].cat.codes
    return df

# 4. Train Model
def train_model(df):
    df = df.dropna(subset=['homeScore', 'awayScore'])
    df = create_features(df)
    X = df[['homeTeam_code', 'awayTeam_code', 'dayofweek']]
    y = df['home_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {acc:.2f}")
    joblib.dump((model, df[['homeTeam', 'awayTeam', 'homeTeam_code', 'awayTeam_code']]), 'nhl_model.pkl')
    return model

# 5. Prediction Function
def predict_game(home, away):
    model, mapping_df = joblib.load('nhl_model.pkl')
    home_code = mapping_df[mapping_df['homeTeam'] == home]['homeTeam_code'].values[0]
    away_code = mapping_df[mapping_df['awayTeam'] == away]['awayTeam_code'].values[0]
    day = datetime.datetime.today().weekday()
    features = pd.DataFrame([[home_code, away_code, day]], columns=['homeTeam_code', 'awayTeam_code', 'dayofweek'])
    pred = model.predict(features)
    return "Home Win" if pred[0] else "Away Win"

# 6. Streamlit Frontend

def run_streamlit():
    st.title("🏒 NHL Game Outcome Predictor")
    st.markdown("""
    This app uses NHL stats and machine learning to predict whether the home team will win.
    Just type in the teams and click Predict.
    """)
    home_team = st.text_input("Home Team", "Boston Bruins")
    away_team = st.text_input("Away Team", "Toronto Maple Leafs")
    if st.button("Predict Winner"):
        try:
            prediction = predict_game(home_team, away_team)
            st.success(f"Predicted Outcome: {prediction}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# 7. Main Execution Point
if __name__ == '__main__':
    st.write("Training the NHL model. This may take a few moments...")
    df_games = get_schedule()
    train_model(df_games)
    run_streamlit()
