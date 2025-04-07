import streamlit as st
import requests
from datetime import datetime

# Set up page
st.set_page_config(page_title="NHL Schedule - RapidAPI Live Test")
st.title("🏒 NHL Schedule - RapidAPI Live Test")

# Date Setup
today = datetime.utcnow()
year = today.strftime("%Y")
month = today.strftime("%m")
day = today.strftime("%d")

st.markdown(f"🗓️ Showing games for **{today.strftime('%B %d, %Y')}** (UTC)")

# API credentials (fetched from Railway variables via st.secrets)
API_KEY = st.secrets["RAPIDAPI_KEY"]
API_HOST = st.secrets["RAPIDAPI_HOST"]

# API URL
url = f"https://{API_HOST}/nhlschedule?year={year}&month={month}&day={day}"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# Fetch data
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    games = data.get("games", [])
    if games:
        st.success(f"✅ {len(games)} games found today.\n")
        for game in games:
            name = game.get("name", "Unnamed Game")
            date = game.get("date", "No date")
            venue = (
                game.get("competitions", [{}])[0]
                .get("venue", {})
                .get("fullName", "Unknown Venue")
            )
            st.subheader(name)
            st.write(f"🕒 **Date**: {date}")
            st.write(f"📍 **Venue**: {venue}")
            st.markdown("---")
    else:
        st.info("🔍 No valid matchups found for today.")

except requests.exceptions.RequestException as e:
    st.error(f"❌ Failed to fetch schedule: {e}")

