# src/data_collector.py
"""
Data Collector Module
Supports: Twitter API v2, Reddit API, CSV Upload, Synthetic Data (for testing)
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (use when no API keys)
# ─────────────────────────────────────────────

SYNTHETIC_TEMPLATES = {
    "Roads & Infrastructure": [
        "The roads in {area} are full of potholes. My car got damaged yesterday!",
        "Road near {area} has been broken for 3 months. No repair work done yet.",
        "Street near {area} is flooded during rain due to bad drainage.",
        "Speed bumps on {area} road are missing, causing accidents.",
        "Road construction in {area} has been going on forever with no completion.",
        "Footpath near {area} is completely blocked and unusable for pedestrians.",
    ],
    "Water Supply": [
        "No water supply in {area} for 2 days. When will this be fixed?",
        "Water coming from tap in {area} is yellowish and smells bad.",
        "Water pipe burst near {area} since morning, nobody has come to fix.",
        "Irregular water supply in {area}. We only get water for 1 hour a day.",
        "Low water pressure in {area} is making daily life very difficult.",
    ],
    "Sanitation": [
        "Garbage not collected in {area} for a week. It's overflowing.",
        "Dustbin in {area} is full and nobody has emptied it in days.",
        "Open garbage dumping near {area} market is causing bad smell.",
        "Drains in {area} are clogged causing water to flood the road.",
        "Stray animals spreading garbage in {area}. Need proper bins.",
        "Great job by sanitation team! {area} is spotlessly clean now.",
    ],
    "Electricity": [
        "Power cut in {area} for 6 hours. No response from electricity board.",
        "Street lights in {area} have not been working for weeks. Very dangerous at night.",
        "Transformer in {area} makes loud noise. Please fix it.",
        "Frequent power fluctuations in {area} damaging our appliances.",
        "New solar street lights installed in {area} are amazing. Thank you!",
    ],
    "Public Transport": [
        "Bus service to {area} has been discontinued. Very inconvenient for residents.",
        "Bus stop at {area} has no shelter. We stand in sun/rain while waiting.",
        "Bus number 47 is always late by 30+ minutes at {area} stop.",
        "Auto rickshaws near {area} are overcharging passengers. Please regulate.",
        "New metro station at {area} is excellent! Very clean and fast.",
        "Overcrowding in buses to {area} is dangerous and uncomfortable.",
    ],
    "Parks & Recreation": [
        "Park in {area} is dirty and benches are broken. Kids can't play.",
        "Playground equipment at {area} park is damaged and unsafe for children.",
        "Park in {area} has no lighting. People are afraid to walk at night.",
        "The renovation of {area} park is looking beautiful. Great initiative!",
        "Stray dogs in {area} park are scaring away children and elderly.",
    ],
    "Public Safety": [
        "Theft incidents increasing in {area}. Need more police patrolling.",
        "Street lights broken in {area} making it unsafe to walk at night.",
        "Speeding vehicles near {area} school are dangerous for students.",
        "Traffic signal at {area} junction is not working. Causing accidents.",
        "Police patrolling in {area} has improved. Feeling much safer now.",
    ],
}

CITY_AREAS = {
    "Hyderabad": [
        "Banjara Hills", "Madhapur", "Gachibowli", "Kukatpally", "Hitech City",
        "Ameerpet", "Begumpet", "Secunderabad", "Jubilee Hills", "Kondapur",
        "Miyapur", "LB Nagar", "Dilsukhnagar", "Mehdipatnam", "Tolichowki",
        "Uppal", "Malakpet", "Nampally", "Abids", "Koti",
    ],
    "Bengaluru": [
        "Koramangala", "Indiranagar", "Jayanagar", "BTM Layout", "HSR Layout",
        "Whitefield", "Electronic City", "Hebbal", "Marathahalli", "JP Nagar",
        "Banashankari", "Rajajinagar", "Malleshwaram", "Yelahanka", "Bellandur",
        "Sarjapur", "Vijayanagar", "RT Nagar", "Basavanagudi", "Domlur",
    ],
    "Mumbai": [
        "Andheri", "Bandra", "Dadar", "Kurla", "Thane", "Borivali", "Malad",
        "Goregaon", "Kandivali", "Churchgate", "Colaba", "Worli", "Lower Parel",
        "Powai", "Mulund", "Ghatkopar", "Chembur", "Santacruz", "Vile Parle", "Juhu",
    ],
    "Delhi": [
        "Connaught Place", "Saket", "Dwarka", "Rohini", "Janakpuri", "Pitampura",
        "Lajpat Nagar", "Karol Bagh", "Nehru Place", "Noida", "Gurugram",
        "Vasant Kunj", "Hauz Khas", "Mayur Vihar", "Preet Vihar", "Shahdara",
        "Punjabi Bagh", "Model Town", "Civil Lines", "Chandni Chowk",
    ],
    "Chennai": [
        "Anna Nagar", "T Nagar", "Adyar", "Velachery", "Tambaram",
        "Porur", "Chromepet", "Mylapore", "Nungambakkam", "Perambur",
        "Ambattur", "Avadi", "Sholinganallur", "OMR", "ECR",
        "Guindy", "Kodambakkam", "Vadapalani", "Saidapet", "Tondiarpet",
    ],
    "New York City": [
        "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island",
        "Harlem", "Midtown", "Downtown", "Flushing", "Jamaica",
        "Astoria", "Williamsburg", "Bushwick", "Flatbush", "Ridgewood",
        "Bayside", "Pelham", "Fordham", "Mott Haven", "Riverdale",
    ],
    "Pune": [
        "Koregaon Park", "Kothrud", "Hadapsar", "Viman Nagar", "Baner",
        "Aundh", "Wakad", "Hinjewadi", "Kharadi", "Magarpatta",
        "Shivajinagar", "Deccan", "Pimpri", "Chinchwad", "Warje",
        "Katraj", "Kondhwa", "Dhayari", "Bibwewadi", "Swargate",
    ],
}

def _get_areas(city: str) -> list:
    if city in CITY_AREAS:
        return CITY_AREAS[city]
    for key in CITY_AREAS:
        if key.lower() == city.lower() or city.lower() in key.lower():
            return CITY_AREAS[key]
    return [f"{city} Zone {i}" for i in range(1, 11)] + \
           [f"{city} District {i}" for i in range(1, 11)]

AREAS = CITY_AREAS["Hyderabad"]  # backward compat

SOURCES = ["Twitter", "Reddit", "Survey", "Portal"]


def generate_synthetic_data(n: int = 500, city: str = "Hyderabad") -> pd.DataFrame:
    """
    Generate realistic synthetic city feedback data for testing.
    Uses city-specific area names.
    """
    records = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    areas = _get_areas(city)  # ← city-specific areas

    for i in range(n):
        department = random.choice(list(SYNTHETIC_TEMPLATES.keys()))
        templates = SYNTHETIC_TEMPLATES[department]
        area = random.choice(areas)
        text = random.choice(templates).format(area=area)

        # Realistic date distribution
        random_days = random.randint(0, 90)
        random_hours = random.randint(0, 23)
        timestamp = end_date - timedelta(days=random_days, hours=random_hours)

        # Engagement metrics (simulate social media)
        likes = random.randint(0, 500)
        retweets = random.randint(0, 100)

        records.append({
            "id": f"FB_{i+1:05d}",
            "text": text,
            "area": area,
            "department": department,
            "source": random.choice(SOURCES),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "likes": likes,
            "retweets": retweets,
            "city": city,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# TWITTER COLLECTOR  (requires API key in .env)
# ─────────────────────────────────────────────

def fetch_tweets(keyword: str, max_results: int = 100) -> list[dict]:
    """
    Fetch tweets using Twitter API v2 (Free tier: 500k tweets/month).
    Set TWITTER_BEARER_TOKEN in .env file.
    """
    try:
        import tweepy
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            print("⚠️  No TWITTER_BEARER_TOKEN found. Using synthetic data.")
            return []

        client = tweepy.Client(bearer_token=bearer_token)
        response = client.search_recent_tweets(
            query=f"{keyword} -is:retweet lang:en",
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "public_metrics", "geo"],
        )

        if not response.data:
            return []

        return [
            {
                "id": str(tweet.id),
                "text": tweet.text,
                "timestamp": str(tweet.created_at),
                "likes": tweet.public_metrics.get("like_count", 0),
                "retweets": tweet.public_metrics.get("retweet_count", 0),
                "source": "Twitter",
            }
            for tweet in response.data
        ]

    except Exception as e:
        print(f"Twitter fetch error: {e}")
        return []


# ─────────────────────────────────────────────
# REDDIT COLLECTOR  (requires API key in .env)
# ─────────────────────────────────────────────

def fetch_reddit_posts(subreddit_name: str, query: str, limit: int = 50) -> list[dict]:
    """
    Fetch Reddit posts. Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET in .env
    """
    try:
        import praw
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="SmartCityFeedbackBot/1.0",
        )

        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        for post in subreddit.search(query, limit=limit):
            posts.append({
                "id": post.id,
                "text": f"{post.title}. {post.selftext}"[:512],
                "timestamp": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "likes": post.score,
                "retweets": post.num_comments,
                "source": "Reddit",
            })
        return posts

    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return []


# ─────────────────────────────────────────────
# NYC 311 OPEN DATASET  (no API key needed!)
# ─────────────────────────────────────────────

def load_nyc_311_sample(filepath: str = None, n: int = 1000) -> pd.DataFrame:
    """
    Load NYC 311 Service Requests dataset.
    Download from: https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv
    Or use the built-in sample generator.
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath, nrows=n)
        # Map NYC 311 columns to our schema
        col_map = {
            "Complaint Type": "department",
            "Descriptor": "text",
            "Created Date": "timestamp",
            "Borough": "area",
        }
        df = df.rename(columns=col_map)
        df["source"] = "NYC 311 Portal"
        df["id"] = [f"NYC_{i:05d}" for i in range(len(df))]
        df["likes"] = 0
        df["retweets"] = 0
        df["city"] = "New York City"
        return df[["id", "text", "area", "department", "source", "timestamp", "likes", "retweets", "city"]].dropna()

    # Fallback: synthetic NYC-style data
    return generate_synthetic_data(n=n, city="New York City")

# ─────────────────────────────────────────────
# MAIN LOADER  (used by app.py)
# ─────────────────────────────────────────────

def load_data(
    source: str = "synthetic",
    city: str = "Hyderabad",
    n: int = 500,
    csv_path: str = None,
) -> pd.DataFrame:
    """
    Unified data loader. source can be:
    - 'synthetic'  → generate mock data (default, no API keys needed)
    - 'csv'        → load from uploaded CSV
    - 'twitter'    → fetch from Twitter API
    - 'nyc311'     → load NYC Open Data
    """
    if source == "synthetic":
        df = generate_synthetic_data(n=n, city=city)

    elif source == "csv" and csv_path:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            raise ValueError("CSV must have a 'text' column with feedback text.")
        # Fill missing columns
        for col in ["area", "department", "source", "timestamp", "likes", "retweets"]:
            if col not in df.columns:
                df[col] = "Unknown" if col in ["area", "department", "source"] else (
                    datetime.now().strftime("%Y-%m-%d") if col == "timestamp" else 0
                )
        df["city"] = city
        df["id"] = [f"CSV_{i:05d}" for i in range(len(df))]

    elif source == "nyc311":
        df = load_nyc_311_sample(n=n)

    else:
        df = generate_synthetic_data(n=n, city=city)

    # Ensure text column is string and not empty
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 10].reset_index(drop=True)

    # Save raw copy
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/latest_feedback.csv", index=False)

    return df


if __name__ == "__main__":
    df = load_data(source="synthetic", n=200)
    print(f"✅ Generated {len(df)} feedback records")
    print(df.head())
    print(f"\nDepartment distribution:\n{df['department'].value_counts()}")