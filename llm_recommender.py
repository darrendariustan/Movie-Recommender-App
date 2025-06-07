import os
import pandas as pd
import json
from typing import List, Dict, Optional
import streamlit as st

# Handle OpenAI import gracefully for deployment
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

class LLMMovieRecommender:
    """LLM-based movie recommender using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM recommender is available"""
        return OPENAI_AVAILABLE and self.client is not None
    
    def create_movie_database_context(self, movies_df: pd.DataFrame, sample_size: int = 200) -> str:
        """Create a context string with movie database information"""
        # Sample movies to avoid token limits
        if len(movies_df) > sample_size:
            sampled_movies = movies_df.sample(n=sample_size, random_state=42)
        else:
            sampled_movies = movies_df
        
        context = "Available Movies Database:\n"
        for _, movie in sampled_movies.iterrows():
            movie_id = movie['movieId']
            title = movie['title']
            genres = movie['genres'] if pd.notna(movie['genres']) else 'Unknown'
            context += f"ID: {movie_id}, Title: {title}, Genres: {genres}\n"
        
        return context
    

    
    def get_user_preference_analysis(self, user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> str:
        """Analyze user's rating history to understand preferences"""
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            return "New user with no rating history."
        
        # Get user's top rated movies
        top_ratings = user_ratings[user_ratings['rating'] >= 4.0].sort_values('rating', ascending=False)
        user_movies = movies_df[movies_df['movieId'].isin(top_ratings['movieId'])]
        
        # Get user's low rated movies
        low_ratings = user_ratings[user_ratings['rating'] <= 2.0].sort_values('rating', ascending=True)
        disliked_movies = movies_df[movies_df['movieId'].isin(low_ratings['movieId'])]
        
        analysis = f"User {user_id} Movie History Analysis:\n"
        analysis += f"Total Ratings: {len(user_ratings)}\n"
        analysis += f"Average Rating: {user_ratings['rating'].mean():.1f}\n\n"
        
        analysis += f"FAVORITE Movies (4+ stars):\n"
        for _, movie in user_movies.head(8).iterrows():
            rating = top_ratings[top_ratings['movieId'] == movie['movieId']]['rating'].iloc[0]
            analysis += f"- {movie['title']} ({movie['genres']}) - Rated: {rating}\n"
        
        if not disliked_movies.empty:
            analysis += f"\nDISLIKED Movies (2 stars or below):\n"
            for _, movie in disliked_movies.head(5).iterrows():
                rating = low_ratings[low_ratings['movieId'] == movie['movieId']]['rating'].iloc[0]
                analysis += f"- {movie['title']} ({movie['genres']}) - Rated: {rating}\n"
        
        return analysis

    def get_personalized_recommendations(self, 
                                       user_id: int,
                                       user_preferences: str, 
                                       ratings_df: pd.DataFrame,
                                       movies_df: pd.DataFrame,
                                       merged_df: pd.DataFrame) -> List[Dict]:
        """Get personalized movie recommendations based on user's rating history and preferences"""
        
        if not self.is_available():
            return []
        
        # Analyze user's rating history
        user_analysis = self.get_user_preference_analysis(user_id, ratings_df, movies_df)
        
        # Create database context
        db_context = self.create_movie_database_context(merged_df, sample_size=250)
        
        # Build the comprehensive prompt
        if user_preferences and user_preferences.strip():
            prompt = f"""You are an expert movie recommender. You have access to a user's complete movie rating history AND their current preferences. Use BOTH to recommend the perfect movies. Decide the optimal number of recommendations (between 6-15 movies) based on how much data you have.

USER'S MOVIE RATING HISTORY:
{user_analysis}

USER'S CURRENT PREFERENCES:
"{user_preferences}"

{db_context}

Instructions:
1. Analyze BOTH the user's actual rating patterns AND their stated current preferences
2. Look for patterns in their favorites vs dislikes to understand their true taste
3. Consider their current mood/preferences as described in their text
4. Find movies that match both their historical preferences AND current desires
5. Avoid movies similar to ones they rated poorly
6. Don't recommend movies they've already rated
7. Choose 6-15 recommendations based on data richness
8. Return ONLY a JSON list with this exact format:
[
  {{"movieId": 123, "title": "Movie Title", "genres": "Genre1|Genre2", "reason": "Why this matches both their history and current preferences"}},
  ...
]

Respond with ONLY the JSON array, no other text."""
        else:
            prompt = f"""You are an expert movie recommender. You have access to a user's complete movie rating history. Based on their rating patterns, recommend movies they would love. Decide the optimal number of recommendations (between 6-12 movies).

USER'S MOVIE RATING HISTORY:
{user_analysis}

{db_context}

Instructions:
1. Analyze the user's rating patterns to understand their taste
2. Look for patterns in genres, themes, and movie characteristics they love vs dislike
3. Find movies that match their demonstrated preferences
4. Avoid movies similar to ones they rated poorly
5. Don't recommend movies they've already rated
6. Choose 6-12 recommendations based on their rating history
7. Return ONLY a JSON list with this exact format:
[
  {{"movieId": 123, "title": "Movie Title", "genres": "Genre1|Genre2", "reason": "Why this matches their demonstrated preferences"}},
  ...
]

Respond with ONLY the JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a movie recommendation expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                recommendations = json.loads(content)
                return recommendations
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    recommendations = json.loads(json_match.group())
                    return recommendations
                else:
                    st.error("Could not parse LLM response as JSON")
                    return []
                    
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            return []

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or user input"""
    # First try environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # If not in environment, ask user (for local development)
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        
        api_key = st.session_state.openai_api_key
    
    return api_key

def setup_llm_recommender() -> Optional[LLMMovieRecommender]:
    """Setup LLM recommender with proper error handling"""
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package not available. Please install: pip install openai")
        return None
    
    api_key = get_openai_api_key()
    
    if not api_key:
        with st.expander("🔑 OpenAI API Key Required", expanded=True):
            st.warning("Please provide your OpenAI API key to use LLM recommendations.")
            input_key = st.text_input("Enter OpenAI API Key:", type="password", 
                                    help="Your API key is not stored and only used for this session.")
            if input_key:
                st.session_state.openai_api_key = input_key
                st.rerun()
        return None
    
    recommender = LLMMovieRecommender(api_key)
    
    if not recommender.is_available():
        st.error("Failed to initialize OpenAI client. Please check your API key.")
        return None
    
    return recommender 