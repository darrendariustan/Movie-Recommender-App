import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

import pandas as pd
import numpy as np
import streamlit as st

# Set page title and favicon - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Movies Recommender", page_icon=":clapper:")
from helper_functions import (
    load_svd_model, 
    load_neural_model, 
    get_unified_recommendations, 
    compare_model_recommendations,
    get_svd_recommendations,
    NeuralRecommender,
    get_unique_genres,
    filter_recommendations_by_genre,
    find_similar_movies,
    get_movie_info_by_id,
    get_hybrid_recommendations,
    analyze_user_characteristics,
    find_movies_by_genre_similarity,
    get_cached_similarity_data
)

# LLM recommender import with lazy loading
LLM_AVAILABLE = False
setup_llm_recommender = None
LLMMovieRecommender = None

def load_llm_functions():
    """Lazy load LLM functions only when needed"""
    global LLM_AVAILABLE, setup_llm_recommender, LLMMovieRecommender
    try:
        from llm_recommender import setup_llm_recommender, LLMMovieRecommender
        LLM_AVAILABLE = True
        return True
    except ImportError:
        LLM_AVAILABLE = False
        return False

# Load data with caching for faster startup
@st.cache_data
def load_app_data():
    """Load and cache main data files"""
    merged_df = pd.read_csv('output_data/merged_movies.csv')
    movies_df = pd.read_csv('dataset/movies.csv') 
    ratings_df = pd.read_csv('dataset/ratings.csv')
    return merged_df, movies_df, ratings_df

# Load cached data
merged_df, movies_df, ratings_df = load_app_data()
userIds = ratings_df['userId'].unique()
all_movies = set(merged_df['movieId'].values)

# Load models only when needed (cached for performance)
@st.cache_resource
def load_models():
    """Load both models and cache them for performance"""
    with st.spinner("🔄 Loading AI models... (first time only)"):
        svd_data = load_svd_model()
        neural_data = load_neural_model()
    return svd_data, neural_data



def recommend_svd(user_id, n_recommendations=10):
    """Generate SVD recommendations for a given user ID."""
    try:
        svd_data, _ = load_models()
        if not svd_data:
            st.error("SVD model not available")
            return pd.DataFrame()
        
        svd_model = svd_data['svd_model']
        recommendations = get_svd_recommendations(svd_model, user_id, movies_df, ratings_df, n_recommendations)
        
        # Filter by genre if selected
        if selected_genres:
            recommendations = filter_recommendations_by_genre(recommendations, selected_genres)
        
        # Add image URLs and create DataFrame
        rec_data = []
        for rec in recommendations:
            movie_details = get_movie_info_by_id(rec['movie_id'], merged_df)
            if movie_details is not None:
                rec_data.append({
                    'movieId': rec['movie_id'],
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'predicted_rating': rec['predicted_rating'],
                    'img_url': movie_details['img_url']
                })
        
        return pd.DataFrame(rec_data)
        
    except Exception as e:
        st.error(f"Error getting SVD recommendations: {str(e)}")
        return pd.DataFrame()

def recommend_neural(user_id, n_recommendations=10):
    """Generate Neural Network recommendations for a given user ID."""
    try:
        _, neural_data = load_models() # _, is used to ignore the first return value
        if not neural_data:
            st.error("Neural Network model not available")
            return pd.DataFrame()
        
        neural_recommender = neural_data['neural_recommender']
        recommendations = neural_recommender.get_recommendations(user_id, ratings_df, n_recommendations)
        
        # Filter by genre if selected
        if selected_genres:
            recommendations = filter_recommendations_by_genre(recommendations, selected_genres)
        
        # Add image URLs and create DataFrame
        rec_data = []
        for rec in recommendations:
            movie_details = get_movie_info_by_id(rec['movie_id'], merged_df)
            if movie_details is not None:
                rec_data.append({
                    'movieId': rec['movie_id'],
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'predicted_rating': rec['predicted_rating'],
                    'img_url': movie_details['img_url']
                })
        
        return pd.DataFrame(rec_data)
        
    except Exception as e:
        st.error(f"Error getting Neural Network recommendations: {str(e)}")
        return pd.DataFrame()

def recommend_hybrid(user_id, n_recommendations=10):
    """Generate Smart Hybrid recommendations for a given user ID."""
    try:
        hybrid_recs, user_analysis = get_hybrid_recommendations(
            user_id, movies_df, ratings_df, n_recommendations, selected_genres
        )
        
        if not hybrid_recs:
            st.error("No hybrid recommendations available")
            return pd.DataFrame(), {}
        
        # Add image URLs and create DataFrame
        rec_data = []
        for rec in hybrid_recs:
            movie_details = get_movie_info_by_id(rec['movie_id'], merged_df)
            if movie_details is not None:
                rec_data.append({
                    'movieId': rec['movie_id'],
                    'title': rec['title'],
                    'genres': rec['genres'],
                    'predicted_rating': rec['predicted_rating'],
                    'img_url': movie_details['img_url'],
                    'svd_rating': rec.get('svd_rating'),
                    'neural_rating': rec.get('neural_rating'),
                    'confidence': rec.get('confidence', 'medium'),
                    'hybrid_info': rec.get('hybrid_info', {})
                })
        
        return pd.DataFrame(rec_data), user_analysis
        
    except Exception as e:
        st.error(f"Error getting hybrid recommendations: {str(e)}")
        return pd.DataFrame(), {}

def display_movie_with_similarity(movie, index, key_prefix=""):
    """Display a movie recommendation"""
    with st.container():
        movie_col1, movie_col2 = st.columns([1, 3])
        with movie_col1:
            st.image(movie['img_url'], width=120)
        with movie_col2:
            st.write(f"**{movie['title']}**")
            st.write(f"*{movie['genres']}*")
            if 'predicted_rating' in movie:
                st.write(f"⭐ Predicted Rating: {movie['predicted_rating']:.2f}")
            if 'similarity_score' in movie:
                st.write(f"🔗 Similarity Score: {movie['similarity_score']:.3f}")
        
        st.write("---")

def display_hybrid_movie(movie, index):
    """Display a hybrid movie recommendation ranked by hybrid score"""
    with st.container():
        movie_col1, movie_col2, movie_col3 = st.columns([1, 3, 1])
        with movie_col1:
            st.image(movie['img_url'], width=120)
        with movie_col2:
            st.write(f"**{movie['title']}**")
            st.write(f"*{movie['genres']}*")
            
            # Show individual model scores
            if pd.notna(movie.get('svd_rating')) and pd.notna(movie.get('neural_rating')):
                st.write(f"🔢 SVD: {movie['svd_rating']:.2f} | 🧠 Neural: {movie['neural_rating']:.2f}")
            elif pd.notna(movie.get('svd_rating')):
                st.write(f"🔢 SVD: {movie['svd_rating']:.2f} | 🧠 Neural: N/A")
            elif pd.notna(movie.get('neural_rating')):
                st.write(f"🔢 SVD: N/A | 🧠 Neural: {movie['neural_rating']:.2f}")
        
        with movie_col3:
            # Show ranked hybrid score
            st.write(f"**#{index + 1}**")
            st.write(f"📊 **Ranked Hybrid Score:** {movie['predicted_rating']:.2f}")
        
        st.write("---")

####################################################################
# Streamlit UI
####################################################################

# Header
st.title('Smart Hybrid Movie Recommender')
st.write('A Hybrid Collaborative Filtering Recommender System hosted on Streamlit')
st.write('**Main Mechanisms: SVD, Neural Network Model, LLM**')

# Display an image
st.image(
    "https://res.cloudinary.com/practicaldev/image/fetch/s--hGvhAGUu--/c_imagga_scale,f_auto,fl_progressive,"
    "h_500,q_auto,w_1000/https://dev-to-uploads.s3.amazonaws.com/i/mih10uhu1464fx1kr0by.jpg",
    use_container_width=True
)

# Introduction and How-to-Use section
with st.expander("🎬 Welcome! How to Use This Movie Recommender", expanded=False):
    st.write("""
    ## 🚀 Welcome to Your Smart Movie Recommender!
    
    ### 🎯 **Get Recommendations**
    1. **Enter User ID** (1-610) - Different preference profiles
    2. **Choose AI Model** - Smart Hybrid (recommended), SVD, Neural Network, or Compare
    3. **Filter by Genre** (optional) - Select specific genres
    4. **Set Count** (1-100) - Number of recommendations
    5. **Get Results** - See personalized suggestions with AI reasoning
    
    ### 🔍 **Find Similar Movies**
    1. **Select Genres** - Choose movie genres you enjoy
    2. **Set Count** (1-50) - Number of similar movies
    3. **Discover Movies** - Uses cosine similarity for genre-based discovery
    
    ### 🤖 **LLM Recommendations** 
    1. **Describe Your Taste** - Tell us your movie preferences in natural language
    2. **Get AI Results** - GPT analyzes your preferences and recommends perfect movies
    3. **Smart Reasoning** - See why each movie matches your specific taste
    
    ### 💡 **Tips:**
    - Try Smart Hybrid for best results
    - Use LLM for natural language movie requests
    - Mix genres for interesting discoveries
    - Different User IDs = different tastes
    
    **Ready to discover movies? Start exploring! 🍿**
    """)

# Main User Selection Section
st.header("👤 User Profile Setup")
st.write("Select a user profile to personalize your movie experience across all recommendation methods.")

# User input - now in main section
userId = st.number_input(
    "Enter a User ID", 
    min_value=1, 
    max_value=int(userIds.max()), 
    value=1, 
    step=1,
    help=f"Please enter a User ID between 1 and {int(userIds.max())}. Each user has different movie preferences and rating history."
)

# Show user's rating summary - now in main section
if userId:
    user_ratings = ratings_df[ratings_df['userId'] == userId]
    if not user_ratings.empty:
        st.subheader(f"📊 User {userId} Profile Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Ratings", len(user_ratings))
        with col2:
            st.metric("Average Rating", f"{user_ratings['rating'].mean():.1f}")
        with col3:
            fav_movies = len(user_ratings[user_ratings['rating'] >= 4.0])
            st.metric("Favorites (4+ stars)", fav_movies)
        with col4:
            # Get user's most common genre
            user_movie_ids = user_ratings['movieId'].tolist()
            user_movies = merged_df[merged_df['movieId'].isin(user_movie_ids)]
            if not user_movies.empty:
                all_genres = []
                for genres_str in user_movies['genres'].dropna():
                    all_genres.extend(genres_str.split('|'))
                if all_genres:
                    from collections import Counter
                    most_common_genre = Counter(all_genres).most_common(1)[0][0]
                    st.metric("Top Genre", most_common_genre)
                else:
                    st.metric("Top Genre", "N/A")
            else:
                st.metric("Top Genre", "N/A")
    else:
        st.warning(f"User {userId} has no rating history.")

st.markdown("---")
st.write("**Now choose your recommendation method below:**")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "🔍 Find Similar Movies", "🤖 LLM Recommendations"])

with tab1:
    st.header("Movie Recommendations")
    st.write(f"🎯 Getting recommendations for **User {userId}**")

    # Model selection
    st.subheader('Choose Recommendation Model')
    model_choice = st.radio(
        "Select Model Type:",
        ["Smart Hybrid (Recommended)", "SVD (Matrix Factorization)", "Neural Network"],
        help="Smart Hybrid automatically chooses the best approach based on your profile. SVD is faster and more interpretable. Neural Network can capture complex patterns."
    )

    # Genre filtering
    st.subheader('🎬 Filter by Genre (Optional)')
    available_genres = get_unique_genres(merged_df)
    selected_genres = st.multiselect(
        "Select genres to filter recommendations:",
        available_genres,
        help="Leave empty to get recommendations from all genres"
    )



    # Number of recommendations
    n_recs = st.number_input(
        "Number of recommendations:", 
        min_value=1, 
        max_value=100, 
        value=10, 
        step=1,
        help="Maximum 100 recommendations (limited by available unrated movies for each user)"
    )

    # Recommendation button
    if st.button('Show Recommendations'):
        
        if model_choice == "Smart Hybrid (Recommended)":
            # Show hybrid recommendations with user analysis
            st.subheader('🤖 Smart Hybrid Recommendations')
            recommended_movies, user_analysis = recommend_hybrid(userId, n_recs)
            
            if user_analysis:
                # Display user analysis
                with st.expander("🔍 User Analysis & Model Selection Reasoning"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("User Type", user_analysis['user_type'].replace('_', ' ').title())
                        st.metric("Total Ratings", user_analysis['total_ratings'])
                    
                    with col2:
                        st.metric("Primary Model", user_analysis['recommended_model'].upper())
                        st.metric("Confidence", f"{user_analysis['confidence']:.1%}")
                    
                    with col3:
                        st.metric("Genre Diversity", user_analysis['genre_diversity'])
                        st.metric("Avg Rating", f"{user_analysis['avg_rating']:.1f}")
                    
                    # Explain the reasoning
                    st.write("**🧠 AI Reasoning:**")
                    if user_analysis['user_type'] == 'new_user':
                        st.info("New user detected. Using SVD as primary model for better cold-start performance.")
                    elif user_analysis['user_type'] == 'casual_user':
                        if user_analysis['genre_diversity'] > 5:
                            st.info("Casual user with diverse preferences. Neural Network can better capture complex patterns.")
                        else:
                            st.info("Casual user with focused preferences. SVD provides reliable recommendations.")
                    else:  # active_user
                        if user_analysis['rating_variance'] > 1.5 and user_analysis['genre_diversity'] > 8:
                            st.info("Active user with complex, diverse preferences. Neural Network excels at capturing intricate patterns.")
                        else:
                            st.info("Active user with consistent preferences. SVD provides excellent performance.")
            
            if not recommended_movies.empty:
                # Display recommended movies
                for index, movie in recommended_movies.iterrows():
                    display_hybrid_movie(movie, index)
            else:
                st.write("No hybrid recommendations found for this user.")
        
        else:
            # Show single model recommendations
            if "SVD" in model_choice:
                st.subheader('🔢 SVD Recommendations')
                recommended_movies = recommend_svd(userId, n_recs)
                display_func = display_movie_with_similarity
            else:  # Neural Network
                st.subheader('🧠 Neural Network Recommendations')
                recommended_movies = recommend_neural(userId, n_recs)
                display_func = display_movie_with_similarity
            
            if not recommended_movies.empty:
                # Display recommended movies
                for index, movie in recommended_movies.iterrows():
                    display_func(movie, index, key_prefix="")
            else:
                st.write("No movie recommendations found for this user.")

    # Add usage info
    with st.expander("ℹ️ About Recommendation Models"):
        st.write("""
        **🤖 How Smart Hybrid works:**
        - Analyzes your rating patterns to choose the best model
        - Combines SVD and Neural Network predictions intelligently  
        - Adapts to different user types (new, casual, active users)
        - Provides reasoning for model selection and confidence scores
        
        **🔢 How SVD (Matrix Factorization) works:**
        - Uses collaborative filtering based on user-item rating patterns
        - Fast and interpretable recommendations
        - Excellent for users with consistent preferences
        - Better for sparse data (users with few ratings) than true cold-start scenarios
        
        **🧠 How Neural Network works:**
        - Deep learning model that captures complex user-movie interactions
        - Better at discovering non-linear patterns in preferences
        - Excels with users who have diverse or complex tastes
        - Can find unexpected but relevant recommendations
        
        **💡 Tips for better recommendations:**
        - Smart Hybrid is recommended for most users
        - Try different genre filters to explore new categories
        - More ratings in your history = better recommendations
        - Different user IDs have completely different preference profiles
        
        **🎯 Model Selection Guide:**
        - **New users (few ratings)**: SVD works best
        - **Diverse preferences**: Neural Network captures complexity  
        - **Consistent preferences**: SVD provides reliable results
        - **Unsure**: Use Smart Hybrid for automatic selection
        """)

with tab2:
    st.header("Find Similar Movies")
    st.write("Filter by genres to discover movies with similar genre combinations")
    
    # Genre filter selection
    st.subheader("🎬 Filter by Genres")
    
    # Get available genres
    available_genres = get_unique_genres(merged_df)
    # Remove empty/invalid genres
    available_genres = [g for g in available_genres if g and g != "(no genres listed)"]
    
    selected_filter_genres = st.multiselect(
        "Select genres to find similar movies:",
        options=available_genres,
        default=["Adventure", "Animation"],  # Default selection for better UX
        help="Choose one or more genres to find movies with similar genre combinations"
    )
    
    if selected_filter_genres:
        # Number of similar movies to show
        n_similar = st.number_input(
            "Number of similar movies to show:", 
            min_value=1, 
            max_value=50, 
            value=10, 
            step=1,
            help="Maximum 50 similar movies"
        )
        
        # Find and display similar movies
        if st.button("Find Similar Movies", key="find_similar_btn"):
            with st.spinner("Finding movies with similar genres..."):
                similar_movies = find_movies_by_genre_similarity(selected_filter_genres, merged_df, n_similar=n_similar)
            
            if similar_movies:
                st.subheader(f"🔍 Movies Similar to: {' + '.join(selected_filter_genres)}")
                st.write(f"Found movies with varying genre overlap and similarity scores")
                
                for i, sim_movie in enumerate(similar_movies, 1):
                    with st.container():
                        sim_col1, sim_col2, sim_col3 = st.columns([1, 3, 1])
                        
                        with sim_col1:
                            st.image(sim_movie['img_url'], width=100)
                        
                        with sim_col2:
                            st.write(f"**{i}. {sim_movie['title']}**")
                            st.write(f"**Genres:** {sim_movie['genres']}")
                            
                            # Show genre overlap
                            movie_genres = set(sim_movie['genres'].split('|'))
                            selected_set = set(selected_filter_genres)
                            overlap = movie_genres & selected_set
                            if overlap:
                                st.write(f"**Genre Overlap:** {', '.join(overlap)}")
                        
                        with sim_col3:
                            # Show similarity as a progress bar and metric
                            similarity_pct = sim_movie['similarity_score'] * 100
                            st.metric("Similarity", f"{sim_movie['similarity_score']:.3f}")
                            st.progress(sim_movie['similarity_score'])
                        
                        st.write("---")
            else:
                st.info("No similar movies found for the selected genre combination. Try different genres or fewer filters.")
    else:
        st.info("👆 Please select at least one genre to find similar movies")

    # Add usage info
    with st.expander("ℹ️ About Genre-Based Movie Discovery"):
        st.write("""
        **🔍 How Genre Similarity works:**
        - Uses cosine similarity to find movies with similar genre combinations
        - Calculates overlap between your selected genres and movie genres
        - Ranks movies by similarity score (0.0 to 1.0)
        - Higher scores mean better genre match
        
        **📊 Understanding Similarity Scores:**
        - **0.8-1.0**: Very similar genre combination
        - **0.6-0.8**: Good genre overlap with some differences
        - **0.4-0.6**: Moderate similarity, some shared genres
        - **0.2-0.4**: Low similarity, minimal genre overlap
        
        **💡 Tips for better discovery:**
        - Start with 2-3 genres you enjoy
        - Try combining different genre types (e.g., Action + Comedy)
        - Experiment with unusual combinations for unique discoveries
        - Use fewer genres for broader results, more for specific matches
        
        **🎭 Popular Genre Combinations:**
        - **Action + Adventure**: Classic blockbusters
        - **Drama + Romance**: Emotional storytelling
        - **Sci-Fi + Thriller**: Mind-bending experiences
        - **Comedy + Romance**: Light-hearted entertainment
        - **Horror + Thriller**: Intense suspense
        
        **Example:** Selecting "Action + Sci-Fi" will find movies like The Matrix, Terminator, etc.
        """)

with tab3:
    st.header("🤖 LLM Movie Recommendations")
    st.write("Get intelligent movie recommendations powered by OpenAI's GPT models")
    st.write(f"🎯 Analyzing preferences for **User {userId}**")
    
    # Lazy load LLM functionality only when tab is accessed
    if not load_llm_functions():
        st.error("LLM functionality not available. Please install OpenAI: `pip install openai`")
    else:
        # Setup LLM recommender
        llm_recommender = setup_llm_recommender()
        
        if llm_recommender:
            # Text input for additional preferences
            st.subheader("🎭 Describe Your Additional Preferences")
            
            user_preferences = st.text_area(
                "Tell us more about your movie taste and preferences:",
                placeholder="e.g., 'I'm really into psychological thrillers lately, prefer movies from the 2000s onwards, love strong character development, but I'm getting tired of superhero movies. Looking for something thought-provoking.'",
                help="Add any specific preferences, current mood, or things you want to explore beyond your rating history",
                height=120
            )
            
            if st.button("🎯 Get My Personalized Recommendations", key="llm_simple_recs"):
                if userId and (user_preferences or len(ratings_df[ratings_df['userId'] == userId]) > 0):
                    with st.spinner("🧠 LLM is analyzing your rating history and preferences..."):
                        
                        # Get LLM recommendations based on user ID and preferences
                        recommendations = llm_recommender.get_personalized_recommendations(
                            userId, user_preferences, ratings_df, movies_df, merged_df
                        )
                    
                    if recommendations:
                        st.subheader(f"🎬 Personalized Recommendations for User {userId}")
                        
                        # Show what the recommendations are based on
                        if user_preferences:
                            st.info(f"**Based on:** Rating history + Your preferences: '{user_preferences[:80]}{'...' if len(user_preferences) > 80 else ''}'")
                        else:
                            st.info(f"**Based on:** User {userId}'s rating history and movie patterns")
                        
                                                # Display recommendations
                        for i, rec in enumerate(recommendations, 1):
                            try:
                                movie_id = rec.get('movieId')
                                title = rec.get('title', 'Unknown Title')
                                genres = rec.get('genres', 'Unknown Genres')
                                reason = rec.get('reason', 'No reason provided')
                                
                                # Get movie image if available
                                movie_details = get_movie_info_by_id(movie_id, merged_df)
                                img_url = movie_details['img_url'] if movie_details is not None else None
                                
                                with st.container():
                                    col1, col2 = st.columns([1, 3])
                                    
                                    with col1:
                                        if img_url:
                                            st.image(img_url, width=120)
                                        else:
                                            st.write("🎬")
                                    
                                    with col2:
                                        st.write(f"**{i}. {title}**")
                                        st.write(f"*{genres}*")
                                        st.write(f"🎭 **Why this matches your taste:** {reason}")
                                    
                                    st.write("---")
                            except Exception as e:
                                st.error(f"Error displaying recommendation {i}: {e}")
                    else:
                        st.warning("No recommendations received. Try providing more detailed preferences.")
                else:
                    st.warning("Please select a valid User ID. You can optionally add text preferences for even better results.")
            
                        # Add usage info
            with st.expander("ℹ️ About LLM Recommendations"):
                st.write("""
                **🤖 How it works:**
                - Uses OpenAI's GPT models to understand your movie preferences
                - Analyzes your text description of your taste and preferences
                - Matches your preferences with our movie database
                - Provides personalized reasoning for each recommendation
                - The LLM decides the optimal number of recommendations for you
                
                **💡 Tips for better recommendations:**
                - Be specific about genres, themes, and movies you love/hate
                - Mention favorite actors, directors, or time periods
                - Include what mood you're looking for
                - Describe specific elements you enjoy (e.g., "strong female leads", "complex plots")
                
                **Example:** "I love sci-fi thrillers like Inception and The Matrix, enjoy movies with plot twists, prefer films from 2000s onwards, but I'm not into horror or musicals."
                """)

# Add some footer text
st.markdown("---")
st.write("Created by Darren Darius Tan")
st.write("*Powered by Surprise Library, TensorFlow, Scikit-learn & OpenAI*")