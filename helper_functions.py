# Helper Functions for Movie Recommender - Deployment Only

import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Handle TensorFlow imports gracefully
try:
    import tensorflow as tf
    from tensorflow import keras
    # Suppress additional TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Neural Network functionality will be limited.")
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

# Handle Surprise imports gracefully
try:
    from surprise import SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-surprise not available. SVD functionality will be limited.")
    SURPRISE_AVAILABLE = False
    SVD = None

class NeuralRecommender:
    """Neural Network based recommender system"""
    
    def __init__(self, model, user_encoder, movie_encoder, movies_df):
        self.model = model
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.movies_df = movies_df
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
            movie_encoded = self.movie_encoder.transform([movie_id])[0]
            
            prediction_proba = self.model.predict([np.array([user_encoded]), np.array([movie_encoded])], verbose=0)[0]
            # Calculate expected rating (weighted average)
            expected_rating = np.sum(prediction_proba * np.arange(1, 6))
            return expected_rating
        except ValueError:
            return 3.0  # Default rating for unknown users/movies
    
    def get_recommendations(self, user_id, ratings_df, n_recommendations=10):
        """Get movie recommendations for a user (optimized with batch predictions)"""
        # Get movies the user hasn't rated
        user_movies = set(ratings_df[ratings_df.userId == user_id].movieId)
        all_movies = set(self.movies_df.movieId)
        unrated_movies = list(all_movies - user_movies)
        
        # Limit to a reasonable number for speed (sample 1000 movies)
        if len(unrated_movies) > 1000:
            unrated_movies = np.random.choice(unrated_movies, 1000, replace=False).tolist()
        
        # Batch prediction for efficiency
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
            movies_encoded = []
            valid_movies = []
            
            for movie_id in unrated_movies:
                try:
                    movie_encoded = self.movie_encoder.transform([movie_id])[0]
                    movies_encoded.append(movie_encoded)
                    valid_movies.append(movie_id)
                except ValueError:
                    continue
            
            if not movies_encoded:
                return []
            
            # Batch prediction
            user_array = np.array([user_encoded] * len(movies_encoded))
            movie_array = np.array(movies_encoded)
            
            predictions_proba = self.model.predict([user_array, movie_array], verbose=0)
            predictions = np.sum(predictions_proba * np.arange(1, 6), axis=1)
            
            # Combine with movie IDs and sort
            movie_predictions = list(zip(valid_movies, predictions))
            movie_predictions.sort(key=lambda x: x[1], reverse=True)
            
        except ValueError:
            # Fallback to individual predictions
            movie_predictions = []
            for movie_id in unrated_movies[:100]:  # Limit for fallback
                try:
                    pred_rating = self.predict_rating(user_id, movie_id)
                    movie_predictions.append((movie_id, pred_rating))
                except:
                    continue
            
            movie_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations with movie details
        recommendations = []
        for movie_id, predicted_rating in movie_predictions[:n_recommendations]:
            movie_info = self.movies_df[self.movies_df.movieId == movie_id]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_data.title,
                    'genres': movie_data.genres if pd.notna(movie_data.genres) else 'Unknown',
                    'predicted_rating': predicted_rating
                })
        
        return recommendations

def load_svd_model(filename='svd_model.pkl'):
    """Load SVD model with metadata"""
    if not SURPRISE_AVAILABLE:
        print("scikit-surprise not available. Cannot load SVD model.")
        return None
        
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"SVD model loaded from {filename}")
        return model_data
    except FileNotFoundError:
        print(f"SVD model file {filename} not found!")
        return None
    except Exception as e:
        print(f"Error loading SVD model: {e}")
        return None

def load_neural_model(pkl_filename='neural_model.pkl', keras_filename='neural_model.keras'):
    """Load Neural Network model with metadata"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot load neural network model.")
        return None
        
    try:
        # Import the class in the current namespace to fix pickle issues
        import sys
        current_module = sys.modules[__name__]
        if not hasattr(current_module, 'NeuralRecommender'):
            setattr(current_module, 'NeuralRecommender', NeuralRecommender)
        
        # Also add to __main__ namespace if we're being imported from there
        if '__main__' in sys.modules:
            main_module = sys.modules['__main__']
            if not hasattr(main_module, 'NeuralRecommender'):
                setattr(main_module, 'NeuralRecommender', NeuralRecommender)
        
        # Load the pickle file with metadata
        with open(pkl_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load the keras model separately
        neural_model = keras.models.load_model(keras_filename)
        
        # Reconstruct the NeuralRecommender if needed or if pickle failed
        if 'neural_recommender' not in model_data or model_data['neural_recommender'] is None:
            user_encoder = model_data['user_encoder']
            movie_encoder = model_data['movie_encoder']
            movies_df = model_data['movies_df']
            
            neural_recommender = NeuralRecommender(neural_model, user_encoder, movie_encoder, movies_df)
            model_data['neural_recommender'] = neural_recommender
        else:
            # Update the model in case it was pickled separately
            model_data['neural_recommender'].model = neural_model
        
        print(f"Neural model loaded from {pkl_filename} and {keras_filename}")
        return model_data
    except FileNotFoundError as e:
        print(f"Neural model file not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading Neural model: {e}")
        # Try to reconstruct without pickle if possible
        try:
            neural_model = keras.models.load_model(keras_filename)
            print("Attempting to create minimal neural model data...")
            # Create a minimal model data structure
            return {
                'neural_model': neural_model,
                'neural_recommender': None  # Will be handled in the app
            }
        except:
            print("Could not load neural model at all.")
            return None

def get_svd_recommendations(svd_model, user_id, movies_df, ratings_df, n_recommendations=10):
    """Get recommendations using SVD model"""
    if not SURPRISE_AVAILABLE:
        print("scikit-surprise not available. Cannot get SVD recommendations.")
        return []
        
    try:
        # Get movies the user hasn't rated
        user_movies = set(ratings_df[ratings_df.userId == user_id].movieId)
        all_movies = set(movies_df.movieId)
        unrated_movies = list(all_movies - user_movies)
        
        # Get predictions for unrated movies
        movie_predictions = []
        for movie_id in unrated_movies:
            try:
                pred = svd_model.predict(user_id, movie_id)
                movie_predictions.append((movie_id, pred.est))
            except:
                continue
        
        # Sort by predicted rating
        movie_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations with movie details
        recommendations = []
        for movie_id, predicted_rating in movie_predictions[:n_recommendations]:
            movie_info = movies_df[movies_df.movieId == movie_id]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_data.title,
                    'genres': movie_data.genres if pd.notna(movie_data.genres) else 'Unknown',
                    'predicted_rating': predicted_rating
                })
        
        return recommendations
    except Exception as e:
        print(f"Error getting SVD recommendations: {e}")
        return []



def get_unified_recommendations(user_id, model_type='svd', n_recommendations=10):
    """Get recommendations from either SVD or Neural Network model"""
    try:
        # Load data
        ratings_df = pd.read_csv('dataset/ratings.csv')
        movies_df = pd.read_csv('dataset/movies.csv')
        
        if model_type.lower() == 'svd':
            svd_data = load_svd_model()
            if svd_data:
                svd_model = svd_data['svd_model']
                return get_svd_recommendations(svd_model, user_id, movies_df, ratings_df, n_recommendations)
        
        elif model_type.lower() == 'neural':
            neural_data = load_neural_model()
            if neural_data:
                neural_recommender = neural_data['neural_recommender']
                return neural_recommender.get_recommendations(user_id, ratings_df, n_recommendations)
        
        return []
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

def compare_model_recommendations(user_id, n_recommendations=5):
    """Compare recommendations from both models side by side"""
    svd_recs = get_unified_recommendations(user_id, 'svd', n_recommendations)
    neural_recs = get_unified_recommendations(user_id, 'neural', n_recommendations)
    
    return {
        'svd_recommendations': svd_recs,
        'neural_recommendations': neural_recs
    }

def format_recommendations_for_display(recommendations):
    """Format recommendations for Streamlit display (maintains compatibility with current app.py)"""
    formatted = []
    for rec in recommendations:
        formatted.append({
            'movieId': rec['movie_id'],
            'title': rec['title'], 
            'genres': rec['genres'],
            'predicted_rating': rec.get('predicted_rating', 0.0)
        })
    return formatted 

def get_unique_genres(merged_df):
    """Extract all unique genres from the dataset"""
    all_genres = set()
    for genres_str in merged_df['genres'].dropna():
        if isinstance(genres_str, str):
            genres = genres_str.split('|')
            all_genres.update(genres)
    return sorted(list(all_genres))

def filter_recommendations_by_genre(recommendations, selected_genres):
    """Filter recommendations based on selected genres"""
    if not selected_genres:
        return recommendations
    
    filtered_recs = []
    for rec in recommendations:
        movie_genres = rec.get('genres', '').split('|')
        if any(genre in selected_genres for genre in movie_genres):
            filtered_recs.append(rec)
    
    return filtered_recs

def create_movie_similarity_matrix(merged_df):
    """Create cosine similarity matrix based on movie genres and features"""
    # Prepare genre features
    genre_matrix = []
    movie_ids = []
    
    for _, movie in merged_df.iterrows():
        movie_ids.append(movie['movieId'])
        # Create genre vector
        genres = movie['genres'].split('|') if pd.notna(movie['genres']) else []
        genre_vector = '|'.join(genres)
        genre_matrix.append(genre_vector)
    
    # Create TF-IDF vectors from genres
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+', lowercase=False)
    genre_tfidf = tfidf.fit_transform(genre_matrix)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(genre_tfidf)
    
    return similarity_matrix, movie_ids

def get_cached_similarity_data(merged_df):
    """Cache the similarity matrix computation for performance"""
    return create_movie_similarity_matrix(merged_df)

def find_similar_movies(movie_id, merged_df, n_similar=5):
    """Find movies similar to a given movie using cosine similarity"""
    try:
        # Get cached similarity data
        similarity_matrix, movie_ids = get_cached_similarity_data(merged_df)
        
        # Find the index of the movie
        if movie_id not in movie_ids:
            return []
        
        movie_idx = movie_ids.index(movie_id)
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        
        # Sort by similarity score (excluding the movie itself)
        sim_scores = [(i, score) for i, score in sim_scores if i != movie_idx and score > 0]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar movies
        similar_movies = []
        for i, score in sim_scores[:n_similar]:
            similar_movie_id = movie_ids[i]
            movie_info = merged_df[merged_df['movieId'] == similar_movie_id].iloc[0]
            similar_movies.append({
                'movie_id': similar_movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'similarity_score': score,
                'img_url': movie_info['img_url']
            })
        
        return similar_movies
        
    except Exception as e:
        print(f"Error finding similar movies: {e}")
        return []

def get_movie_info_by_id(movie_id, merged_df):
    """Get detailed movie information by movie ID"""
    movie_info = merged_df[merged_df['movieId'] == movie_id]
    if not movie_info.empty:
        return movie_info.iloc[0].to_dict()
    return None

def analyze_user_characteristics(user_id, ratings_df, movies_df):
    """Analyze user characteristics to determine best recommendation approach"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    if user_ratings.empty:
        return {
            'total_ratings': 0,
            'avg_rating': 0,
            'rating_variance': 0,
            'genre_diversity': 0,
            'user_type': 'new_user',
            'recommended_model': 'svd',
            'confidence': 0.5
        }
    
    # Basic statistics
    total_ratings = len(user_ratings)
    avg_rating = user_ratings['rating'].mean()
    rating_variance = user_ratings['rating'].var()
    
    # Calculate genre diversity
    user_movie_ids = user_ratings['movieId'].tolist()
    user_movies = movies_df[movies_df['movieId'].isin(user_movie_ids)]
    
    all_genres = set()
    for _, movie in user_movies.iterrows():
        if pd.notna(movie['genres']):
            genres = movie['genres'].split('|')
            all_genres.update(genres)
    
    genre_diversity = len(all_genres)
    
    # Determine user type and best model
    if total_ratings < 10:
        user_type = 'new_user'
        recommended_model = 'svd'  # SVD works better for cold start
        confidence = 0.7
    elif total_ratings < 50:
        user_type = 'casual_user'
        if genre_diversity > 5:
            recommended_model = 'neural'  # Neural better for diverse preferences
            confidence = 0.6
        else:
            recommended_model = 'svd'
            confidence = 0.6
    else:
        user_type = 'active_user'
        if rating_variance > 1.5 and genre_diversity > 8:
            recommended_model = 'neural'  # Neural better for complex patterns
            confidence = 0.8
        else:
            recommended_model = 'svd'
            confidence = 0.7
    
    return {
        'total_ratings': total_ratings,
        'avg_rating': avg_rating,
        'rating_variance': rating_variance,
        'genre_diversity': genre_diversity,
        'user_type': user_type,
        'recommended_model': recommended_model,
        'confidence': confidence
    }

def get_hybrid_recommendations(user_id, movies_df, ratings_df, n_recommendations=10, selected_genres=None):
    """Get hybrid recommendations using intelligent model selection and ensemble methods"""
    try:
        # Analyze user characteristics
        user_analysis = analyze_user_characteristics(user_id, ratings_df, movies_df)
        
        # Load both models
        svd_data = load_svd_model()
        neural_data = load_neural_model()
        
        if not svd_data or not neural_data:
            # Fallback to available model
            if svd_data:
                return get_svd_recommendations(svd_data['svd_model'], user_id, movies_df, ratings_df, n_recommendations)
            elif neural_data:
                return neural_data['neural_recommender'].get_recommendations(user_id, ratings_df, n_recommendations)
            else:
                return []
        
        # Get recommendations from both models
        svd_model = svd_data['svd_model']
        neural_recommender = neural_data['neural_recommender']
        
        svd_recs = get_svd_recommendations(svd_model, user_id, movies_df, ratings_df, n_recommendations * 2)
        neural_recs = neural_recommender.get_recommendations(user_id, ratings_df, n_recommendations * 2)
        
        # Apply genre filtering if specified
        if selected_genres:
            svd_recs = filter_recommendations_by_genre(svd_recs, selected_genres)
            neural_recs = filter_recommendations_by_genre(neural_recs, selected_genres)
        
        # Determine ensemble weights based on user analysis
        if user_analysis['recommended_model'] == 'svd':
            svd_weight = user_analysis['confidence']
            neural_weight = 1 - user_analysis['confidence']
        else:
            neural_weight = user_analysis['confidence']
            svd_weight = 1 - user_analysis['confidence']
        
        # Create ensemble recommendations
        hybrid_recs = create_ensemble_recommendations(
            svd_recs, neural_recs, svd_weight, neural_weight, n_recommendations
        )
        
        # Add hybrid metadata
        for rec in hybrid_recs:
            rec['hybrid_info'] = {
                'primary_model': user_analysis['recommended_model'],
                'svd_weight': svd_weight,
                'neural_weight': neural_weight,
                'user_type': user_analysis['user_type'],
                'confidence': user_analysis['confidence']
            }
        
        return hybrid_recs, user_analysis
        
    except Exception as e:
        print(f"Error in hybrid recommendations: {e}")
        return [], {}

def create_ensemble_recommendations(svd_recs, neural_recs, svd_weight, neural_weight, n_recommendations):
    """Create ensemble recommendations by combining and ranking from both models"""
    
    # Create dictionaries for easy lookup
    svd_dict = {rec['movie_id']: rec for rec in svd_recs}
    neural_dict = {rec['movie_id']: rec for rec in neural_recs}
    
    # Get all unique movie IDs
    all_movie_ids = set(svd_dict.keys()) | set(neural_dict.keys())
    
    ensemble_scores = []
    
    for movie_id in all_movie_ids:
        svd_score = svd_dict.get(movie_id, {}).get('predicted_rating', 0)
        neural_score = neural_dict.get(movie_id, {}).get('predicted_rating', 0)
        
        # Calculate weighted ensemble score
        if movie_id in svd_dict and movie_id in neural_dict:
            # Both models have predictions
            ensemble_score = (svd_score * svd_weight) + (neural_score * neural_weight)
            confidence = 'high'
        elif movie_id in svd_dict:
            # Only SVD has prediction
            ensemble_score = svd_score * svd_weight
            confidence = 'medium'
        else:
            # Only Neural has prediction
            ensemble_score = neural_score * neural_weight
            confidence = 'medium'
        
        # Use movie info from whichever model has it
        movie_info = svd_dict.get(movie_id) or neural_dict.get(movie_id)
        if movie_info:
            ensemble_scores.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': ensemble_score,
                'svd_rating': svd_score if movie_id in svd_dict else None,
                'neural_rating': neural_score if movie_id in neural_dict else None,
                'confidence': confidence
            })
    
    # Sort by ensemble score and return top N
    ensemble_scores.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return ensemble_scores[:n_recommendations]

def find_movies_by_genre_similarity(selected_genres, merged_df, n_similar=10):
    """Find movies similar to selected genres using cosine similarity on TF-IDF vectors"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create a virtual movie with selected genres
        selected_genres_str = '|'.join(sorted(selected_genres))
        
        # Get all movie genres (clean the data)
        genres_list = []
        valid_movies = []
        
        for idx, row in merged_df.iterrows():
            if pd.notna(row['genres']) and row['genres'] != '' and row['genres'] != '(no genres listed)':
                genres_list.append(row['genres'])
                valid_movies.append(row)
        
        # Add our selected genres as the last item
        genres_list.append(selected_genres_str)
        
        if len(genres_list) < 2:
            return []
        
        # Create TF-IDF vectors using genre tokenization
        tfidf = TfidfVectorizer(
            tokenizer=lambda x: x.split('|'), 
            lowercase=False,
            token_pattern=None
        )
        
        # Fit and transform all genre strings into TF-IDF vectors
        tfidf_matrix = tfidf.fit_transform(genres_list)
        
        # Get the vector for our selected genres (last item)
        selected_vector = tfidf_matrix[-1]
        
        # Get vectors for all movies (all except last)
        movie_vectors = tfidf_matrix[:-1]
        
        # Calculate cosine similarity between selected genres and all movies
        similarities = cosine_similarity(selected_vector, movie_vectors).flatten()
        
        # Create results with similarity scores
        similar_movies = []
        for idx, similarity_score in enumerate(similarities):
            if similarity_score > 0:  # Only include movies with some similarity
                movie_data = valid_movies[idx]
                similar_movies.append({
                    'movie_id': movie_data['movieId'],
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity_score': similarity_score,
                    'img_url': movie_data['img_url']
                })
        
        # Sort by cosine similarity score (descending)
        similar_movies.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_movies[:n_similar]
        
    except Exception as e:
        print(f"Error in cosine similarity calculation: {e}")
        return [] 