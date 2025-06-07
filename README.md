# 🎬 Smart Hybrid Movie Recommender App

An intelligent hybrid movie recommendation system that automatically combines **SVD (Singular Value Decomposition)** and **Neural Network** collaborative filtering approaches with advanced **cosine similarity** for personalized movie discovery.

## ✨ Key Features

### 🤖 **Smart Hybrid Recommendation System**
- **Intelligent Model Selection**: Automatically analyzes user behavior to choose optimal recommendation approach
- **User Profiling**: Categorizes users as New, Casual, or Active based on rating patterns
- **Ensemble Learning**: Combines SVD and Neural Network predictions with dynamic weighting
- **Transparent AI**: Shows reasoning behind model selection and confidence scores

### 🎯 **Triple-Tab Interface**
1. **Get Recommendations**: Personalized movie suggestions using SVD, Neural Network, or Smart Hybrid
2. **Find Similar Movies**: Genre-based movie discovery using cosine similarity  
3. **LLM Recommendations**: AI-powered natural language movie recommendations via OpenAI GPT



## 🏗️ Architecture

### **Smart Hybrid System:**
- **New Users (<10 ratings)**: Primary SVD (70%) for cold-start performance
- **Casual Users (10-50 ratings)**: Adaptive based on genre diversity
- **Active Users (50+ ratings)**: Neural Network for complex patterns or SVD for consistency

### **Models Used:**
1. **SVD (Matrix Factorization)**: NDCG@10 of 0.96 - Excellent for consistent preferences
2. **Neural Network (Deep Learning)**: NDCG@10 of 0.76 - Better for complex patterns
3. **Smart Hybrid**: Intelligently combines SVD and Neural Network with dynamic weighting
4. **LLM (OpenAI GPT)**: Natural language understanding for preference-based recommendations
5. **Cosine Similarity**: TF-IDF genre vectors for content-based discovery

### **Tech Stack:**
- **Frontend**: Streamlit with triple-tab interface and centralized user setup
- **ML Libraries**: scikit-surprise, TensorFlow/Keras, scikit-learn  
- **LLM Integration**: OpenAI GPT for natural language recommendations
- **Similarity Engine**: TF-IDF + Cosine Similarity
- **Data Processing**: pandas, numpy
- **Deployment**: Docker containers with optimized multi-stage builds

## 🚀 Quick Start

**Docker (Recommended):**
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key movie-recommender
```

**Local Development:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Access at: `http://localhost:8501`

## 📋 Quick Requirements

- **Docker**: 20.10+ with 4GB+ RAM (recommended)
- **Python**: 3.11+ with 8GB+ RAM (local development)
- **OpenAI API Key**: For LLM recommendations (Tab 3)

## 📁 Project Structure

```
MovieRecommender/
├── app.py                    # Main Streamlit application with tabbed interface
├── helper_functions.py       # Hybrid models, cosine similarity, user analysis
├── llm_recommender.py        # OpenAI-powered LLM recommendations
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container configuration
├── .dockerignore            # Docker build optimization
├── svd_model.pkl            # Trained SVD model
├── neural_model.pkl         # Neural network metadata
├── neural_model.keras       # Trained neural network
├── dataset/
│   ├── movies.csv           # Movie metadata (9,742 movies)
│   ├── ratings.csv          # User ratings data (610 users)
│   └── merged_movies.csv    # Movies with poster URLs
├── output_data/
│   └── merged_movies.csv    # Processed movie data with images
├── Procfile                 # Heroku deployment (legacy)
├── setup.sh                 # Streamlit configuration
└── README.md                # This file
```

## 🎯 How to Use

### **Main User Setup (Required First)**
1. **Enter User ID** (1-610): Each represents a different preference profile
2. **View Profile Summary**: See user's rating history, preferences, and top genre
3. **Choose Your Tab**: Select your preferred recommendation method below

### **Tab 1: Get Recommendations**
1. **Choose AI Model**:
   - **🤖 Smart Hybrid (Recommended)**: AI automatically selects best approach
   - **🔢 SVD (Matrix Factorization)**: Fast, reliable recommendations
   - **🧠 Neural Network**: Advanced pattern recognition for complex preferences
2. **Filter by Genre** (Optional): Select specific genres you're interested in
3. **Set Recommendations Count** (1-100): Number of movie suggestions
4. **View Results**: See ranked recommendations with hybrid scores and model insights

### **Tab 2: Find Similar Movies**
1. **Select Genres**: Choose one or more movie genres you enjoy
2. **Set Results Count** (1-50): Number of similar movies to discover
3. **Find Movies**: Uses cosine similarity on TF-IDF vectors
4. **Explore Results**: See similarity scores and genre overlaps with progress indicators

### **Tab 3: LLM Recommendations**
1. **Describe Your Preferences**: Write what kind of movies you're looking for
2. **Get AI Analysis**: LLM analyzes both your text and user rating history
3. **View Smart Results**: See personalized recommendations with detailed reasoning
4. **Understand Why**: Each suggestion includes explanation of why it matches your taste

## 🔧 Smart Hybrid Weighting Explained

The system analyzes each user and dynamically weights SVD vs Neural Network:

**User Categories & Weights:**
- **New Users** (<10 ratings): SVD 70%, Neural 30% - Better cold-start performance
- **Casual Users** (10-50 ratings): 60/40 split based on genre diversity  
- **Active Users** (50+ ratings): Up to Neural 80%, SVD 20% for complex patterns

**Final Score** = (SVD_score × SVD_weight) + (Neural_score × Neural_weight)





## 🐛 Common Issues

- **Docker Build**: Ensure 4GB+ memory allocated
- **Port Conflicts**: Change `-p 8502:8501` if 8501 is busy  
- **Model Files**: Verify `svd_model.pkl`, `neural_model.pkl`, `neural_model.keras` exist
- **Memory**: Close other apps, system needs significant RAM
- **OpenAI Key**: Set `OPENAI_API_KEY` environment variable for LLM tab



---

🎬 **Built with Intelligence**: Streamlit + TensorFlow + scikit-learn + scikit-surprise  
🤖 **Powered by AI**: Smart Hybrid Recommendations + Cosine Similarity Discovery  
🎯 **User-Centered**: Transparent reasoning + Visual feedback + Intuitive interface
