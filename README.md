# 🎬 Smart Hybrid Movie Recommender App

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-brightgreen?style=for-the-badge)](https://movie-recommender-app-yujk.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)

An intelligent hybrid movie recommendation system that automatically combines **SVD (Singular Value Decomposition)** and **Neural Network** collaborative filtering approaches with advanced **cosine similarity** for personalized movie discovery.

## 🌟 **[Try the Live Demo →](https://movie-recommender-app-yujk.onrender.com/)**

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
- **Deployment**: Render with automatic GitHub integration

## 🚀 Deployment Options

### **🌐 Live Demo (Recommended)**
Visit the deployed app: **[https://movie-recommender-app-yujk.onrender.com/](https://movie-recommender-app-yujk.onrender.com/)**

### **☁️ Deploy Your Own (Render)**
1. Fork this repository
2. Connect to [Render](https://render.com)
3. Create new Web Service from your GitHub repo
4. Set environment variable: `OPENAI_API_KEY=your_openai_key`
5. Deploy automatically!

### **🐳 Docker (Local)**
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key movie-recommender
```

### **💻 Local Development**
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_openai_key  # Linux/Mac
# OR
set OPENAI_API_KEY=your_openai_key     # Windows
streamlit run app.py
```

Access at: `http://localhost:8501`

## 📋 Requirements

- **Live Demo**: Just visit the link! ✨
- **Local Setup**: Python 3.11+ with 4GB+ RAM
- **OpenAI API Key**: For LLM recommendations (Tab 3) - [Get one here](https://platform.openai.com/api-keys)

## 📁 Project Structure

```
MovieRecommender/
├── app.py                    # Main Streamlit application with tabbed interface
├── helper_functions.py       # Hybrid models, cosine similarity, user analysis
├── llm_recommender.py        # OpenAI-powered LLM recommendations
├── requirements.txt          # Python dependencies
├── render.yaml              # Render deployment configuration
├── startup.sh               # Render startup script
├── svd_model.pkl            # Trained SVD model
├── neural_model.pkl         # Neural network metadata
├── neural_model.keras       # Trained neural network
├── dataset/
│   ├── movies.csv           # Movie metadata (9,742 movies)
│   ├── ratings.csv          # User ratings data (610 users)
│   └── merged_movies.csv    # Movies with poster URLs
├── output_data/
│   └── merged_movies.csv    # Processed movie data with images
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

## 🔧 Environment Variables

For deployment or local development with LLM features:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Get your OpenAI API key at: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## 🐛 Troubleshooting

- **Live Demo Issues**: Try refreshing the page or clearing browser cache
- **Local Setup**: Ensure Python 3.11+ and sufficient RAM (4GB+)
- **Docker Build**: Allocate 4GB+ memory to Docker
- **Port Conflicts**: Change to `-p 8502:8501` if port 8501 is busy  
- **Missing Models**: Verify `svd_model.pkl`, `neural_model.pkl`, `neural_model.keras` exist
- **OpenAI Errors**: Check your API key and account credits
- **Memory Issues**: Close other applications to free up RAM

## 🔄 Updates & Maintenance

This app is deployed on Render with automatic updates from the GitHub repository. Any changes pushed to the main branch will automatically redeploy the application.

---

## 🎯 Performance Metrics

- **SVD Model**: NDCG@10 of 0.96 (Excellent)
- **Neural Network**: NDCG@10 of 0.76 (Good)
- **Response Time**: < 2 seconds for most operations
- **Uptime**: 99.9% (Render hosting)

---

🎬 **Built with Intelligence**: Streamlit + TensorFlow + scikit-learn + scikit-surprise  
🤖 **Powered by AI**: Smart Hybrid Recommendations + Cosine Similarity Discovery + OpenAI GPT  
🎯 **User-Centered**: Transparent reasoning + Visual feedback + Intuitive interface  
🚀 **Production Ready**: Live deployment on Render with automatic scaling

**[🌟 Try the Live Demo Now →](https://movie-recommender-app-yujk.onrender.com/)**
