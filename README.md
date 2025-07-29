# 🎬 Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit that suggests movies based on content similarity using natural language processing techniques.

## ✨ Features

- **Movie Recommendations**: Get 5 similar movie recommendations based on your selected movie
- **Interactive UI**: Clean and intuitive Streamlit interface with movie posters
- **Top Rated Movies**: Browse the top 10 highest-rated movies with recommendations
- **Real-time Poster Fetching**: Automatically fetches movie posters from The Movie Database (TMDb) API
- **Content-Based Filtering**: Uses cosine similarity on movie tags for accurate recommendations

## 🚀 Demo []

The application provides two main features:
1. **Recommend Tab**: Select a movie and get 5 similar recommendations
2. **Top Rated Tab**: Browse top-rated movies with expandable recommendation sections

## 🛠️ Installation

### Setup

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd movie-recommender
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (automatically handled by the app)
```python
# The app automatically downloads these NLTK packages:
# - punkt
# - wordnet
# - omw-1.4
# - averaged_perceptron_tagger
```

4. **Prepare the data**
   - Ensure you have `final_df.csv` in the `./data/` directory
   - The CSV should contain columns: `original_title`, `tags`, `id`, `vote_average`

5. **Run the application**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
movie-recommender/
│
├── app.py                 # Main Streamlit application
├── data/
│   └── final_df.csv      # Movie dataset with processed tags
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```


## 🔧 Dependencies

```txt
numpy
pandas
nltk
scikit-learn
streamlit
requests
```

## 🎯 How It Works

1. **Data Processing**: The system uses pre-processed movie tags containing information about genres, cast, crew, and plot
2. **Vectorization**: Converts text tags into numerical vectors using CountVectorizer (max 5000 features)
3. **Similarity Calculation**: Computes cosine similarity between movie vectors
4. **Recommendation**: Returns top 5 most similar movies based on similarity scores
5. **Poster Fetching**: Retrieves movie posters from TMDb API in real-time


**Happy Movie Watching! 🍿**
