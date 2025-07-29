# Import libraries
import numpy as np
import pandas as pd
import  pickle
import ast
import os
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests

# Download if not already done
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


#Load data
movies = pd.read_csv("./data/final_df.csv")
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)





def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path

def recommend(movie):
    index = movies[movies['original_title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].original_title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender')

# Create two tabs
tab1, tab2 = st.tabs(["🎬 Recommend", "⭐ Top Rated"])





movie_list = movies['original_title'].values
with tab1:
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    # 🖼 Show the poster of the selected movie
    selected_movie_id = movies[movies['original_title'] == selected_movie]['id'].values[0]
    selected_movie_poster = fetch_poster(selected_movie_id)
    st.image(selected_movie_poster, caption=f"Selected: {selected_movie}", width=200)

    if st.button('Show Recommendation'):
        recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])

with tab2:
    st.header("Top 10 Rated Movies")

    # Sort and get top 100
    top_rated = movies.sort_values(by='vote_average', ascending=False).head(10).reset_index(drop=True)

    for _, row in top_rated.iterrows():
        movie_title = row['original_title']
        movie_id = row['id']
        with st.expander(f"🎬 {movie_title}  ⭐ {row['vote_average']}"):
            poster = fetch_poster(movie_id)
            st.image(poster, width=150, caption=movie_title)

            recommended_movie_names, recommended_movie_posters = recommend(movie_title)
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.text(recommended_movie_names[i])
                    st.image(recommended_movie_posters[i])