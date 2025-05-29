import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import faiss
from PIL import Image
import asyncio
import aiohttp
import nest_asyncio
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from datetime import datetime

# Apply nest_asyncio to make asyncio work with Streamlit
nest_asyncio.apply()

# Initialize session states
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.shown_movies = set()
    st.session_state.recommendation_history = []
    st.session_state.history_position = -1
    st.session_state.filtered_movies = None
    st.session_state.current_age = None
    st.session_state.current_genres = None
    st.session_state.current_selected_movie = None

# Load data files
if not st.session_state.initialized:
    try:
        # Load the data files in chunks to improve memory usage
        with open('movies_data.pkl', 'rb') as f:
            st.session_state.movies_df = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            st.session_state.tfidf_vectorizer = pickle.load(f)
        
        # Load FAISS index
        st.session_state.tfidf_index = faiss.read_index('tfidf_index.bin')
        
        # Check if 'tags' column exists
        if 'tags' not in st.session_state.movies_df.columns:
            st.error("The 'tags' column is missing from the movie data. This is required for content-based recommendations.")
        
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Fix for the caching issue with async functions
async def is_valid_image_async(url):
    if not url or url == 'placeholder.jpg':
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=5) as response:
                return response.status == 200
    except Exception:
        return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def is_valid_image(url):
    try:
        return asyncio.run(is_valid_image_async(url))
    except Exception:
        return False

# Async function to fetch movie details from OMDB API
async def fetch_movie_details_async(title):
    try:
        api_key = "34e5950a"
        url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("Response") == "True":
                        return {
                            "Title": data.get("Title"),
                            "Year": data.get("Year"),
                            "Runtime": data.get("Runtime"),
                            "Genre": data.get("Genre"),
                            "Plot": data.get("Plot"),
                            "Poster": data.get("Poster"),
                            "imdbRating": data.get("imdbRating"),
                        }
    except Exception:
        pass
    return None

# Fetch multiple movie details concurrently
async def fetch_multiple_movie_details(titles):
    tasks = [fetch_movie_details_async(title) for title in titles]
    return await asyncio.gather(*tasks)

# Wrapper for Streamlit caching
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_multiple_details(titles):
    return asyncio.run(fetch_multiple_movie_details(titles))

# Cache genre list
@st.cache_data
def get_unique_genres(movies_df):
    return sorted(list(movies_df['Genre'].str.split(', ').explode().unique()))

# Streamlit UI
st.header("🎬 Movie Recommender System")

# Recommendation mode selection
recommendation_mode = st.radio(
    "Select recommendation mode:",
    ("Content-Based", "Age and Genre-Based")
)

# Search functionality with caching and clear button
# Replace the search functionality section (around lines 130-140) with this:

# Search functionality with caching and clear button
search_col1, search_col2 = st.columns([4, 1])

# Initialize search query in session state if not exists
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

with search_col1:
    search_query = st.text_input(
        "🔍 Search for a movie by title or keyword", 
        value=st.session_state.search_query,
        key="search_input"
    )
    
with search_col2:
    st.write("")  # Empty space for alignment
    if st.button("🗑️ Clear", help="Clear search"):
        st.session_state.search_query = ""
        st.rerun()

# Update session state when search query changes
if search_query != st.session_state.search_query:
    st.session_state.search_query = search_query



def display_movie_info(row, movie_details, context="default", idx=None):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        poster_url = movie_details.get("Poster") if movie_details else None
        if poster_url and is_valid_image(poster_url):
            st.image(poster_url, caption=f"{row['Title']} Poster")
        else:
            st.image('placeholder.jpg', caption="No poster available")
    
    with col2:
        if movie_details:
            st.write(f"**{movie_details['Title']}** ({movie_details['Year']})")
            st.write(f"Runtime: {movie_details['Runtime']}")
            st.write(f"Genre: {movie_details['Genre']}")
            st.write(f"IMDb Rating: {movie_details['imdbRating']}")
            st.write(f"Plot: {movie_details['Plot']}")

@st.cache_data
def search_movies(query, movies_df):
    return movies_df[movies_df['Title'].str.contains(query, case=False, na=False)]

if search_query:
    search_results = search_movies(search_query, st.session_state.movies_df)
    if not search_results.empty:
        st.write("Search Results:")
        all_titles = search_results['Title'].tolist()
        all_details = fetch_multiple_details(all_titles)
        details_dict = {details['Title']: details for details in all_details if details}
        for idx, (_, row) in enumerate(search_results.iterrows()):
            title = row['Title']
            movie_details = details_dict.get(title)
            with st.expander(f"{title} (Genre: {row['Genre']}, Certificate: {row['Certificate']})"):
                display_movie_info(row, movie_details, context="search", idx=idx)
    else:
        st.warning("No movies found matching your search.")

# Content-Based Recommendation
if recommendation_mode == "Content-Based":
    movies_list = st.session_state.movies_df['Title'].values
    default_movie = "The Avengers" if "The Avengers" in movies_list else movies_list[0]
    default_index = int(np.where(movies_list == default_movie)[0][0]) if default_movie in movies_list else 0
    selectvalue = st.selectbox(
        "Select a movie",
        movies_list,
        index=default_index
    )
else:  # Age and Genre-Based
    age = st.selectbox("Select your age", list(range(1, 101)))
    genres = get_unique_genres(st.session_state.movies_df)
    selected_genres = st.multiselect("Select genres", genres, default=["Action", "Adventure"])

def update_filtered_movies(age, genres):
    """Update filtered movies only if parameters changed"""
    if (st.session_state.current_age != age or 
        st.session_state.current_genres != tuple(genres)):
        
        # Filter movies by age
        if age < 13:
            filtered_df = st.session_state.movies_df[
                st.session_state.movies_df['Certificate'].isin(['G', 'PG'])
            ]
        elif 13 <= age < 18:
            filtered_df = st.session_state.movies_df[
                st.session_state.movies_df['Certificate'].isin(['G', 'PG', 'PG-13'])
            ]
        else:
            filtered_df = st.session_state.movies_df.copy()

        # Filter by genres
        if genres:
            filtered_df = filtered_df[
                filtered_df['Genre'].apply(lambda x: any(genre in x for genre in genres))
            ]

        st.session_state.filtered_movies = filtered_df
        st.session_state.current_age = age
        st.session_state.current_genres = tuple(genres)

def get_recommendation_key():
    """Generate a unique key for the current recommendation parameters"""
    if recommendation_mode == "Content-Based":
        return f"content_{selectvalue}"
    else:
        return f"age_genre_{age}_{'-'.join(selected_genres) if selected_genres else 'none'}"

def recommend_content_based(movie, exclude_shown=True):
    try:
        movie_row = st.session_state.movies_df[st.session_state.movies_df['Title'] == movie]
        if movie_row.empty:
            return [], [], True
        
        # Get the index of the selected movie
        movie_index = movie_row.index[0]
        
        # Get the TF-IDF vector for the selected movie - safely check tags field
        movie_tags = movie_row.iloc[0].get('tags', '')
        if not movie_tags:
            return [], [], True
            
        movie_tfidf = st.session_state.tfidf_vectorizer.transform([movie_tags])
        movie_tfidf_dense = np.asarray(movie_tfidf.todense(), dtype=np.float32)
        
        # Search similar movies using FAISS
        k = 50  # Get more results than we need to account for filtering
        distances, indices = st.session_state.tfidf_index.search(movie_tfidf_dense, k)
        
        # Remove the selected movie itself and optionally filter out already shown movies
        recommend_indices = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(st.session_state.movies_df) and idx != movie_index:
                title = st.session_state.movies_df.iloc[idx].Title
                if not exclude_shown or title not in st.session_state.shown_movies:
                    recommend_indices.append(idx)
                    if len(recommend_indices) >= 5:
                        break
        
        if not recommend_indices:
            return [], [], True
        
        recommend_movie = []
        recommend_poster = []
        for idx in recommend_indices:
            title = st.session_state.movies_df.iloc[idx].Title
            poster = st.session_state.movies_df.iloc[idx].Poster
            recommend_movie.append(title)
            recommend_poster.append(poster)
            if exclude_shown:
                st.session_state.shown_movies.add(title)
        
        return recommend_movie, recommend_poster, False
    except Exception as e:
        st.error(f"Error in recommendation logic: {e}")
        return [], [], True

def recommend_age_genre_based(age, genres, exclude_shown=True):
    try:
        update_filtered_movies(age, genres)
        
        # Remove already shown movies if exclude_shown is True
        if exclude_shown:
            available_movies = st.session_state.filtered_movies[
                ~st.session_state.filtered_movies['Title'].isin(st.session_state.shown_movies)
            ]
        else:
            available_movies = st.session_state.filtered_movies.copy()

        if available_movies.empty:
            return [], [], True

        # Get recommendations
        recommend_movies = available_movies.sample(n=min(5, len(available_movies)))
        
        recommend_movie = recommend_movies['Title'].tolist()
        recommend_poster = []
        for _, movie in recommend_movies.iterrows():
            poster = movie.Poster
            recommend_poster.append(poster)
            if exclude_shown:
                st.session_state.shown_movies.add(movie['Title'])
            
        return recommend_movie, recommend_poster, False
    except Exception as e:
        st.error(f"Error in recommendation logic: {e}")
        return [], [], True

# Enhanced Navigation UI components
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    prev_button = st.button("⬅️ Previous", help="View previous recommendation set")

with col2:
    show_button = st.button("🎯 Get Recommendations", key="show_rec", help="Get initial recommendations")

with col3:
    new_recs_button = st.button("🔄 New Recommendations", help="Get fresh recommendations")

with col4:
    next_button = st.button("➡️ Next", help="View next recommendation set")

# History position indicator and reset button
if st.session_state.recommendation_history:
    col_indicator, col_reset = st.columns([3, 1])
    with col_indicator:
        st.caption(f"📍 Viewing set {st.session_state.history_position + 1} of {len(st.session_state.recommendation_history)}")
    with col_reset:
        if st.button("🔄 Reset All", help="Reset all recommendations and history"):
            st.session_state.shown_movies = set()
            st.session_state.filtered_movies = None
            st.session_state.current_age = None
            st.session_state.current_genres = None
            st.session_state.current_selected_movie = None
            st.session_state.recommendation_history = []
            st.session_state.history_position = -1
            st.success("✅ All recommendations have been reset!")
            st.rerun()

# Enhanced navigation logic
def generate_new_recommendations():
    """Generate new recommendations based on current mode"""
    if recommendation_mode == "Content-Based":
        movie_name, movie_poster, exhausted = recommend_content_based(selectvalue)
        source = selectvalue
    else:  # Age and Genre-Based
        movie_name, movie_poster, exhausted = recommend_age_genre_based(age, selected_genres)
        source = f"Age: {age}, Genres: {', '.join(selected_genres) if selected_genres else 'All'}"

    if exhausted:
        st.warning("⚠️ All available recommendations have been shown! Click 'Reset All' to start over.")
        return False
    elif movie_name:
        # Store new recommendations in history
        recommendation_set = {
            "movies": movie_name,
            "posters": movie_poster,
            "mode": recommendation_mode,
            "source": source,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        st.session_state.recommendation_history.append(recommendation_set)
        st.session_state.history_position = len(st.session_state.recommendation_history) - 1
        return True
    return False

# Navigation logic for previous recommendations
if prev_button and st.session_state.recommendation_history:
    if st.session_state.history_position > 0:
        st.session_state.history_position -= 1
        st.rerun()
    else:
        st.info("📍 You're at the beginning of your recommendation history!")

# Navigation logic for next recommendations
if next_button:
    if st.session_state.recommendation_history and st.session_state.history_position < len(st.session_state.recommendation_history) - 1:
        # Move to next set in history
        st.session_state.history_position += 1
        st.rerun()
    else:
        st.info("📍 You're at the end of your recommendation history! Use 'New Recommendations' to get more.")

# Show initial recommendations logic
if show_button:
    if generate_new_recommendations():
        st.rerun()

# New recommendations logic
if new_recs_button:
    if generate_new_recommendations():
        st.success("🎉 New recommendations generated!")
        st.rerun()

# Display current recommendations with enhanced UI
if st.session_state.recommendation_history and st.session_state.history_position >= 0:
    try:
        display_recommendation_set = st.session_state.recommendation_history[st.session_state.history_position]
        
        # Enhanced header with source info
        st.markdown("### 🎬 Movie Recommendations")
        
        col_source, col_time = st.columns([3, 1])
        with col_source:
            st.markdown(f"**Based on**: {display_recommendation_set['source']}")
        with col_time:
            if 'timestamp' in display_recommendation_set:
                st.caption(f"Generated at: {display_recommendation_set['timestamp']}")
        
        st.markdown("---")
        
        movie_titles = display_recommendation_set["movies"]
        movie_posters = display_recommendation_set["posters"]
        
        # Display movies in a responsive grid
        cols = st.columns(5)  # Show 5 movies per row for desktop/laptop
        
        for i, (name, poster) in enumerate(zip(movie_titles, movie_posters)):
            with cols[i % len(cols)]:
                # Movie card container with fixed height structure
                with st.container():
                    # Title section with fixed height
                    title_container = st.container()
                    with title_container:
                        # Truncate long titles and add tooltip
                        if len(name) > 25:
                            display_name = name[:22] + "..."
                            st.markdown(f'<h4 title="{name}" style="height: 60px; display: flex; align-items: center; margin: 0; padding: 0;">{display_name}</h4>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h4 style="height: 60px; display: flex; align-items: center; margin: 0; padding: 0;">{name}</h4>', unsafe_allow_html=True)
                    
                    # Fixed height poster section
                    poster_container = st.container()
                    with poster_container:
                        if poster and poster != 'placeholder.jpg' and is_valid_image(poster):
                            st.image(poster, use_container_width=True)
                        else:
                            st.image('placeholder.jpg', use_container_width=True, caption="No poster available")
                    
                    # Movie info section
                    info_container = st.container()
                    with info_container:
                        movie_row = st.session_state.movies_df[st.session_state.movies_df['Title'] == name]
                        if not movie_row.empty:
                            movie_info = movie_row.iloc[0]
                            st.markdown(f"**Genre**: {movie_info['Genre']}")
                            st.markdown(f"**Certificate**: {movie_info['Certificate']}")
                            
                            # Add expander for more details
                            with st.expander("More Details"):
                                if 'Overview' in movie_info and pd.notna(movie_info['Overview']):
                                    st.write(f"**Overview**: {movie_info['Overview']}")
                                if 'Director' in movie_info and pd.notna(movie_info['Director']):
                                    st.write(f"**Director**: {movie_info['Director']}")
                                if 'Star' in movie_info and pd.notna(movie_info['Star']):
                                    st.write(f"**Stars**: {movie_info['Star']}")
                
                st.markdown("---")
                    
    except Exception as e:
        st.error(f"❌ Error displaying recommendations: {e}")
