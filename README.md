# Movie Recommender System

A Python-based movie recommendation system that suggests movies based on user preferences and content similarity.

> **This project was completed as a college mini project by a group of 3 members: K. Satwika, A. Chandrika, and P. Abhi Shekar (myself).**

## Features

- Content-based movie recommendations
- TF-IDF vectorization for movie descriptions
- Interactive movie search and recommendations

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required data files from Google Drive:
   - [movies_data.pkl](https://drive.google.com/file/d/1bmn6WJ2b2UCHF_SQkgAT3oyxSAuQF42U/view?usp=drive_link)
   - [tfidf_index.bin](https://drive.google.com/file/d/1bnnYz-Y0L5QWwEZfQWlHhoLggBUDklpm/view?usp=drive_link)
   - [tfidf_vectorizer.pkl](https://drive.google.com/file/d/1I_rxnLpaJxoh_HypjGIylaU3-MdBvc63/view?usp=drive_link)
   - [imdb-movies.csv](https://drive.google.com/file/d/1ff4dlynJte3_YCOX2AAC5ZQHk1L-Hkx5/view?usp=drive_link)
   
   Place these files in the root directory of the project.

## Usage

Run the main script:
```bash
python Movies.py
```

## Example Screenshots

Below are some example screenshots of the Movie Recommender System in action:

![Search Example](image%20(1).png)
*Search for movies by title or keyword.*

![Age and Genre Recommendations](image%20(2).png)
*Recommendations based on age and selected genres.*

![Content-Based Recommendations](image%20(3).png)
*Content-based recommendations for a selected movie.*

![Movie Details](image%20(4).png)
*Detailed view of a recommended movie.*

## Project Structure

- `Movies.py`: Main application file
- `requirements.txt`: Project dependencies
- `movie_recommender_development.ipynb`: Jupyter notebook with development and analysis

## Data Files

The following data files are required to run the application (not included in the repository due to size):
- `movies_data.pkl`: Contains processed movie data
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer model
- `tfidf_index.bin`: TF-IDF index for fast similarity search
- `imdb-movies.csv`: Raw movie dataset from IMDB

These files can be downloaded from the Google Drive links provided in the Setup section.

## Note

This project uses TF-IDF vectorization and requires the following data files (not included in the repository due to size):
- movies_data.pkl
- tfidf_vectorizer.pkl
- tfidf_index.bin 