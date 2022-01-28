"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv')
#ratings = pd.read_csv('resources/data/ratings.csv')
#movies.dropna(inplace=True)

def data_preprocessing(df):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    #movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    #movies_subset = movies[:subset_size]
    #return movies_subset

        
    genres = df['genres']
    genres = [genre.split("|") for genre in genres]
    df['genre_corpus']= genres
    df['genre_corpus'] = df.genre_corpus.apply(lambda x:" ".join(x))
    cvect = CountVectorizer() 
    vectors = cvect.fit_transform(df['genre_corpus']).toarray()
    #print(vectors.shape)
    return vectors

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list, top_n):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    new_df = movies.copy()
        
        
    movie_index_1 = new_df[new_df['title'] == movie_list[0]].index[0]
    movie_index_2 = new_df[new_df.title == movie_list[1]].index[0]
    movie_index_3 =  new_df[new_df.title == movie_list[2]].index[0]

    # Limit sample to 10 % for initial app to run    
    df_1 = new_df.sample(frac = 0.25)
    df_2 = new_df.iloc[[movie_index_1,movie_index_2,movie_index_3]]
    df_2 = df_2.append(df_1)
        
    vectors = data_preprocessing(df_2)
    similarity = cosine_similarity(vectors)
        
    distances_1 = similarity[0]
    distances_2 = similarity[1]
    distances_3 = similarity[2]
        
    sim_score_1 = pd.Series(distances_1).sort_values(ascending = False)
    sim_score_2 = pd.Series(distances_2).sort_values(ascending = False)
    sim_score_3 = pd.Series(distances_3).sort_values(ascending = False)
        
    # Getting the indexes of the 10 most similar movies
    sim_score_list = sim_score_1.append(sim_score_2).append(sim_score_3).sort_values(ascending = False)


    # Appending the names of movies
    indexes = list(sim_score_list.index)
        
    recommended_movies = []
        
    top_n = 10
    for i in indexes:
            
        if df_2.iloc[i].title not in movie_list and len(recommended_movies) < top_n:
            recommended_movies.append(df_2.iloc[i].title)
        
            
    return recommended_movies


