"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')


# Backround image
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
                
        }
        </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","EDA","About Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)


        # Set # Backround image
        set_png_as_page_bg('resources/imgs/home_bg.png')

        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[:])
        movie_2 = st.selectbox('Second Option',title_list[:])
        movie_3 = st.selectbox('Third Option',title_list[:])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,top_n=10)
                    st.title("We think you'll like:")
                    #for i,j in enumerate(top_recommendations):
                    for i in top_recommendations:
                        st.subheader(i)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("We created two movie recommender systems to choose from for this project. The systems will be discused below.")
        st.markdown("To see the complete code used to create the recommender systems please visit our [GitHub Repo](https://github.com/henre7860/unsupervised_leanring_movie_recommender.git)")
        st.write("### Content Based Filtering")

        st.write("This type of filtering system recommends movies to you on the basis of what you actually like. Imagine you love to watch comedy movies so a content-based recommender system will recommend you other related comedy movies which belong to this category.")
        

        st.write("### Collaborative Filtering")
        st.write("Collaborative filtering is a method to create predictions based on what a user is interested in by collecting information, in this case the ratings of certain movies.")

        st.image('resources/imgs/recommender.png',use_column_width=True,caption= "Simple illustration of Filters")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    if page_selection == "EDA":
        st.write("# Exploratory Data Analysis")
        st.info("Exploratory data analysis (EDA) is used by data scientists to analyze datasets and to find the main characteristics of the data to better identify the quality of the data. By doing EDA it may help us to manipulate the data to find the best answer to a proposed question.")

        st.write("Below we can see the visualization of the data.")
        st.image('resources/imgs/movies_p_year.png',use_column_width=True, caption="Movies per Year")
        st.write('The max of 2513 movies was released in the year 2015.')

        st.image('resources/imgs/movies_p_genre.png',use_column_width=True, caption="Movies per Genre")
        st.write('From the graph we can see that the genre Drama is the most common genre for the movies in the dataset with 25606 movies followed by comedy with 16870 movies.')

        # Set # Backround image
        #set_png_as_page_bg('resources/imgs/eda_bg_2.png')

    if page_selection == "About Us":
        st.write("### Team 9: Members")

        st.write("Hendrik van den Berg")
        st.write("Neo Ntsako Mashele")
        st.write("Tracy Zanele Lushaba")
        st.write("Lwazi Cele")
        st.write("Botseetsa Loveness Nkadimeng")
        st.write("Collen Bothma")

        st.info("In this project our team was tasked to create a movie recommender system as part of the Data Science course at Explore Data Science Academy.")
        st.markdown("The full project can be found on our [GitHub Repo](https://github.com/henre7860/unsupervised_leanring_movie_recommender.git)")





        st.image('resources/imgs/EDSA_logo.png',use_column_width=True)


if __name__ == '__main__':
    main()
