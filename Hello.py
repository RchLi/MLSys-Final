
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # information
    # st.markdown("# Restaurant Information")
    # state = pd.read_csv('./data/distribution_state.csv', index_col=[0])
    # city = pd.read_csv('./data/distribution_city.csv', index_col=[0])
    # category = state.index
    # x = st.selectbox('Choose the type of your restaurant', category)
    # fig = state_distribution(state.loc[x])
    # st.pyplot(fig)
    # fig = city_distribution(city.loc[x])
    # st.pyplot(fig)
    
    # model prediction
    # st.markdown("# Review Sentiment Analysis")
    # # load model with pickle
    # with open('./models/svm.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # with open('./models/vec.pkl', 'rb') as f:
    #     vectorizer = pickle.load(f)
    # review = st.text_input('Your Review', '')
    # if review == '':
    #     label = ''
    #     st.write('The review is ')
    # else:
    #     x = [review]
    #     x = vectorizer.transform(x)
    #     label = model.predict(x)[0]
    #     st.write('The review is **%s**'%(label))

    st.set_page_config(
        page_title="Hello"
    )
    st.write('# Hello 👋')
    st.markdown(
        """
        In this project we build a model to analyze the sentiment
        of a review based on Yelp review data.
        
        We use our model to find key factors leading to the 
        sucess for restaurants of particular type. We also integrate
        important information from the dataset to help restaurant owner
        make better decisions.
        """
    )
    