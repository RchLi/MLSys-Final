from math import dist
import streamlit as st
# from metaflow import Flow, namespace
# from metaflow import get_metadata, metadata
# import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(y=data.index, x= data.values, palette= sns.color_palette("RdBu_r", len(data)),ax=ax)
    ax.set_ylabel('Category', fontsize=14)
    ax.set_xlabel('Count of reviews', fontsize=14)
    ax.set_title('Count of Reviews by Cuisine Type', fontsize=15)
    for i,v in enumerate(data):
        ax.text(v, i+0.15, str(v),fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)

    return fig

if __name__ == '__main__':
    # information
    st.markdown("# Restaurant Information")
    distribution = pd.read_csv('./data/distribution_by_category.csv', index_col=[0])
    category = distribution.index
    x = st.selectbox('Choose the type of your restaurant', category)
    fig = plot_distribution(distribution.loc[x].sort_values(ascending=False))
    st.pyplot(fig)
    
    # model prediction
    st.markdown("# Review Sentiment Analysis")
    # load model with pickle
    with open('./models/svm.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./models/vec.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    review = st.text_input('Your Review', '')
    if review == '':
        label = ''
        st.write('The review is ')
    else:
        x = [review]
        x = vectorizer.transform(x)
        label = model.predict(x)[0]
        st.write('The review is **%s**'%(label))

    