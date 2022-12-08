import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
def state_distribution(data):
    data = data.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(y=data.index, x= data.values, palette= sns.color_palette("RdBu_r", len(data)),ax=ax)
    ax.set_ylabel('State', fontsize=14)
    ax.set_xlabel('Number', fontsize=14)
    ax.set_title('Number of Restaurants by State', fontsize=15)
    for i,v in enumerate(data):
        ax.text(v, i+0.15, str(v),fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)

    return fig


def city_distribution(data, k=15):
    data = data.sort_values(ascending=False)[:k]
    fig, ax = plt.subplots(figsize=(1.1*k, 7))

    sns.barplot(x=data.index, y=data.values, palette=sns.color_palette("GnBu_r", len(data)),ax=ax)
    ax.set_xlabel('City', labelpad=10, fontsize=14)
    ax.set_ylabel('Number', fontsize=14)
    ax.set_title('Number of Restaurants by City (Top %d)' % (k), fontsize=15)
    ax.tick_params(labelsize=14)
    plt.xticks(rotation=15)
    for  i, v in enumerate(data):
        ax.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)
    
    return fig

def word_img(words):
    text = ' '.join(words)
    wordcloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(wordcloud)
    plt.axis("off")
    #plt.tight_layout(pad = 0)
    return fig

if __name__ == '__main__':
    st.set_page_config(page_title="Restaurant Information",layout='wide')

    st.sidebar.header("Restaurant Information")

    col1, col2 = st.columns(2)
    state = pd.read_csv('./data/distribution_state.csv', index_col=[0])
    city = pd.read_csv('./data/distribution_city.csv', index_col=[0])
    with open('./data/cusine_words','rb') as f:
        cusine_words = pickle.load(f)

    with col1:
        st.markdown("# Restaurant Information")
        category = state.index
        x = st.selectbox('Choose the type of your restaurant', category)

        st.markdown('## Important factors for %s restaurant' % (x))
        fig = word_img(cusine_words[x])
        st.pyplot(fig)

    with col2:
        st.markdown('## Distribution of %s restaurant' % (x))
        fig = state_distribution(state.loc[x])
        st.pyplot(fig)
        fig = city_distribution(city.loc[x])
        st.pyplot(fig)
  
    