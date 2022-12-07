import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def state_distribution(data):
    data = data.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(y=data.index, x= data.values, palette= sns.color_palette("RdBu_r", len(data)),ax=ax)
    ax.set_ylabel('Category', fontsize=14)
    ax.set_xlabel('Number of Restaurants', fontsize=14)
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

if __name__ == '__main__':
    st.set_page_config(
        page_title="Restaurant Information",
    )
    st.sidebar.header("Restaurant Information")
    st.markdown("# Restaurant Information")
    state = pd.read_csv('./data/distribution_state.csv', index_col=[0])
    city = pd.read_csv('./data/distribution_city.csv', index_col=[0])
    category = state.index
    x = st.selectbox('Choose the type of your restaurant', category)
    fig = state_distribution(state.loc[x])
    st.pyplot(fig)
    fig = city_distribution(city.loc[x])
    st.pyplot(fig)
    