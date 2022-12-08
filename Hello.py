
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
    st.set_page_config(
        page_title="Hello"
    )
    st.write('# Hello')
    st.markdown(
        """Z
        In this project we build a model to analyze the sentiment
        of a review based on Yelp review data.
        
        We use our model to find key factors leading to the 
        sucess for restaurants of particular type. We also integrate
        important information from the dataset to help restaurant owner
        make better decisions.
        """
    )
    