import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    st.set_page_config(
        page_title="Model Prediction",
    )
    st.sidebar.header("Model Prediction")
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