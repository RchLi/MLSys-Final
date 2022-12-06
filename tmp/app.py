import streamlit as st
# from metaflow import Flow, namespace
# from metaflow import get_metadata, metadata
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
# @st.cache
# def get_latest_successful_run(flow_name: str):
#     for r in Flow(flow_name).runs():
#         #if r.successful: 
#         return r

if __name__ == '__main__':
    # FLOW_NAME = 'MyClassificationFlow'
    # namespace(None)
    # metadata('./tmp')
    # get_metadata()
    # latest_run = Flow(FLOW_NAME).latest_successful_run
    # model = latest_run.data.svm_Jap
    # vectorizer = latest_run.data.vectorizer_Jap
    # positive_words = latest_run.data.positive_words
    # negative_words = latest_run.data.negative_words

    # Directly load from pickle
    with open('./models/svm.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./models/vec.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    st.markdown("# Review Sentiment Analysis")
    review = st.text_input('Your Review', '')
    if review == '':
        label = ''
        st.write('The review is ')
    else:
        x = [review]
        x = vectorizer.transform(x)
        label = model.predict(x)[0]
        st.write('The review is **%s**'%(label))

    