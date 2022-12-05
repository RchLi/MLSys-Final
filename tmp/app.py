import streamlit as st
from metaflow import Flow
from metaflow import get_metadata, metadata
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

@st.cache
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

if __name__ == '__main__':
    FLOW_NAME = 'MyClassificationFlow'
    metadata('./')
    print(get_metadata())
    latest_run = get_latest_successful_run(FLOW_NAME)
    model = latest_run.data.svm_Jap
    vectorizer = latest_run.data.vectorizer_Jap
    positive_words = latest_run.data.positive_words
    negative_words = latest_run.data.negative_words

    st.markdown("# Review Sentiment Analysis")
    review = st.text_input('Your Review', '')
    if review == '':
        label = ''
    else:
        x = [review]
        x = vectorizer.transform(x)
        label = model.predict(x)[0]
    st.write('The review is **%s**'%(label))

    