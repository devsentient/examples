import requests
import streamlit as st
import pandas as pd
import json
from io import StringIO

def run_sentiment_analysis(string):
    ## send request
    endpoint = "your_model_end_point:8787/"
    data = {
        "string": string
    }
    headers = {
        "Content-Type": "application/json",
    }

    try:
        result = requests.post(endpoint, data = json.dumps(data), headers=headers).json()
        # print('result obtained')
        # print(result)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    return result
    
    
    
# defines an h1 header
st.title("Sentence analysis")

st.subheader('Paste some text here get classified')

## type text in the box
txt = st.text_area('Text to analyze', '''
     ''')

st.write("Sentence:", txt) 

txt_result = run_sentiment_analysis(txt)
# print('txt_result', txt_result)

# st.write('Topic:', txt_result)
st.metric(label="Topic", value=txt_result)

st.subheader('Upload a text file here to get classified')

## file uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)
    
#     # To read file as string:
    string_data = stringio.read()
#     st.write(string_data)

    
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    # print(dataframe)
    results = []
    for sentence in dataframe.Sentence.tolist():
        results.append(run_sentiment_analysis(sentence))
    dataframe['Topic Prediction'] = results
    st.write(dataframe)