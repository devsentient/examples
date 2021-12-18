# frontend/main.py

import requests
import streamlit as st
from PIL import Image
import json
import numpy as np
import pandas as pd
import base64
import argparse
import time
import random


def sent_infer_request_in_cluster(img):
    ## send request to a model inference endpoint that's served on hyperplane
    ## As the services are all internally authenticated, no need to pass JWT tokens 
    endpoint = "http://image_recognition-inference-endpoint:8787/infer"
    data = {
        "image": img
    }
    headers = {
        "Content-Type": "application/json",
    }
    
    # result = requests.post(endpoint, data = json.dumps(data), headers=headers).json()
    
    ## for demo purpose we are going to use a random result
    try:
        result = {"category": random.choice(["cat", "cat", "cat"])}
        print(result)    
    except:
        result = {}

    return result


## 
## Streamlit UI components
##
    
    
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Image Recognition Cats Dogs Fruits")

# displays a file uploader widget
image_upload = st.file_uploader("Choose an image")

# displays a file uploader widget
image_type = st.selectbox("Choose an image type", ['fruit', 'animal', 'other'])

# displays a button
if st.button("Get annotation"):
    if image_upload is not None:
        # try:
        image_raw = Image.open(image_upload)
        image = np.array(image_raw)[:, :, ::-1]

        result = sent_infer_request_in_cluster(image)
        st.json(result)
        # except: 
            # st.text("Could not process image")


image = Image.open('cat.png')

st.image(image, caption='Cat')

