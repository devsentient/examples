"""
### simple FastAPI App for inference with the trained AutoML model
"""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import time
import pandas as pd
import numpy as np
import json
import io
import os
import sys
from autogluon.tabular import TabularDataset, TabularPredictor
from utils import upload_to_cloud, download_from_cloud


def download_model():
    try:
        local_file_path = 'models'
        remote_file_path =f"gs://shakdemo-hyperplane/data/environmental-sensor-data/{local_file_path}"
        download_from_cloud(local_file_path, remote_file_path)
        print('downloaded model successfully')
    except Exception as e:
        print(f'download failed with {e}')
    return 


def run_inference(data:dict) ->pd.DataFrame:
    df = pd.DataFrame.from_dict(data, orient = 'index').T
    data = TabularDataset(df)
    predictor = TabularPredictor.load('models') 
    y_pred = predictor.predict(df)
    return y_pred


global model
model = download_model()

class MyRequest(BaseModel):
    data: dict

app = FastAPI()


@app.get("/health-ready")
def health_check():
    return jsonable_encoder({"message": "Ready"})

@app.get("/")
def root():
    return {"hello":"world"}

@app.post("/infer")
async def infer(req: Request):
    if req.headers['Content-Type'] == 'application/json':
        i = MyRequest(** await req.json())
    elif req.headers['Content-Type'] == 'multipart/form-data':
        i = MyRequest(** await req.form())
    elif req.headers['Content-Type'] == 'application/x-www-form-urlencoded':
        i = MyRequest(** await req.form())
    r = json.loads(i.json())
    return {
        'prediction': str(run_inference(data = r['data']).values[0])
    }


if __name__ == '__main__':
    jsondata = {}
    data = {
        'ts': 1594512094.3859746, 
        'co': 0.0049559386483912, 
        'humidity': 51.0,
        'light': 0.0,
        'lpg': 0.0076508222705571,
        'smoke': 0.0204112701224129,
        'temp': 22.7
    }
    jsondata['data'] = data
    jsondata = json.dumps(jsondata)
    print(jsondata)
    starttime = time.time()
    r = json.loads(jsondata)
    output = run_inference(r['data'])
    print(output)
    print(f'used {time.time() - starttime} seconds')
