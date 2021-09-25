import os
import json
from flask import request
from flask import jsonify
from datetime import datetime

from geventhttpclient import client
from clients.image_client import run

from flask import Flask, Response, render_template

application = Flask(__name__)

def get_boolean(val, default):
    try:
        if(str(val).lower() == "false"):
            val = False
        elif(str(val).lower() == "true"):
            val = True
    except Exception as e:
        print(e)
        return default

def get_number(val, default):
    try:
        n = int(val)
        return n
    except Exception as e:
        return default
    
@application.route("/health-ready", methods=['GET'])
def health_check():
    return jsonify({"message": "Ready"})

@application.route("/", methods=['POST'], strict_slashes=False)
def image_client():
    try:
        data = request.get_json()
        if(data != None):
            image = data.get("image", "")
            model_name = data.get("model_name", "")
            url = data.get("url", "hyperplane-triton.default:8000")
        elif(request.form != None):
            image = request.form.to_dict()["image"]
            model_name = request.form.to_dict()["model_name"]
            url = request.form.to_dict()["url"]
        else:
            data = request.get_json(force=True)
            image = data.get("image", "")
            model_name = data.get("model_name", "")
            url = data.get("url", "hyperplane-triton.default:8000")
            
        client_run = run(
            image = image,
            model_name = model_name,
            verbose = False, 
            async_set = False, 
            streaming = False,
            model_version = "", 
            batch_size= 1,
            classes = 3,
            scaling = "INCEPTION", 
            url = url, 
            protocol = "http"
        )
        response = application.response_class(
        response=client_run,
        status=200,
        mimetype='application/json'
        )
        return response
    except Exception as e:
        print(e)
        response = json.dumps({
            "hasError" : True,
            "error": "There was an error processing your request. Please check your inputs",
            "errorMessage": str(e)
        })
        return jsonify(response)


if __name__ == "__main__":
    application.run(debug=True)
