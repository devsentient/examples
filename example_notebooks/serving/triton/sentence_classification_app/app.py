import os
import json
from flask import request
from flask import jsonify
from datetime import datetime

from geventhttpclient import client
from clients.deepset import run_inference

from flask import Flask, Response, render_template

application = Flask(__name__)

    
@application.route("/health-ready", methods=['GET'])
def health_check():
    return jsonify({"message": "Ready"})

@application.route("/", methods=['POST'], strict_slashes=False)
def client():
    try:
        data = request.get_json()
        url = data.get("url", "hyperplane-triton.default:8000")
        print('line 24 data', data)
        results = run_inference(sentence=data["string"], url=url)
        print('line 26', results)
        return json.dumps(results)
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
