pip install -r hyperplane-triton-api/requirements.txt
export FLASK_APP='hyperplane-triton-api/app.py'
pip install flask
python -m flask run -p 8787 --host=0.0.0.0