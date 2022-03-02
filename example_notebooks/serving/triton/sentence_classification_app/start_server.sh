PROJECT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
pip install -r requirements.txt
export FLASK_APP='app.py'
python -m flask run -p 8787 --host=0.0.0.0