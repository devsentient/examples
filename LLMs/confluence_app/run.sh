PROJECT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"
pip install -r requirements.txt

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export STREAMLIT_RUNONSSAVE=True
streamlit run app.py --server.port 8787 --browser.serverAddress localhost
