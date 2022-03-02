PROJECT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
pip install streamlit fastapi opencv-python pillow uvicorn
pip install --force-reinstall --no-deps bokeh==2.4.1
streamlit run slnlp.py --server.port 8787 --browser.serverAddress localhost