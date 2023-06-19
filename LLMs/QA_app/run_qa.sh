PROJECT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

# Set the name of the virtual environment
VENV_NAME=.venv

## Check if the virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
  # Create the virtual environment
  python -m venv "$VENV_NAME"
fi
source "$VENV_NAME/bin/activate"
# export PATH="/root/.local/bin:$PATH"
pip freeze
pip install -r requirements.txt --ignore-installed
pip freeze
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export STREAMLIT_RUNONSSAVE=True


streamlit run streamlit_app_blog.py --server.port 8787 --browser.serverAddress localhost --server.fileWatcherType none

deactivate