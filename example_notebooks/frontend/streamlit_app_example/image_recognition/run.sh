apt-get update
apt-get -y install libgl1
apt-get install -yq libgtk2.0-dev
pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
pip install opencv-python==4.5.2.52
pip install fastapi==0.70.0
pip install gcsfs

pip install streamlit fastapi opencv-python pillow uvicorn
pip install --force-reinstall --no-deps bokeh==2.4.1
streamlit run frontend_examples/slexample.py --server.port 8787 --browser.serverAddress localhost