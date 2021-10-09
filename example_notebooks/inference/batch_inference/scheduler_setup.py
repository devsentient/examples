"""
startup script to be preloaded to the workers

"""
import pip._internal as pip
def install(package):
    pip.main(['install', package])

try:
    import s3fs
except ImportError:
    install('s3fs')
    import s3fs

import gcsfs
import os
import pickle
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification



## start up script

def download_from_cloud(local_file_name, remote_file_name):
    """
    Download a file to gcp or s3.
    """
    import os
    import s3fs
    import gcsfs
    cloud_name = remote_file_name.split('://')[0]
    if cloud_name =='gs':
        fs = gcsfs.GCSFileSystem(project=os.environ['GCP_PROJECT'])
    elif cloud_name =='s3':
        fs = s3fs.S3FileSystem()
    else:
        raise NameError(f'cloud name {cloud_name} unknown')
    try:    
        print(f'download from {remote_file_name} to {local_file_name}')
        fs.get(remote_file_name, local_file_name, recursive=True)
        print("done downloading!")
    except Exception as exp:
        print(f"download failed: {exp}")

    return


def load_models(model_path):
    """
    Load the model from the unzipped local model path
    """

    model = TFDistilBertForSequenceClassification.from_pretrained(f'{model_path}/clf')
    model_name, max_len = pickle.load(open(f'{model_path}/info.pkl', 'rb'))
    loaded_models['model'] = (model, model_name, max_len)

    return loaded_models


def load_model_from_pretrained(model_name):
    """
    Load the model from the unzipped local model path
    """

    model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
    max_len = 20
    loaded_models['model'] = (model, model_name, max_len)

    return loaded_models

loaded_models = {}
model_name = "distilbert-base-uncased"
loaded_models = load_model_from_pretrained(model_name)
