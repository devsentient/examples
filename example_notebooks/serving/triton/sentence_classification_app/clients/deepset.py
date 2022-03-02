import numpy as np
import sys
from functools import partial
import os
import tritonclient.grpc as tritongrpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as tritonhttpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
import torch
from transformers import AutoTokenizer
from torch.nn import functional as F
import json

tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
VERBOSE = False
# sentence1 = 'Who are you voting for 2021?'
# sentence2 = 'Jupiter’s Biggest Moons Started as Tiny Grains of Hail'
# sentence3 = 'Hi Matt, your payment is one week past due. Please use the link below to make your payment'
labels = ['business', 'space and science', 'politics']
input_name = ['input__0', 'input__1']
output_name = 'output__0'


def run_inference(sentence, model_name='deepset', url='hyperplane-triton.default:8000', model_version='1'):
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=VERBOSE)
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version)
    # I have restricted the input sequence length to 256
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                     return_tensors='pt', max_length=256,
                                     truncation=True, padding='max_length')
    
    input_ids = inputs['input_ids']
    input_ids = np.array(input_ids, dtype=np.int32)
    mask = inputs['attention_mask']
    mask = np.array(mask, dtype=np.int32)
    mask = mask.reshape(4, 256) 
    input_ids = input_ids.reshape(4, 256)
    input0 = tritonhttpclient.InferInput(input_name[0], (4,  256), 'INT32')
    input0.set_data_from_numpy(input_ids, binary_data=False)
    input1 = tritonhttpclient.InferInput(input_name[1], (4, 256), 'INT32')
    input1.set_data_from_numpy(mask, binary_data=False)
    output = tritonhttpclient.InferRequestedOutput(output_name,  binary_data=False)
    response = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1], outputs=[output])
    embeddings = response.as_numpy('output__0')
    embeddings = torch.from_numpy(embeddings)
    sentence_rep = embeddings[:1].mean(dim=1)
    label_reps = embeddings[1:].mean(dim=1)
    similarities = F.cosine_similarity(sentence_rep, label_reps)
    closest = similarities.argsort(descending=True)
    # results = []
    # for ind in closest:
    #     results.append(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
    # return json.dumps(results)
    return labels[closest[0]]

if __name__ == '__main__':
    import sys
    sent = sys.argv[1]
    print(run_inference(sys.argv[1]))
    

# def run(string="", url="hyperplane-triton.default:8000"):
#     results = run_inference(sentence=string, url=url)
#     return results