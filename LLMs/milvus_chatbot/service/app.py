import streamlit as st

import json

import numpy
import copy

from pymilvus import Collection, utility

import os
from dotenv import load_dotenv
load_dotenv(override=True)

import langchain
from langchain.embeddings import HuggingFaceEmbeddings

import text_generation
from langchain import PromptTemplate

from pymilvus import connections
connection = connections.connect(
  alias="default",
  host="milvus.hyperplane-milvus.svc.cluster.local",
  port=19530,
)

SVC_EP=os.environ['HYPERPLANE_JOB_PARAMETER_LLM_ENDPOINT']
client = text_generation.Client(SVC_EP)

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

whcollection = Collection("WikiHow")
whcollection.load()

assistant_string = "ASSISTANT:\n"
user_string = "USER:\n"
document_string="DOCUMENT:\n"

prompt_template = PromptTemplate.from_template(
    f"""\
{{turns}}
{document_string}{{context}}
{user_string}What does the document say about {{prompt}}
Give me a summary. If the information is not there let me know.

{assistant_string}
"""
)

def generate(what, turns, context, topic, whole_context):
    found = whcollection.search(
            [embeddings.embed_query(what)],
            anns_field="vector",
            param={'metric_type': 'L2',
                        'offset': 0,
                        'params': {'nprobe': 1}
                        },
            limit=1,
            output_fields=['text', 'title'])
    match_title = found[0][0].entity.get('title')
    match_text = found[0][0].entity.get('text')
    match_dist = found[0][0].distance

    retrieved = ""

    if match_title != topic and match_dist < 0.75:
        retrieved = match_text
        retrieved = retrieved[:1024]
        context = retrieved
        whole_context = match_text
        topic = match_title
    turntxt = "\n".join(turns)[-2048:]
    preface = ("No information available" if context is None else context)

    return { 
        'stream': client.generate_stream(prompt=prompt_template.format(turns=turntxt, prompt=what, context=preface),
                                            max_new_tokens=512,
                                            repetition_penalty=1.2,
        ),

        'topic': topic,
        'context': context,
        'whole_context': whole_context
        }

if "messages" not in st.session_state.keys():
    st.session_state.messages = []

if "cache" not in st.session_state.keys():
    st.session_state.cache = [[], "", None, ""]

TURNS = 0
CTX = 1
TOPIC = 2
WHOLE_CTX = 3

turns = st.session_state.cache[TURNS]
context = st.session_state.cache[CTX]
topic = st.session_state.cache[TOPIC]
whole_context = st.session_state.cache[WHOLE_CTX]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

ipt = st.chat_input(key="chat_query", placeholder="How can I run faster?")

if ipt is not None:
    st.session_state.messages.append({'role': 'user', 'content': ipt})
    with st.chat_message("user"):
        st.write(ipt)

if len(st.session_state.messages) != 0 and st.session_state.messages[-1]["role"] != "response":
    with st.chat_message("response"):
        with st.spinner("Thinking..."):
            response = generate(ipt, turns, context, topic, whole_context)
            st.session_state.cache[CTX] = response['context']
            st.session_state.cache[TOPIC] = response['topic']
            st.session_state.cache[WHOLE_CTX] = response['whole_context']
            with st.expander("Active Document"):
                st.write(response['whole_context'])

            turns.append(user_string + ipt)

            resp = ""
            placeholder = st.empty()
            for tok in response['stream']:
                if not tok.token.special:
                    resp += tok.token.text
                    placeholder.markdown(resp)
            placeholder.markdown(resp)

            st.session_state.cache[TURNS].append(assistant_string + resp)

    message = {"role": "response", "content": resp}
    st.session_state.messages.append(message)
