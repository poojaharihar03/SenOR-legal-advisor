import os
import io
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma,FAISS
from langchain_community.llms import HuggingFaceHub
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PDFPlumberLoader,CSVLoader
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from streamlit.components.v1 import components_v1 as components
from gtts import gTTS
import base64
import platform
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3


hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

pdf_file_path='dataset'

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = hf_token,
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
loader =  PyPDFDirectoryLoader(pdf_file_path)
docs = loader.load()

db = FAISS.from_documents(docs, embeddings)
prompt_template = """
            You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            Do not say thank you and tell you are an AI Assistant and be open about everything.
            Use the following pieces of context to answer the users question.
            Context: {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
             Only return the helpful answer below and nothing else.
            """

def model(user_query, max_length, temp):
    repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"max_length": max_length, "temperature": temp})
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=db.as_retriever(k=2),
                                     return_source_documents=True,
                                     verbose=True,
                                     chain_type_kwargs={"prompt": prompt_template})
    response = qa(user_query)["result"]
    return response
        
def text_speech(text):
    tts = gTTS(text=text, lang='en')

    # Save speech to a BytesIO object
    speech_bytes = io.BytesIO()
    tts.write_to_fp(speech_bytes)
    speech_bytes.seek(0)

    # Convert speech to base64 encoding
    speech_base64 = base64.b64encode(speech_bytes.read()).decode('utf-8')

    return speech_base64
