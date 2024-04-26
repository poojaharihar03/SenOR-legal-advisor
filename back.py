import os
import io
import pickle
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
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
# loader =  PyPDFDirectoryLoader(pdf_file_path)
#docs = loader.load()
new_db = FAISS.load_local("merge",embeddings, allow_dangerous_deserialization=True)
# db = FAISS.from_documents(docs, embeddings)
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

def model(user_query, max_length, temp):
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"max_length": max_length, "temperature": temp})
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=new_db.as_retriever(k=2),
                                     return_source_documents=True,
                                     verbose=True,
                                     chain_type_kwargs={"prompt": prompt})
    # return qa(user_query)["result"]
    response = qa(user_query)["result"]
    
    answer_start = response.find("Answer:")
    if answer_start != -1:
        assistant_index = response.find("assistant:")
        if assistant_index != -1:
            response = response[:assistant_index].strip()
        answer = response[answer_start + len("Answer:"):].strip()
        last_period_index = answer.rfind('.')
        if last_period_index != -1:
            answer = answer[:last_period_index + 1]
        return answer
    
   

        
def text_speech(text):
    tts = gTTS(text=text, lang='en')

    # Save speech to a BytesIO object
    speech_bytes = io.BytesIO()
    tts.write_to_fp(speech_bytes)
    speech_bytes.seek(0)

    # Convert speech to base64 encoding
    speech_base64 = base64.b64encode(speech_bytes.read()).decode('utf-8')
    return speech_base64
