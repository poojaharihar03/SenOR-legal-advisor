import os
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
import toml

hf_token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# if hf_token is None:
#     raise ValueError("Hugging Face token is not set. Please set it in Streamlit settings.")
# else:
#     # Set Hugging Face token as environment variable


# csv_file_path = 'rights.csv'
pdf_file_path='https://github.com/poojaharihar03/SenOR-legal-advisor/tree/main/dataset'

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = hf_token,
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
loader =  PyPDFDirectoryLoader(pdf_file_path)
docs = loader.load()

db = FAISS.from_documents(docs, embeddings)
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

def model(user_query, max_length, temp):
    repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"max_length": max_length, "temperature": temp})
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=db.as_retriever(k=2),
                                     return_source_documents=True,
                                     verbose=True,
                                     chain_type_kwargs={"prompt": prompt})
    # return qa(user_query)["result"]
    response = qa(user_query)["result"]
    answer_start = response.find("Answer:")
    if answer_start != -1:
        answer = response[answer_start + len("Answer:"):].strip()
        return answer
    else:
        return "Sorry, I couldn't find the answer."

def text_speech(text):
    # Save the generated text to an audio file
    tts = gTTS(text=text, lang='en')
    audio_file_path = "generated_audio.mp3"
    tts.save(audio_file_path)
    
    # Play the audio file
    play_audio(audio_file_path)

    # Remove the audio file after playing (optional)
    os.remove(audio_file_path)

def play_audio(audio_file_path):
    # Check the platform to determine the appropriate command for playing audio
    if platform.system() == "Darwin":
        os.system("afplay " + audio_file_path)
    elif platform.system() == "Linux":
        os.system("aplay " + audio_file_path)
    elif platform.system() == "Windows":
        os.system("start " + audio_file_path)
    else:
        st.warning("Unsupported platform for text-to-speech")


def stop_audio():
    play.stop()
