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
# Set Hugging Face Hub API token as environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# hf_token = os.getenv("HUGGINGFACE_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# csv_file_path = 'rights.csv'
pdf_file_path='https://github.com/poojaharihar03/SenOR-legal-advisor/tree/main/dataset'

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = hf_token,
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
loader =  PyPDFDirectoryLoader(pdf_file_path)
docs = loader.load()

db = Chroma.from_documents(docs, embeddings)
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

def model(user_query, max_length, temp):
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'
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


# # Frontend code
st.title("🤖 SenOR ")
with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ SenOR Legal Advisor </h1>",
                unsafe_allow_html=True)
    st.markdown("<h7 style='text-align:left;font-size:20px;'>This app is a smart legal chatbot that is integrated into an easy-to-use platform. This would give lawyers "
                "instant access to legal information of Women’s Legal Rights and remove the need for laborious manual research in books or regulations using the power "
                "of Large Language Models</h7>", unsafe_allow_html=True)
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)

    st.markdown(" - Users can adjust token length to control the length of generated responses, allowing for customization based on specific requirements or constraints.")
    st.markdown(" - Users can adjust the temp to control response randomness. Higher values (e.g., 0.5) produce diverse but less focused responses, while low values (e.g., 0.1) result in more focused but less varied answers.")
    st.markdown("- users can now use text to speech")
    st.markdown("<h2 style='font-family:Georgia;font-size:20px;'>Press enter to stop the audio</h1>", unsafe_allow_html=True)

    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                unsafe_allow_html=True)
    max_length = st.slider("Token Max Length", min_value=512, max_value=1024, value=512, step=128)
    temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def submit():
    st.session_state.something = st.session_state.widget
    st.empty()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

if user_prompt := st.chat_input("enter your query"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model(user_prompt, max_length, temp)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


# if st.checkbox("Convert to speech"):
if st.button("Convert to Speech"):
        if st.session_state.messages[-1]["role"] == "assistant":
            text_speech(st.session_state.messages[-1]["content"])

