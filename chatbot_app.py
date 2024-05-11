from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain_community.llms import Ollama

import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
import streamlit as st

def get_chroma_collections(server_url):
    # Initialize the LangChain database client
    # 連線設定
    httpClient = chromadb.HttpClient(
        host=server_url, port=8000,
        settings=Settings(chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",chroma_client_auth_credentials="admin:admin")
    )
    # Get all collections from the Chroma Server
    collections = httpClient.list_collections()

    for c in collections:
        print(c)

get_chroma_collections("64.176.47.89")
#ngrok http 11434 --host-header="localhost:11434" --scheme http

with st.sidebar:
    llm_model = st.text_input("LLM 模型", key="llm_model")
    ollama_api_url = st.text_input("Ollama API URL", key="ollama_api_url")
    collection_version = st.text_input("Choose collection version")

#    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 AIA Chatbot")
st.caption("🚀 AIA 課程查詢機器人")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

llm = Ollama(model=llm_model, base_url=ollama_api_url)

if prompt := st.chat_input():
    print(f"model = ${llm_model}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print(st.session_state.messages)
    response = llm.invoke(prompt)
    #response = llm.invoke(st.session_state.messages)
    #msg = response.choices[0].message.content
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

