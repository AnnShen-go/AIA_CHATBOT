from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain_community.llms import Ollama

#import chromadb
#from chromadb.config import Settings
#from langchain.vectorstores import Chroma
import streamlit as st

__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#def get_chroma_collections(server_url):
    # Initialize the LangChain database client
    # 連線設定
#    httpClient = chromadb.HttpClient(
#        host=server_url, port=8000,
#        settings=Settings(chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",chroma_client_auth_credentials="admin:admin")
#   )
    # Get all collections from the Chroma Server
#    collections = httpClient.list_collections()

    #for c in collections:
    #    print(c)



#get_chroma_collections("64.176.47.89")
#ngrok http 11434 --host-header="localhost:11434" --scheme http

with st.sidebar:
    model_platform_type = st.radio(
        "請選擇想用的模型平台",
        ["ollama", "open_ai"],
        captions = ["Ollama 地端平台", "Open AI 雲端平台"],
        label_visibility="hidden")

    st.divider()

    if model_platform_type == "ollama":
        llm_model = st.text_input("LLM 模型", key="llm_model")
        ollama_api_url = st.text_input("Ollama API URL", key="ollama_api_url")
    else:   
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    #    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    #    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    #    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    #collection_version = st.text_input("Choose collection version")
st.title("💬 AIA Chatbot")
st.caption("🚀 AIA 課程查詢機器人")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def run_rag_process(prompt):

    if model_platform_type == 'ollama':
        if llm_model == "" or ollama_api_url == "":
            st.warning('請輸入 LLM 模型名稱及 Ollama API URL', icon="⚠️")
            return
    elif openai_api_key == "":
        st.warning('請輸入 OpenAI API Key', icon="⚠️")
        return

    from query_rag.qa_chain_module import QARetrievalPipeline

    # 初始化流水線，傳入必要的参数
    pipeline = QARetrievalPipeline(
        collection_name="stella_400_18admission_and_qa",
        nomic_api_key="nk-BCjzlgQXfYdZNA6kUZ-6Jeq1NIG2JhlQJaTe9BedpDc", #換成自己的nomic_api_key
        chroma_host="64.176.47.89",
        chroma_port=8000,
        groq_api_key="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",#換成自己的groq_api_key
        groq_model_name="llama3-8b-8192",
        ollama_model=llm_model,
        ollama_base_url=ollama_api_url, # 設定自建的LLM服務位置
        openai_api_key=openai_api_key,
        openai_model_name="gpt-4o"
    )

    # 選擇要使用的 LLM 模型（True 使用 PrimeHub Ollama 模型，False 使用 Groq 模型）
    if model_platform_type == 'ollama':
        pipeline.set_llm(use_primehub=True)
    else:
        pipeline.set_llm(use_openai=True)

    response = pipeline.query(prompt)
    return response


if prompt := st.chat_input():
    print(f"model = ${llm_model}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print(st.session_state.messages)
    response = run_rag_process(prompt)
    #response = llm.invoke(st.session_state.messages)
    #msg = response.choices[0].message.content
    if response != None:
        msg = response["result"]
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

