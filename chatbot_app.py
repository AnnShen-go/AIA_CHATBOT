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

    openai_api_key = ""
    llm_model = ""
    ollama_api_url = ""

    if model_platform_type == "ollama":
        llm_model = st.text_input("LLM 模型", key="llm_model", value="ycchen/breeze-7b-instruct-v1_0")
        ollama_api_url = st.text_input("Ollama API URL", key="ollama_api_url", value="http://8cf8-140-109-17-45.ngrok-free.app")
    else:   
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    #    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    #    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    #    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    #collection_version = st.text_input("Choose collection version")
st.title("💬 AIA 課程小幫手")
st.caption("🚀 AIA Course Assistant")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "嗨! 我是 AIA 台灣人工智慧學校的虛擬助理，隨時準備回答您的課程問題"}]

if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Example prompts
example_prompts = [
    "AIA 有什麼課程?",
    "請介紹經理人專班",
    "有 LLM 相關的課程嗎?",
    "請給我技術領袖班的課程簡章"
]

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

    pipeline.set_retriever("default")

    response = pipeline.query(prompt)
    return response

clicked_button_text = None

cols = st.columns(4)

for i, button_text in enumerate(example_prompts):
    with cols[i]:
        if st.button(button_text, disabled=st.session_state.button_disabled):
            clicked_button_text = button_text
            st.session_state.button_disabled = True

# 處理按鈕點擊事件
if clicked_button_text:
    st.session_state.messages.append({"role": "user", "content": clicked_button_text})
    with st.chat_message("user"):
        st.write(clicked_button_text)
    response = run_rag_process(clicked_button_text)  # 調用回應函數
    if response:
        print(response)
        msg = response["result"]
        st.session_state.messages.append({"role": "assistant", "content": msg})
        with st.chat_message("assistant"):
            st.write(msg)
    # 重新啟用按鈕
    st.session_state.button_disabled = False        
    st.experimental_rerun()


if prompt := st.chat_input():
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

