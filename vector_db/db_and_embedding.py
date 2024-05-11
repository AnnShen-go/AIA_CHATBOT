### 寫一個環境引入參數取代現有的APIKEY
### 把collection 做成列表，並且對應 embedding
### 包成一個 class 方便使用
### 建立不同的 embedding
### 建立dict給外部用
### get embedding name from data


from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
import chromadb
import os
from typing import List
from sentence_transformers import SentenceTransformer

class Stella_Base_zh_v3_1792d:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query]).tolist()[0]

# 各項 API KEY
os.environ['NOMIC_API_KEY'] = 'nk-MT_fzB1g18s_js7cf54ecyAeN4eLvd1S4pa8FFwaJqI'



class CollectionSelector:
    # 各個 collection 所用 embedding 對應表
    collections_info={
        'xt131028_v1.1':'nomic-embed-text-v1.5',
        'xt131028_v1':'nomic-embed-text-v1.5',
        'xt131028':'nomic-embed-text-v1.5',
        'simple_html_nomic_embed_text_v1_f16_t5':'nomic-embed-text-v1.5',
        'simple_html_Stella_Base_zh_v3_1792d_t13_aia_with_qa':'infgrad/stella-base-zh-v3-1792d',
    }
    # 顯示列表
    def show_collection_list(self):
        print("collection列表:")
        for k,i in self.collections_info.items():
            print(f"{k}: {i}")

    # 建立 embedding_model
    def build_embedding_model(self,collection_name):
        match collection_name:
            case "xt131028_v1.1":
                return NomicEmbeddings(model=self.collections_info[collection_name])
            case "xt131028_v1":
                return NomicEmbeddings(model=self.collections_info[collection_name])
            case "xt131028":
                return NomicEmbeddings(model=self.collections_info[collection_name])
            case "simple_html_nomic_embed_text_v1_f16_t5":
                return NomicEmbeddings(model=self.collections_info[collection_name])#"infgrad/stella-base-zh-v3-1792d"
            case "simple_html_Stella_Base_zh_v3_1792d_t13_aia_with_qa":
                return Stella_Base_zh_v3_1792d(self.collections_info[collection_name])#"infgrad/stella-base-zh-v3-1792d"
    # 建立 db 連線
    def db_connection(self,httpClient,collection_name,embedding_function):
        return Chroma(
            client=httpClient,
            collection_name=collection_name,
            embedding_function= embedding_function,
        )
    
    # 初始化
    def __init__(self):

        # 連線設定
        self.collection_name = "xt131028_v1"

        self.httpClient = chromadb.HttpClient(
            host='64.176.47.89', port=8000,
            settings=chromadb.config.Settings(chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",chroma_client_auth_credentials="admin:admin")
        )

        # build embedding model
        self.embedding = self.build_embedding_model(self.collection_name)
        
        # db connection
        self.db = self.db_connection(self.httpClient, self.collection_name, self.embedding)

        self.show_collection_list()

    # 切換 database
    def switch_collection(self,collection_name):
        if collection_name in self.collections_info:
            self.collection_name = collection_name
            self.embedding = self.build_embedding_model(self.collection_name)
            print(self.embedding)
            self.db = self.db_connection(self.httpClient, self.collection_name, self.embedding)
            print(f"(已切換 collection: {self.collection_name} )")
            return self.db
        else:
            print(f"你所輸入的 collection: {collection_name} 不存在。")


    # collections = httpClient.list_collections() #資料庫列表
    # httpClient.delete_collection(name="xt131028") #刪除資料庫
    # collection = httpClient.create_collection("xt131028") #創建資料庫
    # tell LangChain to use our client and collection name


if __name__ == '__main__':

    query = "技術領袖培訓全域班"
    aia_collection = CollectionSelector()
    db = aia_collection.db
    documents = db.similarity_search(query)
    for i in documents:
        print(i,"\n\n")
    
    print("\n\n----------------\n\n")

    # 切換 DB
    db = aia_collection.switch_collection("simple_html_Stella_Base_zh_v3_1792d_t13_aia_with_qa")
    documents = db.similarity_search(query)
    for i in documents:
        print(i,"\n\n")
    
    # 顯示可以用的 collections
    aia_collection.show_collection_list()

    # 使用 embedding
    embedding = aia_collection.embedding

    #print(aia_collection.httpClient.list_collections())
    print(db.get())
