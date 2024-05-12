import sys
from vector_db import db_and_embedding




aia_collection = db_and_embedding.CollectionSelector("xt131028_v1")

# 明確指定 db 詳細參數的寫法:
# aia_collection = CollectionSelector(collection_name="xt131028_v1", host="64.176.47.89", port=8000,chroma_client_auth_credentials="admin:admin")

db = aia_collection.db

query = "技術領袖培訓全域班"
documents = db.similarity_search(query)

print(documents)
import sys
sys.exit

# 切換 DB
db = aia_collection.switch_collection("xt131028_v1.2")

# 顯示可以用的 collections
aia_collection.show_collection_list()

# 使用 embedding
embedding = aia_collection.embedding

print(aia_collection.host)
print(aia_collection.port)
print(aia_collection.chroma_client_auth_credentials)
print(aia_collection.collections_info)
print(aia_collection.embedding_model_mapping)
