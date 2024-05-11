import db_and_embedding

aia_collection = db_and_embedding.CollectionSelector()
db=aia_collection.switch_collection("simple_html_Stella_Base_zh_v3_1792d_t13_aia_with_qa")
print(aia_collection.embedding)
print(db)