from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain.storage._lc_store import create_kv_docstore
import chromadb
from chromadb.config import Settings
from os import path
import json
import yaml
import sqlite3
from typing import Tuple, Iterator
from typing import Sequence, Optional
from langchain.schema import BaseStore
from pypika import Query, Table, Field, Column
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


class _ChromaStore_sqlite3(BaseStore[str, bytes]):

    def __init__(self, doc_path, table_name, sqlite3_db_path):
        self.doc_path = doc_path
        self.sqlite3_db_path = sqlite3_db_path
        self.db_path = path.join(self.doc_path, self.sqlite3_db_path)
        self.table = Table(table_name)
        self.id_column = Field('id')
        self.data_column = Field('data')
        self._create_table()

    def _create_table(self):
        id_column = Column('id', 'VARCHAR(50)', nullable=False)
        data_column = Column('data', 'VARCHAR(2500)', nullable=False)
        create_table_query = Query.create_table(self.table).columns(id_column, data_column).if_not_exists()
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(create_table_query.get_sql())

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            data_list = []

            for key in keys:
                key = [key]
                select_query = Query.from_(self.table).select(self.data_column).where(self.id_column.isin(key))
                cursor.execute(select_query.get_sql())
                results = cursor.fetchall()

                for result in results:
                    if result[0] is not None:
                        data_list.append(json.loads(result[0]).encode("utf-8"))
                    else:
                        data_list.append(None)

        return data_list

    def mset(self, key_value_pairs: Sequence[Tuple[int, bytes]]) -> None:
        insert_queries = []
        for key, value in key_value_pairs:
            insert_query = Query.into(self.table).columns(self.id_column, self.data_column).insert(key, json.dumps(
                value.decode('utf-8')))
            insert_queries.append(insert_query)

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            for query in insert_queries:
                cursor.execute(query.get_sql())

            connection.commit()

    def mdelete(self, keys: Sequence[int]) -> None:
        delete_query = Query.from_(self.table).delete().where(self.id_column.isin(keys))

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(delete_query.get_sql())

            connection.commit()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        select_query = Query.from_(self.table).select(self.id_column)
        if prefix:
            select_query = select_query.where(self.id_column.like(f'{prefix}%'))

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(select_query.get_sql())

            for row in cursor.fetchall():
                yield row[0]


class _SentenceTransformerModel:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query]).tolist()[0]


class CustomMultiVectorRetriever():

    def get_retriever(self, collection_name: str) -> RunnableParallel:
        """
        Get retriever as Runnable base on selected collection name.

        Args:
            collection_name: collection name in vectordb

        Returns:
            RunnableParallel: Langchain runnable
        """

        # read config
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # Define Chromadb Client
        try:
            httpClient = chromadb.HttpClient(
                host=config['vectorstore']['host'], port=config['vectorstore']['port'],
                settings=Settings(chroma_client_auth_provider=config['vectorstore']['chroma_client_auth_provider_auth'],
                                  chroma_client_auth_credentials=config['vectorstore']['chroma_client_auth_credentials'])
            )
        except ModuleNotFoundError:
            try:
                httpClient = chromadb.HttpClient(
                    host=config['vectorstore']['host'], port=config['vectorstore']['port'],
                    settings=Settings(
                        chroma_client_auth_provider=config['vectorstore']['chroma_client_auth_provider_authn'],
                        chroma_client_auth_credentials=config['vectorstore']['chroma_client_auth_credentials'])
                )
            except ModuleNotFoundError:
                httpClient = chromadb.HttpClient(
                    host=config['vectorstore']['host'], port=config['vectorstore']['port'],
                    settings=Settings(
                        chroma_client_auth_provider=config['vectorstore']['chroma_client_auth_provider'],
                        chroma_client_auth_credentials=config['vectorstore']['chroma_client_auth_credentials'])
                )

        # Select embedding model from config
        embedding_model_name = config['embedding_model'][collection_name]
        embedding_model = _SentenceTransformerModel(embedding_model_name)

        # Define remote vectorstore
        remote_vectorestore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            client=httpClient
        )

        # Define doc store
        cs = _ChromaStore_sqlite3(
            doc_path=config['docstore']['doc_path'],
            table_name="docstore",
            sqlite3_db_path=config['docstore'][collection_name]
        )
        store = create_kv_docstore(cs)

        id_key = "doc_id"

        # The retriever (empty to start)
        mvr_retriever = MultiVectorRetriever(
            vectorstore=remote_vectorestore,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": config['topK']['k']},  # top k
        )

        # wrap to langchain runnable
        # retriever = RunnableParallel(
        #     {"context": mvr_retriever, "question": RunnablePassthrough()}
        # )
        retriever = mvr_retriever

        return retriever