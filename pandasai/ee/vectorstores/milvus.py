import uuid
from typing import Any, Dict, Iterable, List, Optional

from pymilvus import DataType, CollectionSchema, FieldSchema, MilvusClient, model

from pandasai.helpers.logger import Logger
from pandasai.vectorstores.vectorstore import VectorStore

DEFAULT_COLLECTION_NAME = "pandasai"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
UUID_NAMESPACE = "f55f1395-e097-4f35-8c20-90fdea7baa14"


class Milvus(VectorStore):
    """Implementation of VectorStore for Milvus - https://milvus.io/

    Supports adding, updating, deleting and querying code Q/As and documents.

    Args:
        collection_name: Name of the collection.
        embedding_model: Name of the embedding model to use.
        uri: URI of the Milvus instance. Default is "./milvus.db".
        timeout: Timeout for API requests. Default is 5 seconds.
        logger: Optional custom Logger instance.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = model.DefaultEmbeddingFunction(),
        uri: str = "./milvus.db",
        logger: Optional[Logger] = None,
    ) -> None:
        self._qa_collection_name = f"{collection_name}_qa"
        self._docs_collection_name = f"{collection_name}_docs"
        self._logger = logger or Logger()
        self._embedding_model = embedding_model
        self._embedding_dim = self._embedding_model.encode_documents(["foo"])[0].shape[0]
        self._client = MilvusClient(uri=uri)
        self._setup_collections()

    # def _setup_collections(self):
    #     if not self._client.has_collection(self._qa_collection_name):
    #         self._create_collection(self._qa_collection_name)
    #     self._client.load_collection(collection_name=self._qa_collection_name)
    #     # load_state = self._client.get_load_state(collection_name=self._qa_collection_name)
    #     # print(f"Load state of {self._qa_collection_name}: {load_state}")

    #     if not self._client.has_collection(self._docs_collection_name):
    #         self._create_collection(self._docs_collection_name)
    #     self._client.load_collection(collection_name=self._docs_collection_name)
    #     # load_state = self._client.get_load_state(collection_name=self._docs_collection_name)
    #     # print(f"Load state of {self._docs_collection_name}: {load_state}")

    # def _create_collection(self, name: str):
    #     index_params = self._client.prepare_index_params()
    #     index_params.add_index(
    #         field_name="vector",
    #         index_type="AUTOINDEX",
    #         # metric_type="IP",
    #         metric_type="L2",
    #     )
    #     fields = [
    #         FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
    #         FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
    #         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    #     ]
    #     schema = CollectionSchema(fields=fields, description=f"Collection for {name}")
    #     self._client.create_collection(collection_name=name, schema=schema, index_params=index_params)

    def _setup_collections(self):
        self._create_qa_collection(self._qa_collection_name)
        self._create_doc_collection(self._docs_collection_name)

    def _create_qa_collection(self, name: str):
        if not self._client.has_collection(name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            schema.add_field(field_name="qa", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="L2",
            )
            self._client.create_collection(collection_name=name, schema=schema, index_params=index_params)

    def _create_doc_collection(self, name: str):
        if not self._client.has_collection(name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
            schema.add_field(field_name="document", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                # metric_type="IP",
                metric_type="L2",
            )
            self._client.create_collection(collection_name=name, schema=schema, index_params=index_params)


    def add_question_answer(
        self,
        queries: Iterable[str],
        codes: Iterable[str],
        ids: Optional[Iterable[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if len(queries) != len(codes):
            raise ValueError(f"Queries and codes length doesn't match. {len(queries)} != {len(codes)}")

        milvus_ids = self._convert_ids(ids) if ids else [str(uuid.uuid4()) for _ in range(len(queries))]

        qa_str = [self._format_qa(query, code) for query, code in zip(queries, codes)]
        embeddings = self._embedding_model.encode_documents(qa_str)
        self._client.insert(
            self._qa_collection_name,
            [{"id": id_, "qa": qa, "vector": emb} for id_, qa, emb in zip(milvus_ids, qa_str, embeddings)]
        )

        return milvus_ids

    def add_docs(
        self,
        docs: Iterable[str],
        ids: Optional[Iterable[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if ids and len(docs) != len(ids):
            raise ValueError(f"Docs and ids length doesn't match. {len(docs)} != {len(ids)}")
        
        milvus_ids = self._convert_ids(ids) if ids else [str(uuid.uuid4()) for _ in range(len(docs))]
        embeddings = self._embedding_model.encode_documents(docs)

        self._client.insert(
            self._docs_collection_name,
            [{"id": id_, "document": doc, "vector": emb} for id_, doc, emb in zip(milvus_ids, docs, embeddings)]
        )

        return milvus_ids

    def update_question_answer(
        self,
        ids: Iterable[str],
        queries: Iterable[str],
        codes: Iterable[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if not (len(ids) == len(queries) == len(codes)):
            raise ValueError(f"Queries, codes and ids length doesn't match. {len(queries)} != {len(codes)} != {len(ids)}")

        milvus_ids = self._convert_ids(ids)

        self.delete_question_and_answers(milvus_ids)
        return self.add_question_answer(queries, codes, ids=milvus_ids, metadatas=metadatas)

    def update_docs(
        self,
        ids: Iterable[str],
        docs: Iterable[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        if len(ids) != len(docs):
            raise ValueError(f"Docs and ids length doesn't match. {len(docs)} != {len(ids)}")

        milvus_ids = self._convert_ids(ids)

        self.delete_docs(milvus_ids)
        return self.add_docs(docs, ids=milvus_ids, metadatas=metadatas)

    def delete_question_and_answers(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        if ids:
            ids = self._convert_ids(ids)
            self._client.delete(collection_name=self._qa_collection_name, ids=ids)
            return True

    def delete_docs(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        if ids:
            ids = self._convert_ids(ids)
            self._client.delete(collection_name=self._docs_collection_name, ids=ids)
            return True

    def delete_collection(self, collection_name: str) -> Optional[bool]:
        self._client.drop_collection(f"{collection_name}_qa")
        self._client.drop_collection(f"{collection_name}_docs")

    def get_relevant_question_answers(self, question: str, k: int = 1) -> List[dict]:
        embedding = self._embedding_model.encode_documents([question])[0]
        search_params = {
            "metric_type": "L2",
            "param": {"nprobe": 128},
        }
        results = self._client.search(
            collection_name = self._qa_collection_name, 
            data = [embedding.tolist()], 
            limit = k,
            output_fields = ["id", "qa"],
            anns_field = "vector",
            search_params = search_params
        )[0]
        return self._format_search_results(results)

    def get_relevant_docs(self, question: str, k: int = 1) -> List[dict]:
        embedding = self._embedding_model.encode_documents([question])[0]
        search_params = {
            "metric_type": "L2",
            "param": {"nprobe": 128},
        }
        results = self._client.search(
            collection_name = self._docs_collection_name, 
            data = [embedding.tolist()], 
            limit = k,
            output_fields = ["id", "document"],
            anns_field = "vector",
            search_params = search_params
        )[0]
        print('get relevant docs', results)
        return self._format_search_results(results)

    def get_relevant_question_answers_by_id(self, ids: Iterable[str]) -> List[dict]:
        milvus_ids = self._convert_ids(ids)
        results = self._client.get(self._qa_collection_name, ids=milvus_ids)
        return self._format_retrieve_results(results)

    def get_relevant_docs_by_id(self, ids: Iterable[str]) -> List[dict]:
        milvus_ids = self._convert_ids(ids)
        results = self._client.get(self._docs_collection_name, ids=milvus_ids)
        return self._format_retrieve_results(results)
    
    def get_relevant_qa_documents(self, question: str, k: int = 1) -> List[str]:
       
        results = self.get_relevant_question_answers(question, k)
        print('get relevant qa results', results)
        return [res['document'] for res in results]

    def get_relevant_docs_documents(self, question: str, k: int = 1) -> List[str]:

        results = self.get_relevant_docs(question, k)
        return [res["document"] for res in results]

    def _convert_ids(self, ids: Iterable[str]) -> List[str]:
        return [
            id_ if self._is_valid_uuid(id_) else str(uuid.uuid5(uuid.UUID(UUID_NAMESPACE), id_))
            for id_ in ids
        ]

    def _format_search_results(self, results) -> List[dict]:
        print('format search results', results)
        # return [{"id": res.entity["id"], "document": res.entity["document"]} for res in results]
        return [{"id": res['id'], "document": res['entity']['qa']} for res in results]

    def _format_retrieve_results(self, results) -> List[dict]:
        return [{"id": res.id, "document": res.entity["document"]} for res in results]

    def _is_valid_uuid(self, id_: str) -> bool:
        try:
            uuid.UUID(id_)
            return True
        except ValueError:
            return False
