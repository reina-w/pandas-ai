import unittest
from unittest.mock import patch, MagicMock

from pandasai.ee.vectorstores.milvus import Milvus


class TestMilvus(unittest.TestCase):
    @patch("pandasai.ee.vectorstores.milvus.MilvusClient", autospec=True)
    def setUp(self, mock_client):
        self.mock_client = mock_client.return_value
        # self.mock_client.set_timeout = MagicMock()
        self.milvus = Milvus()

    def test_init(self):
        self.assertEqual(self.milvus._qa_collection_name, "pandasai-qa")
        self.assertEqual(self.milvus._docs_collection_name, "pandasai-docs")
        # self.mock_client.set_timeout.assert_called_once_with(5)
        self.mock_client.has_collection.assert_any_call("pandasai-qa")
        self.mock_client.has_collection.assert_any_call("pandasai-docs")

    @patch("pandasai.ee.vectorstores.milvus.Milvus._create_collection")
    def test_setup_collections(self, mock_create_collection):
        self.mock_client.has_collection.return_value = False
        self.milvus._setup_collections()
        mock_create_collection.assert_any_call("pandasai-qa")
        mock_create_collection.assert_any_call("pandasai-docs")

    @patch("pandasai.ee.vectorstores.milvus.MilvusClient", autospec=True)
    def test_add_question_answer(self, mock_client):
        queries = ["What is AI?", "How does it work?"]
        codes = ["print('Hello')", "for i in range(10): print(i)"]
        mock_embedding = MagicMock()
        self.milvus._embedding_model = mock_embedding
        self.milvus._client = mock_client
        mock_embedding.encode_documents.return_value = ["embedding1", "embedding2"]

        ids = self.milvus.add_question_answer(queries, codes)

        mock_client.insert.assert_called_once()
        self.assertEqual(len(ids), 2)

    def test_add_question_answer_different_dimensions(self):
        with self.assertRaises(ValueError):
            self.milvus.add_question_answer(["What is AI?"], ["print('Hello')", "for i in range(10): print(i)"])

    @patch("pandasai.ee.vectorstores.milvus.MilvusClient", autospec=True)
    def test_add_docs(self, mock_client):
        docs = ["Document 1", "Document 2"]
        mock_embedding = MagicMock()
        self.milvus._embedding_model = mock_embedding
        self.milvus._client = mock_client
        mock_embedding.encode_documents.return_value = ["embedding1", "embedding2"]

        ids = self.milvus.add_docs(docs)

        mock_client.insert.assert_called_once()
        self.assertEqual(len(ids), 2)

    def test_add_docs_different_dimensions(self):
        with self.assertRaises(ValueError):
            self.milvus.add_docs(["Document 1"], ["id1", "id2"])

    @patch("pandasai.ee.vectorstores.milvus.Milvus._convert_ids", side_effect=lambda x: x)
    def test_update_question_answer(self, mock_convert_ids):
        ids = ["id1", "id2"]
        queries = ["What is AI?", "How does it work?"]
        codes = ["print('Hello')", "for i in range(10): print(i)"]
        mock_embedding = MagicMock()
        self.milvus._embedding_model = mock_embedding
        mock_embedding.encode_documents.return_value = ["embedding1", "embedding2"]

        new_ids = self.milvus.update_question_answer(ids, queries, codes)

        self.mock_client.delete.assert_called_once_with(collection_name="pandasai-qa", ids=ids)
        self.mock_client.insert.assert_called_once()
        self.assertEqual(new_ids, ids)

    @patch("pandasai.ee.vectorstores.milvus.Milvus._convert_ids", side_effect=lambda x: x)
    def test_update_docs(self, mock_convert_ids):
        ids = ["id1", "id2"]
        docs = ["Document 1", "Document 2"]
        mock_embedding = MagicMock()
        self.milvus._embedding_model = mock_embedding
        mock_embedding.encode_documents.return_value = ["embedding1", "embedding2"]

        new_ids = self.milvus.update_docs(ids, docs)

        self.mock_client.delete.assert_called_once_with(collection_name="pandasai-docs", ids=ids)
        self.mock_client.insert.assert_called_once()
        self.assertEqual(new_ids, ids)

    @patch("pandasai.ee.vectorstores.milvus.Milvus._convert_ids", side_effect=lambda x: x)
    def test_delete_question_and_answers(self, mock_convert_ids):
        ids = ["id1", "id2"]
        result = self.milvus.delete_question_and_answers(ids)
        self.mock_client.delete.assert_called_once_with(collection_name="pandasai-qa", ids=ids)
        self.assertTrue(result)

    @patch("pandasai.ee.vectorstores.milvus.Milvus._convert_ids", side_effect=lambda x: x)
    def test_delete_docs(self, mock_convert_ids):
        ids = ["id1", "id2"]
        result = self.milvus.delete_docs(ids)
        self.mock_client.delete.assert_called_once_with(collection_name="pandasai-docs", ids=ids)
        self.assertTrue(result)

    @patch("pandasai.ee.vectorstores.milvus.MilvusClient", autospec=True)
    def test_get_relevant_question_answers(self, mock_client):
        question = "What is AI?"
        mock_embedding = MagicMock()
        self.milvus._embedding_model = mock_embedding
        mock_embedding.return_value = ["embeddings"] 
          
        self.milvus._client = mock_client
        self.milvus.get_relevant_question_answers(question, k=3)
        
        mock_client.search.assert_called_once_with(
            "pandasai-qa",
            data=["embeddings"], 
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=3,
            expr="",
            output_fields=["id", "document"]
        )

    @patch("pandasai.ee.vectorstores.milvus.MilvusClient", autospec=True)
    def test_get_relevant_docs(self, mock_client):
        question = "What is AI?"
        mock_embedding = MagicMock()
        self.milvus._embedding_model = mock_embedding
        mock_embedding.encode_documents.return_value = ["embedding"]

        self.milvus._client = mock_client
        self.milvus.get_relevant_docs(question, k=3)

        mock_client.search.assert_called_once_with(
            "pandasai-docs",
            data=["embedding"],
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=3,
            expr="",
            output_fields=["id", "document"]
        )

    @patch("pandasai.ee.vectorstores.milvus.Milvus._convert_ids", side_effect=lambda x: x)
    def test_get_relevant_question_answers_by_id(self, mock_convert_ids):
        ids = ["id1", "id2"]
        self.milvus.get_relevant_question_answers_by_id(ids)
        self.mock_client.get.assert_called_once_with("pandasai-qa", ids=ids)

    @patch("pandasai.ee.vectorstores.milvus.Milvus._convert_ids", side_effect=lambda x: x)
    def test_get_relevant_docs_by_id(self, mock_convert_ids):
        ids = ["id1", "id2"]
        self.milvus.get_relevant_docs_by_id(ids)
        self.mock_client.get.assert_called_once_with("pandasai-docs", ids=ids)

    def test_convert_ids(self):
        ids = ["id1", "id2"]
        converted_ids = self.milvus._convert_ids(ids)
        self.assertEqual(len(converted_ids), len(ids))

    def test_is_valid_uuid(self):
        valid_uuid = "f55f1395-e097-4f35-8c20-90fdea7baa14"
        invalid_uuid = "invalid-uuid"
        self.assertTrue(self.milvus._is_valid_uuid(valid_uuid))
        self.assertFalse(self.milvus._is_valid_uuid(invalid_uuid))


if __name__ == "__main__":
    unittest.main()
