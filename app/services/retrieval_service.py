from __future__ import annotations
# standarisasi knowledge base (kb)/ tool retrieval
# app/services/retrieval_service.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import CHROMA_DB_PATH, KNOWLEDGE_BASE_COLLECTION, EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K

class RetrievalService:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.kb = Chroma(
            collection_name=KNOWLEDGE_BASE_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_PATH,
        )

    def retrieve(self, query: str, k: int = RETRIEVAL_TOP_K):
        return self.kb.similarity_search(query, k=k)