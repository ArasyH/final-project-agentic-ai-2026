# config.py
from __future__ import annotations
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "saham-idx30-chat")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

LANGFUSE_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

CHROMA_DB_PATH       = os.getenv("CHROMA_DB_PATH")
KNOWLEDGE_BASE_COLLECTION = os.getenv("KNOWLEDGE_BASE_COLLECTION", "stock_knowledge_base")
SEMANTIC_CACHE_COLLECTION = os.getenv("SEMANTIC_CACHE_COLLECTION", "semantic_cache")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L12-v2",
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.85))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 8))# Cache akan kadaluarsa setelah 8 jam (1 sesi market)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 3))

DEFAULT_EXPERIMENT_MODE = os.getenv("DEFAULT_EXPERIMENT_MODE", "mode_3_full_agentic")
 