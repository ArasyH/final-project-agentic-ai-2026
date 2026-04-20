# config.py
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY  = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY  = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
EMBED_MODEL_NAME = SentenceTransformer("all-MiniLM-L6-v2")


CHROMA_DB_PATH       = os.getenv("CHROMA_DB_PATH")
GROQ_MODEL_NAME      = os.getenv("GROQ_MODEL_NAME")
SIMILARITY_THRESHOLD = os.getenv("SIMILARITY_THRESHOLD", 0.85)
CACHE_TTL_HOURS      = os.getenv("CACHE_TTL_HOURS", 8) # Cache akan kadaluarsa setelah 8 jam (1 sesi market)