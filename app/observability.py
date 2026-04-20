# import os
# from langfuse import Langfuse
# from langfuse.decorators import langfuse_context
# from dotenv import load_dotenv

# load_dotenv()
# LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
# LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
# LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

# langfuse_client = Langfuse(
#     public_key = LANGFUSE_PUBLIC_KEY,
#     secret_key = LANGFUSE_SECRET_KEY,
#     host = LANGFUSE_BASE_URL, 
# )

# def create_trace(session_id: str, question: str):
#     """Buat trace baru untuk satu interaksi."""
#     return langfuse_client.trace(
#         name="chatbot-query",
#         session_id=session_id,
#         input=question,
#         tags=["saham-idx30"],
#     )

# def log_cache_span(hit: bool, score: float, latency_ms: float):
#     """Log hasil pengecekan semantic cache ke trace yang sedang aktif."""
#     langfuse_context.update_current_observation(
#         name="semantic-cache-check",
#         input={"similarity_score": score},
#         output={"cache_hit": hit},
#         metadata={
#             "hit":            hit,
#             "score":          score,
#             "latency_ms":     latency_ms,
#             "threshold_used": 0.85,
#         },
#     )


# def log_rag_generation(question: str, answer: str,
#                         contexts: list[str], latency_ms: float,
#                         token_usage: dict | None = None):
#     """Log output RAG agent — ini yang diekspor untuk RAGAS."""
#     langfuse_context.update_current_observation(
#         name="rag-answer",
#         model="gpt-3.5-turbo",
#         input=question,
#         output=answer,
#         metadata={
#             "contexts":   contexts,
#             "latency_ms": latency_ms,
#         },
#     )
#     if token_usage:
#         langfuse_context.update_current_observation(usage=token_usage)

# def update_trace_metadata(answer: str, cache_hit: bool,
#                    total_latency_ms: float):
#     """Tutup trace dengan ringkasan akhir."""
#     langfuse_context.update_current_observation(
#         output=answer,
#         metadata={
#             "cache_hit":         cache_hit,
#             "total_latency_ms":  total_latency_ms,
#             "scenario":          "hit" if cache_hit else "miss",
#         },
#     )
#     langfuse_client.flush()

# observability.py
import os
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"]       = LANGFUSE_HOST

langfuse_client = Langfuse()

def log_cache_span(hit: bool, score: float, latency_ms: float):
    langfuse_context.update_current_observation(
        metadata={
            "cache_hit":        hit,
            "similarity_score": score,
            "latency_ms":       round(latency_ms, 2),
            "threshold":        0.85,
        }
    )

def log_rag_generation(question: str, answer: str,
                       contexts: list, latency_ms: float):
    langfuse_context.update_current_observation(
        input=question,
        output=answer,
        metadata={
            "contexts":   contexts,
            "latency_ms": round(latency_ms, 2),
        }
    )

def update_trace_metadata(cache_hit: bool, total_ms: float):
    langfuse_context.update_current_observation(
        tags=["saham-idx30", "hit" if cache_hit else "miss"],
        metadata={
            "cache_hit":        cache_hit,
            "total_latency_ms": round(total_ms, 2),
            "scenario":         "cache_hit" if cache_hit else "cache_miss",
        }
    )

def flush():
    langfuse_client.flush()