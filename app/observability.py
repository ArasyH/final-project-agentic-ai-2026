import langfuse
from langfuse import Langfuse
from langfuse import observe, propagate_attributes
from config import (
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)

langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

def create_trace(session_id: str, question: str):
    """Membuat trace utama untuk satu sesi percakapan."""
    return langfuse_client.trace(
        trace_name="saham-idx30-chat",
        session_id=session_id,
        input=question
    )

def log_cache_span(trace, hit: bool, score: float, latency_ms: float):
    """Mencatat aktivitas pengecekan cache sebagai span turunan dari trace."""
    trace.span(
        name="semantic_cache_check",
        metadata={
            "cache_hit": str(hit), # Dikonversi ke string sesuai aturan v4
            "similarity_score": str(round(score, 4)),
            "latency_ms": str(round(latency_ms, 2)),
            "threshold": "0.85",
        }
    )

def log_rag_generation(trace, question: str, answer: str,
                       contexts: list, latency_ms: float):
    """Mencatat aktivitas LLM generation jika cache miss."""
    trace.generation(
        name="rag_agent_response",
        input=question,
        output=answer,
        metadata={
            "contexts": str(contexts), # Dikonversi ke string agar aman
            "latency_ms": str(round(latency_ms, 2)),
        }
    )

def finalize_trace(trace, answer: str, cache_hit: bool, total_ms: float):
    """Memperbarui trace utama dengan hasil akhir dan tag sentral."""
    trace.update(
        output=answer,
        tags=["saham-idx30", "hit" if cache_hit else "miss"],
        metadata={
            "cache_hit": str(cache_hit),
            "total_latency_ms": str(round(total_ms, 2)),
            "scenario": "cache_hit" if cache_hit else "cache_miss",
        }
    )

def flush():
    """Memaksa pengiriman data telemetri ke server Langfuse."""
    langfuse_client.flush()