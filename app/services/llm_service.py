from __future__ import annotations
# app/services/llm_service.py
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, GROQ_MODEL_NAME

from app.config import (
    GROQ_API_KEY,
    GROQ_CRITIC_MODEL,
    GROQ_GENERATOR_MODEL,
    GROQ_MODEL_NAME,
)

def build_llm(temperature: float = 0.0) -> ChatGroq:
    """Backward-compat factory (digunakan mode 1/2/3 lama). Maps ke generator."""
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=700,
    )

def build_generator_llm(temperature: float = 0.0) -> ChatGroq:
    """Generator LLM factory: Llama-3.1-8B-Instant (frozen §3).

    Args:
        temperature: default 0.0 (system prompt §7.4 reproducibility).
    """
    return ChatGroq(
        model=GROQ_GENERATOR_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=700,
    )

def build_critic_llm(temperature: float = 0.0) -> ChatGroq:
    """Critic LLM factory: Llama-3.3-70B-Versatile (frozen §3).

    Args:
        temperature: default 0.0 (deterministic verdict, system prompt §7.4).

    Returns:
        ChatGroq instance dengan max_tokens lebih besar untuk JSON output
        + 4 rationale strings.
    """
    return ChatGroq(
        model=GROQ_CRITIC_MODEL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=1024,
    )