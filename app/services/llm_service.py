from __future__ import annotations
# app/services/llm_service.py
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, GROQ_MODEL_NAME

from app.config import (
    GEMINI_GENERATOR_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_CRITIC_MODEL,
    GROQ_GENERATOR_MODEL,
    GROQ_MODEL_NAME,
    MISTRAL_API_KEY,
    MISTRAL_CRITIC_MODEL,
    MISTRAL_GENERATOR_MODEL,
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


def build_generator_gemini(temperature: float = 0.0):
    """Generator LLM factory alternatif: Gemini 2.5 Flash (Google).

    Digunakan HANYA oleh run_experiment_gemini.py — tidak menggantikan
    model Generator frozen (Llama-3.1-8B via Groq).

    Args:
        temperature: default 0.0 untuk reproducibility.

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_GENERATOR_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=700,
    )


def build_generator_mistral(temperature: float = 0.0):
    """Generator LLM factory alternatif: Mistral Small 2603.

    Digunakan HANYA oleh run_experiment_gemini.py — tidak menggantikan
    model Generator frozen (Llama-3.1-8B via Groq).

    Args:
        temperature: default 0.0 untuk reproducibility.

    Returns:
        ChatMistralAI instance.
    """
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model=MISTRAL_GENERATOR_MODEL,
        mistral_api_key=MISTRAL_API_KEY,
        temperature=temperature,
        max_tokens=700,
    )


def build_critic_mistral(temperature: float = 0.0):
    """Critic LLM factory alternatif: Mistral Medium Latest.

    Digunakan HANYA oleh run_experiment_gemini.py — tidak menggantikan
    model Critic frozen (Llama-3.3-70B via Groq).

    Args:
        temperature: default 0.0 untuk deterministic verdict.

    Returns:
        ChatMistralAI instance.
    """
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(
        model=MISTRAL_CRITIC_MODEL,
        mistral_api_key=MISTRAL_API_KEY,
        temperature=temperature,
        max_tokens=1024,
    )