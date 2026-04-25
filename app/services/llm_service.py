from __future__ import annotations
# app/services/llm_service.py
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, GROQ_MODEL_NAME

def build_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=700,
    )