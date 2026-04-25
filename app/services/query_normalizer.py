from __future__ import annotations
# kebutuhan riset semantic equivalence || Normalisasi ringan untuk ticker/sinonim/typo.
# app/services/query_normalizer.py
import re
from dataclasses import dataclass

@dataclass
class NormalizedQuery:
    raw_query: str
    normalized_query: str
    detected_tickers: list[str]
    intent: str

ISSUER_SYNONYMS = {
    "bank central asia": "BBCA",
    "bca": "BBCA",
    "bank rakyat indonesia": "BBRI",
    "bri": "BBRI",
    "bank mandiri": "BMRI",
    "mandiri": "BMRI",
    "telkom indonesia": "TLKM",
    "telkom": "TLKM",
}

TYPO_MAP = {
    "bbac": "bbca",
    "bcaa": "bbca",
    "tlmk": "tlkm",
    "bmrii": "bmri",
}

def infer_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["harga", "price", "closing", "close", "last"]):
        return "price_lookup"
    if any(k in q for k in ["laporan", "revenue", "profit", "fundamental", "keuangan"]):
        return "fundamental_lookup"
    if any(k in q for k in ["tren", "volatilitas", "trend", "volatility"]):
        return "trend_analysis"
    return "general_stock_qa"

def normalize_query(question: str) -> NormalizedQuery:
    q = question.strip().lower()
    q = re.sub(r"\s+", " ", q)

    for wrong, correct in TYPO_MAP.items():
        q = re.sub(rf"\b{re.escape(wrong)}\b", correct, q)

    detected = []
    for name, ticker in ISSUER_SYNONYMS.items():
        if re.search(rf"\b{re.escape(name)}\b", q):
            q = re.sub(rf"\b{re.escape(name)}\b", ticker.lower(), q)
            if ticker not in detected:
                detected.append(ticker)

    for token in re.findall(r"\b[A-Za-z]{4}\b", q):
        token = token.upper()
        if token in {"BBCA", "BBRI", "BMRI", "TLKM"} and token not in detected:
            detected.append(token)

    q = q.replace("saham ", "")
    q = q.replace("emiten ", "")

    return NormalizedQuery(
        raw_query=question,
        normalized_query=q,
        detected_tickers=detected,
        intent=infer_intent(q),
    )