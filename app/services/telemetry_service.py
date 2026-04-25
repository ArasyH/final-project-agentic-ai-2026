from __future__ import annotations
# Langfuse logging yang konsisten
# app/services/telemetry_service.py
import time
from langfuse import Langfuse
from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

class TelemetryService:
    def __init__(self) -> None:
        self.client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )

    def start_trace(self, session_id: str, question: str, mode: str):
        return self.client.trace(
            name="chat-query",
            session_id=session_id,
            input=question,
            metadata={"mode": mode},
            tags=["idx30", mode],
        )

    def event(self, trace, name: str, metadata: dict | None = None, input_data=None, output_data=None):
        trace.event(
            name=name,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )

    def flush(self):
        self.client.flush()