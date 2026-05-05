# from __future__ import annotations

# import json
# import logging
# import sys
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any


# class JsonLineFormatter(logging.Formatter):
#     def format(self, record: logging.LogRecord) -> str:
#         payload: dict[str, Any] = {
#             "ts": datetime.now(timezone.utc).isoformat(),
#             "level": record.levelname,
#             "logger": record.name,
#             "event": getattr(record, "event", record.getMessage()),
#             "message": record.getMessage(),
#         }
#         for key, value in record.__dict__.items():
#             if key.startswith("etl_"):
#                 payload[key.removeprefix("etl_")] = value
#         if record.exc_info:
#             payload["exception"] = self.formatException(record.exc_info)
#         return json.dumps(payload, ensure_ascii=False, default=str)


# def get_logger(name: str = "etl", log_dir: str | Path = "logs") -> logging.Logger:
#     logger = logging.getLogger(name)
#     if logger.handlers:
#         return logger

#     logger.setLevel(logging.INFO)
#     Path(log_dir).mkdir(parents=True, exist_ok=True)

#     formatter = JsonLineFormatter()
#     file_handler = logging.FileHandler(Path(log_dir) / "etl.jsonl", encoding="utf-8")
#     file_handler.setFormatter(formatter)

#     stream_handler = logging.StreamHandler(sys.stdout)
#     stream_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
#     logger.propagate = False
#     return logger


# def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
#     logger.info(event, extra={"event": event, **{f"etl_{k}": v for k, v in kwargs.items()}})


# def log_error(logger: logging.Logger, event: str, **kwargs: Any) -> None:
#     logger.exception(event, extra={"event": event, **{f"etl_{k}": v for k, v in kwargs.items()}})
