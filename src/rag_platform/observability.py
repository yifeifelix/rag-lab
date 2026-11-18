"""JSON events with an explicit field allowlist; never serialize request bodies."""
import json
import logging
from datetime import datetime, timezone

LOGGER = logging.getLogger('rag_platform')
FIELDS = {'event', 'request_id', 'method', 'status', 'duration_ms', 'provider',
          'error_type', 'corpus_revision', 'documents', 'chunks', 'operation',
          'generation_mode', 'refused', 'cache_tier', 'role', 'stage'}


def event(name: str, level=logging.INFO, **fields):
    payload = {'time': datetime.now(timezone.utc).isoformat(), 'event': name}
    payload.update({key: value for key, value in fields.items() if key in FIELDS})
    LOGGER.log(level, json.dumps(payload, ensure_ascii=False))


def configure_logging():
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
