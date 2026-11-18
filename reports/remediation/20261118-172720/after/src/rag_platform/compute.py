"""Bounded shared CPU execution; model inference never runs on the event loop."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import weakref

_POOLS = {}
_GATES = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


async def run_cpu(function, *args, workers=4, **kwargs):
    loop = asyncio.get_running_loop()
    with _LOCK:
        if workers not in _POOLS:
            _POOLS[workers] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rag-cpu')
        pool = _POOLS[workers]
        gates = _GATES.setdefault(loop, {})
        gate = gates.setdefault(workers, asyncio.Semaphore(workers))
    async with gate:
        # Cancellation must not release capacity while the worker is still running.
        future = loop.run_in_executor(pool, partial(function, *args, **kwargs))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await asyncio.shield(future)
            raise
