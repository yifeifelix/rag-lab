"""Actual HTTP workload, separate from the in-process synthetic experiment."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import httpx
import numpy as np

from rag_platform.config import RAGConfig


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--output', type=Path, default=RAGConfig().reports_dir/'http_benchmark.json')
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error('requests and concurrency must be positive')
    headers = {'Authorization': f'Bearer {os.environ["RAG_ADMIN_TOKEN"]}'} if os.getenv('RAG_ADMIN_TOKEN') else {}
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=30, limits=limits) as client:
        health = await client.get('/health'); health.raise_for_status()
        stats = await client.get('/api/v1/stats'); stats.raise_for_status()
        questions = [q['question'] for q in stats.json()['sample_queries']]
        if not questions:
            raise SystemExit('No QA samples configured for this server')
        gate = asyncio.Semaphore(args.concurrency)
        async def one(i):
            async with gate:
                start = time.perf_counter()
                try:
                    r = await client.post('/api/v1/query', json={'query':questions[i % len(questions)], 'use_cache':args.cache})
                    payload = r.json() if r.status_code == 200 else {}
                    return {'status':r.status_code, 'latency_ms':(time.perf_counter()-start)*1000,
                            'generation_mode':payload.get('generation_mode'),
                            'degraded':payload.get('degraded_mode'), 'refused':payload.get('refused')}
                except httpx.HTTPError as exc:
                    return {'status':type(exc).__name__, 'latency_ms':(time.perf_counter()-start)*1000}
        start = time.perf_counter()
        rows = await asyncio.gather(*(one(i) for i in range(args.requests)))
        elapsed = time.perf_counter()-start
    latencies = [r['latency_ms'] for r in rows]
    result = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'scope':'actual_http_closed_loop',
              'url':args.url, 'requests':args.requests, 'client_concurrency_cap':args.concurrency,
              'use_cache':args.cache, 'server_health':health.json(), 'duration_seconds':elapsed,
              'achieved_rps':args.requests/elapsed,
              'status_counts':{str(status):sum(r['status']==status for r in rows) for status in set(r['status'] for r in rows)},
              'latency_ms':{f'p{p}':float(np.percentile(latencies,p)) for p in (50,95,99)},
              'quality_validated':False, 'real_cloud_model_validated':False,
              'notes':['Latency starts after acquiring a client slot; excludes client queue waiting',
                       'No production capacity or multi-node claim', 'No server CPU or memory telemetry collected'],
              'responses':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='responses'},indent=2))
    if any(r['status'] != 200 for r in rows):
        raise SystemExit(1)


if __name__=='__main__':
    asyncio.run(main())
