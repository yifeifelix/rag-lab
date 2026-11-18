# 10,000 Coroutine Synthetic RAG Experiment

Scope: one Python process, no HTTP, local extractive answers, simulated LLM wait, prewarmed caches.
Completion is not answer correctness. Traffic proportions are constructed, not measured in production.

## Executive Summary

- **Total Concurrent Requests**: `10,000`
- **Succeeded / Failed**: `10,000 / 0` (`100.00%` success rate)
- **Wall-Clock Duration**: `2.750 s`
- **Achieved Throughput**: `3,636.7 RPS`
- **Peak Backend Concurrency (Gate Cap=256)**: `256`
- **Cache + Singleflight Offload Rate**: `74.86%`

## Overall Latency Distribution (ms)

- **P50**: `311.928 ms` | **P95**: `1960.058 ms` | **P99**: `2328.818 ms` | **Max**: `2380.917 ms`

## Execution Tier Breakdown

| Tier | Count | Traffic Share | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l1_exact` | 3,500 | 35.00% | 0.011 | 0.015 | 0.017 | 0.012 |
| `l2_semantic` | 1,156 | 11.56% | 0.020 | 0.025 | 0.029 | 0.020 |
| `singleflight_coalesced` | 2,830 | 28.30% | 400.265 | 565.352 | 610.530 | 413.710 |
| `miss` | 2,514 | 25.14% | 1306.859 | 2315.700 | 2367.871 | 1345.475 |
