# 10,000-Concurrency Enterprise RAG Benchmark Report

## Executive Summary

- **Total Concurrent Requests**: `10,000`
- **Succeeded / Failed**: `10,000 / 0` (`100.00%` success rate)
- **Wall-Clock Duration**: `2.866 s`
- **Achieved Throughput**: `3,488.6 RPS`
- **Peak Backend Concurrency (Gate Cap=256)**: `256`
- **Cache + Singleflight Offload Rate**: `74.96%`

## Overall Latency Distribution (ms)

- **P50**: `0.017 ms` | **P95**: `2061.733 ms` | **P99**: `2462.802 ms` | **Max**: `2520.680 ms`

## Execution Tier Breakdown

| Tier | Count | Traffic Share | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l1_exact` | 3,500 | 35.00% | 0.007 | 0.009 | 0.019 | 0.007 |
| `l2_semantic` | 3,000 | 30.00% | 0.017 | 0.025 | 0.045 | 0.018 |
| `singleflight_coalesced` | 996 | 9.96% | 359.191 | 519.271 | 563.493 | 373.747 |
| `miss` | 2,504 | 25.04% | 1339.544 | 2449.089 | 2506.315 | 1377.693 |
