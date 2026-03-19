# Performance

## Query Latency

| Query Type | Median Latency | Tool Calls | Chunks Retrieved | Description |
|-----------|---------------|------------|-----------------|-------------|
| Grounded (evidence exists) | ~11s | 1–2 | 10–30 | Orchestrator finds relevant indexes, retrieves, answers |
| Fabricated (no evidence) | ~93s | 3–4 | 0 | Orchestrator exhausts tool budget searching before reporting absence |
| Context-only (no retrieval) | ~3s | 0–1 | 0 | Answer derived from context document, `no_retrieval` called |

### Latency Asymmetry

Fabricated queries (where no relevant indexed data exists) take ~8x longer than grounded queries. This is expected: the orchestrator searches multiple index sets, reformulates queries, and tries alternative approaches before concluding no evidence exists. The latency pattern is itself a diagnostic signal — abnormally long query times indicate the system is working correctly by being thorough rather than hallucinating.

---

## V3 Query Results (Mar 14 2026)

Six-query benchmark run on the agentic retrieval system:

| # | Query | Time | Chunks | Tool Calls | Outcome |
|---|-------|------|--------|------------|---------|
| 1 | Grounded domain query | 9.2s | 15 | 1 | Direct hit, single retrieval sufficient |
| 2 | Cross-domain query | 14.7s | 28 | 2 | Two index sets searched, reranked |
| 3 | Context-document-only | 2.8s | 0 | 0 | `no_retrieval` — answer from context |
| 4 | Fabricated entity query | 87.4s | 0 | 4 | Max tool calls, correctly reported absence |
| 5 | Authorship-filtered query | 16.3s | 22 | 3 | Search + classify_chunks(authorship) + rerank |
| 6 | Temporal-filtered query | 11.1s | 12 | 2 | Date-filtered search + rerank |

---

## Chunk Retrieval Budget

| Parameter | Value |
|-----------|-------|
| Max chunks per `search_indexes` call | 30 (configurable via top_k) |
| Max chunks across all retrievals | 30 (server-enforced ceiling) |
| Max tool calls per query | 4 |
| Typical chunks for grounded query | 10–15 |
| Typical chunks for cross-domain query | 20–28 |

The 30-chunk ceiling prevents context window exhaustion. At ~200 tokens per chunk average, 30 chunks consume ~6,000 tokens — well within the orchestrator's 32K context budget.

---

## Infrastructure Bottleneck Analysis

### CPU FAISS Search (Current)

FAISS IndexFlatIP brute-force search on CPU:

| Shard Size | Search Time | Notes |
|-----------|-------------|-------|
| 500K vectors | ~1.5s | Small indexes, acceptable |
| 2M vectors | ~8s | Noticeable but usable |
| 4M vectors | ~14s | Dominant latency contributor |
| 8M vectors | ~27s | Unacceptable for interactive use |

The SEC filings corpus (4 shards × 4M vectors at float32 = 16GB per shard) was the worst case at ~27 seconds per shard on CPU.

### GPU FAISS Search (Design)

FAISS `index_cpu_to_gpu()` with 4GB temporary memory cap:

| Shard Size | Estimated Time | Notes |
|-----------|---------------|-------|
| 500K vectors | <0.2s | Near-instantaneous |
| 2M vectors | <0.5s | Negligible |
| 4M vectors | <1s | Acceptable |
| 8M vectors | <2s | Within latency budget |

Falls back to CPU if shard exceeds temporary GPU memory allocation. VRAM budget: 48GB total - 35GB inference models = 13GB available, with 4GB allocated to FAISS temp memory and headroom for GPU memory fragmentation.

---

## Scaling Characteristics

| Corpus Size | Expected Grounded Latency | Bottleneck |
|-------------|--------------------------|------------|
| <100M vectors | <10s | Orchestrator reasoning |
| 100M–500M vectors | 10–15s | FAISS search (CPU) |
| 500M–1B vectors | 15–30s | FAISS search (CPU), mitigated by sharding |
| 500M–1B vectors (GPU) | <10s | Orchestrator reasoning |

At current scale (606M vectors, 447 datasets), the bottleneck is CPU-based FAISS search on large shards. GPU acceleration reduces this to sub-second, making orchestrator reasoning time the dominant factor.

---

## Optimization Path: 316s → 29s

Three sequential optimizations produced an 11x cumulative latency reduction:

| Optimization | Before | After | Reduction |
|---|---|---|---|
| System prompt trimming (8.1K → 5.4K tokens) | 316s | ~180s | ~43% |
| Context window reduction (32K → 16K) | ~180s | ~60s | ~67% |
| Quantization change (Q8_0 → Q4_K_XL) | ~60s | 29s | ~52% |
| **Cumulative** | **316s** | **29s** | **91% (11x)** |

**System prompt trimming:** Removed redundant tool descriptions and verbose routing instructions. The LLM performed identically with compact tool specs.

**Context window reduction:** The orchestrator's actual context usage (system prompt + query + tool results + reasoning) rarely exceeded 8K tokens. The 32K allocation reserved memory for hypothetical deep conversations that never materialized. Halving the context window freed KV cache memory and reduced per-token inference overhead.

**Quantization change:** Qwen3-Coder-30B-A3B moved from Q8_0 (~32GB VRAM) to Q4_K_XL (~17GB VRAM). Retrieval routing quality — the primary function — showed no degradation. The freed ~15GB VRAM provides headroom for concurrent GPU FAISS search and future model co-loading.

---

## Router-First Refactor: Latency Impact

Moving routing decisions from LLM inference to deterministic Python code eliminated an entire class of query failures and reduced latency for previously-broken query types:

| Query Type | Before Router Refactor | After Router Refactor |
|---|---|---|
| Personal query (e.g., financial history) | FAIL (routed to domain indexes) | 5–7s correct |
| Context-doc query (no retrieval needed) | 3–6s | 3–6s (unchanged) |
| Full domain retrieval | 16–35s | ~20s average |
| Summarization (BERTopic path) | N/A (not supported) | 6–7s |

System prompt reduction from ~5.4K to ~1.5K tokens contributed additional per-query savings by reducing prompt processing overhead on every request.
