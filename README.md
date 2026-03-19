> **Author:** John Mitchell (@whmatrix)
> **Status:** ACTIVE
> **Audience:** ML Engineers / RAG Architects / Infrastructure Engineers
> **Environment:** Local GPU inference (A6000 48GB or equivalent)

# Agentic Retrieval System

Architectural inversion from linear preprocessing pipeline to agentic orchestration. The reasoning model is no longer the terminal endpoint receiving pre-selected chunks — it is the first stage, with retrieval as a callable tool.

---

## Problem Statement

Linear RAG pipelines hardcode every preprocessing stage upstream of the reasoning model. Intent classification, query expansion, NER extraction, dataset selection, embedding search, reranking, and credibility scoring all execute in a fixed order before the model ever sees the query. The model receives 15 pre-selected chunks and writes a summary.

This breaks at scale:
- The model cannot reject irrelevant chunks
- The model cannot re-query with a reformulated question
- The model cannot skip retrieval when the answer is already in its context
- The model cannot filter by date, domain, or authorship
- The model cannot choose which indexes to search
- Every query pays the full pipeline cost regardless of complexity

At 447 datasets and 606M+ vectors across 12 domains, a fixed pipeline cannot make intelligent retrieval decisions. The model must.

---

## Architecture: Before and After

### Before (Linear Pipeline)

```
query → DeBERTa (intent) → FLAN-T5 (expansion) → spaCy/GLiNER (NER)
      → meta-index (dataset selection) → FAISS (search)
      → BGE reranker → credibility classifier → stance tagger
      → R1 (receives 15 chunks, writes summary)
```

Every stage runs unconditionally. R1 is a terminal consumer with no control over what it receives.

### After (Agentic Orchestration)

```
query + context document (system prompt)
  → Orchestrator (Qwen3-Coder-30B-A3B)
     ├── thinks: "Do I already know this from context doc?"
     ├── thinks: "What indexes are relevant?"
     ├── CALLS: search_indexes(indexes, query, filters)
     ├── EVALUATES: "Are these chunks sufficient?"
     ├── CALLS: rerank(query, chunks, top_k)  [optional]
     ├── CALLS: classify_chunks(chunks, ["authorship"])  [optional]
     └── GENERATES: final answer from accumulated evidence
```

The orchestrator decides what to retrieve, whether to retrieve, and when to stop retrieving.

---

## Two-Model Architecture

### Why R1 Cannot Orchestrate

DeepSeek R1-Distill-Qwen-32B produces extended think traces — typically consuming 20% of the context window per iteration. In a stateless multi-turn loop where the full conversation is re-sent each iteration, this compounds: 4 iterations × 20% think overhead = context exhaustion before useful work completes.

R1 also lacks native tool-calling capability. Its outputs must be parsed heuristically, producing brittle integrations.

### Orchestrator: Qwen3-Coder-30B-A3B

- **MoE architecture:** 30B total parameters, 3.3B active per token
- **Native tool calling:** Structured `<tool_call>` output with reliable JSON arguments
- **Structured output:** Consistent formatting without prompt engineering workarounds
- **VRAM footprint:** ~32GB at Q8_0 quantization, fits alongside embedding and reranking models

### Reasoning Model: DeepSeek R1-Distill-Qwen-32B

Called once at the end when deep reasoning over curated evidence is required. Receives the orchestrator's accumulated chunks — already filtered, reranked, and classified — and produces the final analytical output.

---

## Tool Definitions

The orchestrator has access to 5 tools:

### `search_indexes`
Embeds the query via e5-large-v2, searches specified FAISS indexes, returns chunks with metadata.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query (prefixed with "query: " for asymmetric encoding) |
| `indexes` | list[string] | Index names to search (from canonical registry) |
| `top_k` | int | Results per index (default 10) |
| `date_filter` | dict | Optional date range filter on chunk metadata |

### `rerank`
Scores query-chunk pairs using BGE-reranker-v2-m3. Returns chunks sorted by relevance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Original query |
| `chunks` | list[dict] | Chunks to rerank |
| `top_k` | int | Number of top results to keep |

### `classify_chunks`
Annotates chunks with classifier outputs. Supports multiple classifiers per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunks` | list[dict] | Chunks to classify |
| `classifiers` | list[string] | Classifier names: `credibility`, `stance`, `authorship` |

### `list_indexes`
Queries the canonical dataset registry. Returns matching index names and metadata.

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain` | string | Filter by domain (e.g., "legal", "financial") |
| `keyword` | string | Filter by keyword in dataset name/description |

### `no_retrieval`
Signals that the answer can be derived from the context document alone. No retrieval is executed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reason` | string | Why retrieval is unnecessary |

---

## ReAct Loop Design

```
1. SYSTEM PROMPT: context document + tool definitions + output format
2. USER: the query
3. ORCHESTRATOR OUTPUT: either TOOL_CALL or FINAL_ANSWER
4. PYTHON DISPATCHER: parses output, executes tool, appends result
5. GOTO 3 (until FINAL_ANSWER or max 4 iterations)
```

### Guardrails

- **Max 4 tool calls** per query — prevents infinite loops
- **30 chunk budget** across all retrievals — bounds context window consumption
- **Context document authoritative** — when context document contradicts stale chunks, context document wins
- **Stateless multi-turn** — full conversation re-sent each iteration (no hidden state accumulation)

---

## Performance

| Query Type | Latency | Chunk Count | Description |
|-----------|---------|-------------|-------------|
| Grounded (chunks exist) | ~11s | 10-30 | Orchestrator finds relevant indexes, retrieves, answers |
| Fabricated (no chunks) | ~93s | 0 | Orchestrator searches, finds nothing relevant, reports absence |

The latency asymmetry is itself a diagnostic signal — fabricated queries take longer because the orchestrator exhausts its tool call budget searching before concluding no relevant evidence exists.

---

## Authorship Classifier

DeBERTa zero-shot classification with heuristic boosts and penalties for first-person content filtering. Addresses the 0–17% first-person purity problem in discourse corpus retrieval: when querying Reddit/forum data, chunks containing "I think..." or "my experience with..." dilute analytical results.

Wired as a classifier option in `classify_chunks`. The orchestrator calls it when the query targets third-party analysis rather than personal anecdotes.

---

## Router-First Refactor

### Problem Discovered

The agentic orchestrator described above was the intended architecture. In practice, the LLM ignored system prompt routing rules. Three failure modes emerged:

- **Misrouted personal queries:** Questions about the operator searched domain indexes (financial, legal, discourse) instead of personal conversation indexes containing exported chat history
- **Misrouted domain queries:** Domain questions searched personal indexes, diluting results with irrelevant personal context
- **Query contamination:** The LLM included the operator's name in search queries against personal indexes where every chunk already belongs to the operator — biasing embedding similarity toward identity/profile chunks instead of the actual topic
- **Classifier mismatch:** The DeBERTa authorship classifier, designed to filter forum data (job listings, promotional content), misclassified the operator's own exported conversations as "job-listing" and filtered them out

### Solution

Routing decisions moved from the LLM to a deterministic Python router (`router.py`). The LLM no longer decides what to search or how to formulate queries. The router handles:

1. **Personal vs domain classification** — Regex-based detection on operator name, personal pronouns, and known personal topics. Hard gate, not probabilistic.
2. **Index selection** — Keyword scoring against the canonical registry. Personal queries route exclusively to personal indexes. Domain queries route to semantically matched domain indexes.
3. **Query cleaning** — Operator name and personal pronouns stripped from search queries before embedding. Personal indexes contain only operator data; including the name biases toward identity chunks rather than topical content.
4. **Authorship filter exemption** — Personal indexes bypass the DeBERTa authorship classifier entirely. The classifier was trained on forum discourse patterns and cannot meaningfully classify exported conversation data.
5. **Follow-up awareness** — Router tracks the previous query's route type. If the last query was personal and the current query uses pronouns ("he/him/his") or definite references ("the loan," "that"), it routes as `personal_followup` to the same personal indexes.

### Revised Architecture

```
query
  → Python router (regex classification, index selection, query cleaning)
  → FAISS search (on router-selected indexes)
  → BGE reranker
  → Authorship filter (domain indexes only — personal indexes exempt)
  → LLM receives pre-retrieved, reranked, filtered evidence
  → LLM synthesizes answer
```

System prompt dropped from ~5.4K to ~1.5K tokens. Queries that previously failed (e.g., personal financial questions routing to domain indexes) now resolve correctly in 5–7 seconds.

**Design principle:** Mechanical decisions (routing, filtering, query formulation) belong in deterministic code, not in LLM inference. The LLM's strength is synthesis and reasoning over evidence, not routing decisions that can be solved with regex and keyword matching.

---

## BERTopic Summarization Path

### Problem

FAISS search retrieves the top-K most similar chunks to a query. This works for specific questions ("What was the loan amount?") but fails structurally for summarization questions ("What topics are discussed across the corpus?"). A summarization question requires seeing the entire corpus, not 10–30 chunks.

### Solution

Offline topic extraction using BERTopic. A batch job reads all chunks from targeted indexes, clusters them by topic using precomputed e5-large-v2 embeddings, and produces a structured topic map (JSON artifact). At query time, the router detects summarization-style questions ("what topics," "what themes," "summarize conversations") and loads the precomputed topic map instead of searching FAISS. The LLM receives the full topic structure and synthesizes from that.

### Implementation

- BERTopic with precomputed embeddings (no redundant re-embedding)
- `min_topic_size=50` to suppress noise clusters
- Output: `topic_map.json` with labels, chunk counts, percentages, top keywords, and sample chunks per topic
- Initial deployment: 79,337 chunks across personal indexes → 277 topic clusters
- Extensible to any index — same script, different target

### Two Retrieval Paths

```
Specific question → Router → FAISS search → Rerank → Filter → LLM synthesis
Summarization question → Router → Precomputed topic map → LLM synthesis
```

---

## Anti-Hallucination Placement

Anti-hallucination rules (instructions prohibiting the LLM from fabricating details not present in retrieved chunks) were moved from the system prompt to a position immediately adjacent to the evidence block in the synthesis prompt. When placed in the system prompt, the rules were ~4K tokens away from the evidence they governed. When placed adjacent to the evidence, the LLM's attention mechanism keeps them in close proximity during generation.

This eliminated fabrication of specific details (dollar amounts, dates, actions) not present in source chunks. The general principle: any instruction governing how evidence should be used must be placed next to the evidence, not at the top of the prompt.

---

## Infrastructure Requirements

| Component | Specification |
|-----------|--------------|
| Machine | Dell Precision 5810 |
| CPU | Xeon E5-2699 v4 (22 cores / 44 threads) |
| GPU | NVIDIA RTX A6000 48GB |
| RAM | 128GB DDR4 |
| Storage | 40TB (multi-drive) |
| OS | Pop!_OS (Ubuntu-based) |

### VRAM Budget

| Model | VRAM |
|-------|------|
| Qwen3-Coder-30B-A3B (Q4_K_XL) | ~17GB |
| e5-large-v2 (embedding) | ~1.3GB |
| BGE-reranker-v2-m3 | ~1.3GB |
| **Total inference** | **~20GB** |
| Remaining for GPU FAISS search | ~28GB |

---

## What This Replaces

The linear pipeline from the `multi-domain-indexing-registry` era. Individual stages are preserved as callable tools:

| Component | Before | After |
|-----------|--------|-------|
| DeBERTa intent classification | Fixed pipeline stage | Available via `classify_chunks` |
| FLAN-T5 query expansion | Fixed pipeline stage | **Retired** — orchestrator reformulates natively |
| spaCy/GLiNER NER | Fixed pipeline stage | Available via `classify_chunks` |
| BGE reranker | Fixed pipeline stage | Available via `rerank` |
| Dataset selection | Meta-index lookup | Orchestrator calls `list_indexes` + `search_indexes` |
| R1 reasoning | Terminal consumer | Called once after orchestration completes |

---

## Related Repositories

- [universal-protocol-v4.23](https://github.com/whmatrix/universal-protocol-v4.23) — Deliverable spec and audit standards
- [multi-domain-indexing-registry](https://github.com/whmatrix/multi-domain-indexing-registry) — Pipeline validation at scale
- [semantic-indexing-batch-02](https://github.com/whmatrix/semantic-indexing-batch-02) — Production indexing (8.35M vectors)
- [corpus-scale-operations](https://github.com/whmatrix/corpus-scale-operations) — Infrastructure operations log
