# Architecture Specification

## System Prompt Structure

Each orchestrator invocation receives a system prompt with three sections:

### 1. Context Document
A structured document providing the orchestrator with domain knowledge relevant to the current session. The context document is authoritative — when it contradicts information in retrieved chunks, the context document wins. This handles stale data in indexes that have not been re-indexed since the context document was last updated.

### 2. Tool Specifications
JSON-schema definitions for all 5 tools (`search_indexes`, `rerank`, `classify_chunks`, `list_indexes`, `no_retrieval`). Each definition includes parameter types, required/optional flags, and expected return format.

### 3. Output Format
The orchestrator must produce output in exactly one of two formats per turn:

**Tool call:**
```
<think>
[Internal reasoning about what to do next]
</think>
<tool_call>
{"name": "search_indexes", "arguments": {"query": "...", "indexes": ["..."], "top_k": 10}}
</tool_call>
```

**Final answer:**
```
<think>
[Internal reasoning about accumulated evidence]
</think>
<final_answer>
[Structured response to the user's query]
</final_answer>
```

---

## Stateless Multi-Turn Mechanics

The orchestrator is stateless. Each iteration of the ReAct loop sends the full conversation history:

```
Turn 1: [system prompt] [user query]
Turn 2: [system prompt] [user query] [assistant: tool_call] [tool: result]
Turn 3: [system prompt] [user query] [assistant: tool_call] [tool: result]
         [assistant: tool_call_2] [tool: result_2]
Turn 4: [system prompt] [user query] [...all prior turns...] [assistant: final_answer]
```

### Context Window Budget Analysis

| Component | Tokens (approx) |
|-----------|-----------------|
| System prompt (context doc + tools + format) | 4,000–8,000 |
| User query | 50–200 |
| Per tool call (think + call + result) | 500–2,000 |
| Max 4 tool calls accumulated | 2,000–8,000 |
| Final answer generation budget | 2,000–4,000 |
| **Total per query** | **8,000–28,000** |

Qwen3-Coder-30B-A3B supports 32K context. The 30-chunk budget across all retrievals keeps accumulated tool results within the window even at maximum iteration depth.

### Why Not Streaming / Stateful

Stateful approaches (maintaining hidden state between turns) introduce:
- State corruption on error recovery
- Memory leaks on long-running sessions
- Non-reproducible behavior (same query, different state → different answer)

Stateless replay is more expensive per-turn but deterministic and debuggable.

---

## Component Demotion: FLAN-T5 and DeBERTa

### FLAN-T5 Query Expansion — Retired

FLAN-T5 was used in the linear pipeline to expand terse queries into richer search terms. Example: "SEC enforcement" → "SEC enforcement actions penalties violations settlements."

The orchestrator handles this natively. When initial search results are insufficient, it reformulates the query in its next tool call — with the advantage of seeing what the first query returned. FLAN-T5's blind expansion (no feedback from results) is strictly inferior.

### DeBERTa Intent Classification — Demoted to Tool

DeBERTa zero-shot classification was a mandatory first stage in the linear pipeline, classifying every query into intent categories before routing. In the agentic architecture, intent classification is implicit in the orchestrator's reasoning. DeBERTa remains available as a classifier option within `classify_chunks` for cases where explicit authorship or stance labels are needed on retrieved chunks.

---

## Stale Data Detection

When the context document contains information that is more recent than indexed data:

1. **Context document wins** — The orchestrator is instructed to prefer context document facts over contradicting chunks
2. **Temporal metadata** — Chunks carry indexing timestamps; the orchestrator can assess recency
3. **Explicit conflict reporting** — When the orchestrator detects a contradiction between context document and retrieved chunks, it notes the conflict in its final answer rather than silently choosing one

This is a pragmatic solution for a corpus where some indexes are re-built monthly and others are static archives (e.g., historical pushshift data from 2012–2018 that will never change).

---

## Dispatcher Implementation

The Python dispatcher between orchestrator turns:

```
1. Receive orchestrator output
2. Parse for <tool_call> or <final_answer> tags
3. If <final_answer>: return to user, loop ends
4. If <tool_call>:
   a. Validate tool name against registered tools
   b. Validate arguments against tool schema
   c. Execute tool function
   d. Format result as tool response message
   e. Append to conversation history
   f. Check iteration count (max 4)
   g. Re-send full conversation to orchestrator
5. If max iterations reached without <final_answer>:
   Force a final answer by appending "You have used all tool calls.
   Provide your final answer now." and sending one last time.
```

### Error Handling

- **Invalid tool name:** Return error message as tool result, orchestrator can self-correct
- **Schema validation failure:** Return error with expected schema, orchestrator can retry
- **Tool execution failure:** Return error description, orchestrator can try alternative approach
- **Parse failure:** If output contains neither `<tool_call>` nor `<final_answer>`, treat as malformed and request retry (counts against iteration budget)

---

## Index Discovery Flow

When the orchestrator does not know which indexes to search:

```
Orchestrator: "I need SEC filing data"
  → CALLS: list_indexes(domain="financial", keyword="SEC")
  → RECEIVES: [{"name": "sec_10k_2020_2023", "vectors": 4200000, "domain": "FINANCE"}, ...]
  → CALLS: search_indexes(indexes=["sec_10k_2020_2023"], query="enforcement actions", top_k=15)
  → EVALUATES results
  → CALLS: rerank(query="enforcement actions", chunks=[...], top_k=5)
  → GENERATES final answer
```

The canonical registry (447 datasets, 606M vectors) is too large to embed in the system prompt. `list_indexes` provides a filtered view.
