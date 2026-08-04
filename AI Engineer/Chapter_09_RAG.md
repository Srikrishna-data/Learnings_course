# Chapter 9 — Retrieval-Augmented Generation: Grounding AI in Trusted Knowledge

> **Part 9 — RAG**  
> **Target time:** 12–16 hours  
> **Outcome:** You can build a citation-first RAG system, choose an indexing strategy and vector store, tune retrieval quality, enforce access control, and evaluate the pipeline separately from generation.

## 1. What RAG is and why it exists

**Retrieval-augmented generation (RAG)** retrieves relevant, authorized information at request time and includes that evidence in a model prompt. It addresses a fundamental LLM limitation: a model’s parameters are not a current, permission-aware, auditable company knowledge base.

```text
                 INGESTION
documents ─► parse ─► clean ─► chunk + metadata ─► embed ─► indexes
                                                              │
                 QUESTION PATH                                │
user question ─► auth ─► retrieve + filter + rerank ──────────┤
                                                              ▼
                                              LLM answer + source citations
```

### Problems RAG solves

- Private facts: employee handbooks, support articles, contracts, tickets.
- Current facts: policies that change after model training.
- Auditable answers: show the excerpt and source version supporting a claim.
- Smaller prompts: retrieve a few relevant passages instead of sending a whole archive.

### What RAG does not solve

RAG cannot make bad source material correct, authorize data access, or guarantee a model faithfully uses retrieved evidence. It cannot compensate for missing documents, weak retrieval, or an ambiguous question. Design an abstention response when evidence is insufficient.

> **RAG is a search-and-evidence system first, and an LLM feature second.** Measure retrieval quality independently before debugging generation.

---

## 2. The RAG pipeline in detail

### Ingestion

Ingestion transforms source material into searchable records. It should be asynchronous, repeatable, and traceable.

```text
source connector ─► file/type safety ─► parse OCR/layout ─► normalize ─► quality checks
                                                                         │
                                     version + ACL + source URL ◄───────┤
                                                                         ▼
                                                        chunk / hash / embed / index
```

**Best practices:**

- Preserve canonical source ID, document version, owner, tenant, access-control metadata, timestamps, language, and source URL.
- Hash raw content and chunks so unchanged material is not re-embedded.
- Mark deleted/expired documents as tombstoned and remove them from retrieval.
- Keep original artifacts in governed object storage; indexes are derived data.
- Separate parsing failures from empty/low-quality document content.

### Query path

At query time, authenticate first; filter results by tenant and permission before the model sees text; retrieve candidates; rerank; build an evidence-bounded prompt; validate/cite the response.

```text
question + identity ─► query rewrite? ─► hybrid search with ACL filter ─► top candidates
                                                                          │
                                                               cross-encoder reranker
                                                                          │
                                                                prompt evidence bundle
                                                                          │
                                                                  answer or abstain
```

### Real-world example

An HR assistant answers “Can a remote employee work temporarily from Canada?” It retrieves the travel-work policy, filters it to the employee’s country/role entitlement, reranks passages mentioning Canada and remote work, cites the exact policy revision, and says it cannot answer if the policy is absent or contradictory.

---

## 3. Embeddings and vector search

An **embedding** is a learned numeric vector representing text, images, or other data. Similar content tends to have nearby vectors. A vector index retrieves nearby vectors efficiently.

```text
"return policy" ─► embedding model ─► [0.18, -0.04, ...] ─► vector index
"Can I send it back?" ─► same model ─► query vector ───────► nearest chunks
```

### Similarity measures

| Measure | Meaning | Typical note |
|---|---|---|
| Cosine similarity | compares vector direction | common for normalized embeddings |
| Dot product | weighted directional match | often equivalent after normalization |
| Euclidean distance | geometric straight-line distance | used by some indexes/models |

The score is a ranking signal, not a probability that an answer is correct. Its value may vary by model, language, query type, and index configuration.

### Embedding model choice

Evaluate candidate embedding models on your retrieval set. Consider language coverage, domain language, dimensions, speed, cost, privacy/deployment needs, and support for query/document prefixes where applicable. Keep query and document embeddings compatible: changing models typically requires a full re-index.

### Production code: content-addressed chunking

```python
from hashlib import sha256
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict

def make_chunk(document_id: str, text: str, metadata: dict) -> Chunk:
    stable = f"{document_id}:{metadata.get('version')}:{text}".encode("utf-8")
    return Chunk(sha256(stable).hexdigest(), document_id, text, metadata)
```

Content hashes make incremental embeddings and deletion workflows much easier to reason about.

---

## 4. Chunking: the highest-leverage retrieval decision

### What chunking is

Chunking splits a document into retrieval units. A chunk should contain enough context to answer a meaningful question but not so much unrelated material that it dilutes retrieval or wastes context tokens.

```text
Policy document
  ├─ Eligibility and definitions
  ├─ Return window and exceptions
  ├─ International returns
  └─ Refund processing timeline

Each section/subsection becomes one or more labeled chunks with source metadata.
```

### Chunk size and overlap

There is no universal token count. Chunk size depends on document structure, query style, embedding model, and answer granularity.

| Strategy | Advantage | Limitation | Good starting use |
|---|---|---|---|
| Fixed-size with overlap | easy implementation | cuts sentences/sections; duplicate context | baseline only |
| Paragraph/heading based | preserves meaning | variable length | handbooks, Markdown, HTML |
| Parent-child chunks | precise retrieval + broader context | more index complexity | long manuals, technical docs |
| Semantic chunking | can align topic changes | extra processing and variability | heterogeneous prose |

**Overlap** helps avoid losing a boundary-spanning fact, but too much overlap creates duplicates, index cost, and repeated prompt text. Start with structure-aware chunks, then tune on a labeled retrieval set.

### Metadata

Metadata lets a search system filter and cite chunks. Required fields usually include:

```json
{
  "tenant_id": "tenant_42",
  "document_id": "hr-remote-work",
  "version": "2026-07-01",
  "section": "International work",
  "source_url": "https://intranet/policies/remote-work",
  "access_groups": ["employees-us"],
  "updated_at": "2026-07-01T00:00:00Z",
  "content_hash": "..."
}
```

> **Security callout:** Apply tenant and access filters before retrieval results reach the model. Prompt wording such as “only use authorized documents” is not access control.

---

## 5. Hybrid search and reranking

### Why vector search alone is not enough

Semantic search is good at meaning but can miss exact terms: product codes, policy numbers, error messages, names, dates, and uncommon legal phrases. Keyword search is good at exact matching but can miss paraphrases.

**Hybrid search** combines lexical (for example BM25) and vector retrieval.

```text
query ─► keyword/BM25 retrieval ─┐
                                 ├─► fuse candidates ─► rerank ─► top evidence
query ─► vector retrieval ───────┘
```

### Reranking

A reranker examines the query and each candidate together, usually more precisely than a fast embedding search. It is used after broad retrieval on a smaller candidate set.

| Stage | Goal | Typical trade-off |
|---|---|---|
| Candidate retrieval | high recall; do not miss evidence | fast but noisier |
| Metadata/ACL filter | eliminate forbidden/irrelevant scope | must be correct before model use |
| Reranking | high precision at top positions | extra latency/cost |
| Context assembly | reduce duplication and include provenance | may omit needed neighboring context |

### Query rewriting and decomposition

For difficult questions, a small model or deterministic rules can normalize spelling, expand abbreviations, or split a multi-part request. Rewriting can also change intent or create a security issue, so preserve the original query and evaluate it. Do not add “agentic” rewriting by default.

### Citation design

Citations must map to actual retrieved source IDs, URLs, sections, and versions. A model should select from provided citations only; application code should validate that cited IDs belong to the evidence bundle.

```python
def validate_citations(cited_ids: list[str], retrieved_ids: set[str]) -> bool:
    return set(cited_ids).issubset(retrieved_ids)
```

Valid citations do not prove that the cited passage supports the claim. Add groundedness evaluation and human review for important domains.

---

## 6. Vector database options

There is no universal best vector database. Selection depends on index size, filter complexity, latency, cloud posture, operations capacity, and existing data systems.

| Option | Strength | Consideration | Common fit |
|---|---|---|---|
| PostgreSQL + pgvector | relational metadata/ACLs and vectors together | not always ideal for massive vector scale | strong first production choice |
| Pinecone | managed vector service | provider dependency/cost model | teams prioritizing managed operations |
| Weaviate | vector-native platform and ecosystem | operate or use managed service | rich vector/search workloads |
| Qdrant | filtering-oriented vector platform | evaluate operations/hosting | metadata-heavy retrieval |
| Milvus | large-scale vector-search ecosystem | more platform complexity | high-scale/self-managed needs |
| Chroma | easy local development/prototyping | assess operational maturity for production | learning and prototypes |

### Evaluation criteria

- Can it enforce/filter on tenant and ACL metadata efficiently?
- Does it support the needed hybrid search and reranking integration?
- What are indexing/update/delete semantics and operational recovery paths?
- What is latency at realistic concurrent load and collection size?
- What does it cost including replicas, storage, backups, and engineering time?
- Can it meet encryption, network, residency, and audit requirements?

Avoid migrating storage systems before measuring a real bottleneck. For many early products, Postgres plus a simple vector extension and a good document pipeline is more valuable than a specialized index.

---

## 7. Evaluating RAG systems

### Separate retrieval from generation

If the correct passage was never retrieved, rewriting the answer prompt will not fix the core issue. Evaluate each stage.

```text
question + expected source
       │
       ├─ retrieval metrics: Did the source appear in top-k?
       ├─ reranking metrics: Did it reach a useful position?
       ├─ answer metrics: Is the response correct and grounded?
       └─ system metrics: Is it safe, fast, permission-aware, affordable?
```

### Core metrics

| Metric | Question answered |
|---|---|
| Recall@k | Did top-k include the expected evidence? |
| MRR / nDCG | How highly was relevant evidence ranked? |
| Citation precision | Are cited sources actually relevant? |
| Groundedness / faithfulness | Are claims supported by supplied evidence? |
| Answer correctness | Does the answer solve the user’s question? |
| Abstention quality | Does it safely decline when evidence is insufficient? |
| p95 latency / cost per answer | Can the system operate at required scale? |

### Evaluation dataset design

Include normal queries, synonyms, exact identifiers, multilingual queries, ambiguous questions, outdated documents, documents with conflicting policies, and requests the knowledge base cannot answer. Include ACL test cases that prove one tenant cannot retrieve another tenant’s content.

### LLM judges and human review

LLM judges can scale subjective quality checks but must be calibrated against human labels and should not be the only release signal. Deterministic retrieval/citation checks are cheaper and more reliable for what they can measure.

---

## 8. Production architecture, cost, and failure handling

```text
Sources ─► event/cron ingestion ─► object store + parser workers ─► index
                                                     │                │
                                                     └─ quality logs ─┘

User ─► API/auth ─► ACL-filtered hybrid retrieval ─► reranker ─► LLM
                   │                                      │          │
                cache                                trace/evals   citations
                   │                                      │          │
                   └───────────── monitoring / cost / feedback ─────┘
```

### Cost optimization

- Embed only changed chunks using content hashes.
- Choose chunk structure before increasing `top_k`.
- Retrieve few high-quality passages and deduplicate overlapping text.
- Cache permission-safe query results with index/model/version in the key.
- Run expensive reranking only on a limited candidate set.
- Use asynchronous ingestion and batch embeddings when latency does not matter.
- Track embedding, storage, retrieval, reranking, generation, and human-review costs separately.

### Failure behavior

| Failure | Safe behavior |
|---|---|
| Index unavailable | say knowledge lookup is temporarily unavailable; do not invent answer |
| No relevant evidence | abstain or request clarification |
| Stale source | show version/date; trigger index refresh workflow |
| ACL filter returns none | explain access limitation without revealing document existence |
| Conflicting passages | expose conflict and route to owner/human review |
| LLM timeout | return citations/candidate passages where appropriate, or retry safely |

---

## 9. Hands-on project — citation-first employee handbook bot

Build a RAG assistant for a fictional employee handbook.

### Requirements

1. Ingest Markdown/PDF documents with document/version/section/ACL metadata.
2. Use structural chunking; compare it with a fixed-size baseline.
3. Store vectors in pgvector, Chroma, Qdrant, or another selected option; document why.
4. Implement hybrid retrieval, tenant/access filtering, and optional reranking.
5. Return an answer, confidence/abstention state, and validated citations with source links.
6. Create 100 questions with expected source IDs, including out-of-scope, permission, and injection cases.
7. Report Recall@3, citation precision, groundedness, p95 latency, token usage, and cost/query.
8. Add incremental update and tombstone tests for changed/deleted policies.

### Suggested repository layout

```text
handbook-rag/
├── ingestion/
│   ├── parse.py
│   ├── chunk.py
│   └── index.py
├── retrieval/
│   ├── hybrid.py
│   ├── rerank.py
│   └── permissions.py
├── app/
│   └── api.py
├── evals/
│   ├── questions.jsonl
│   └── run.py
├── tests/
└── architecture.md
```

### Interview questions

1. Why does a large context window not eliminate RAG?
2. How would you choose chunk size and overlap?
3. What is the difference between Recall@k and groundedness?
4. Why apply ACL filters before generation?
5. When does hybrid retrieval beat vector-only retrieval?
6. How do you delete a document reliably from a RAG system?

---

## Summary

RAG grounds generative systems in current, private, auditable evidence. Its quality depends on ingestion, chunking, metadata, embedding compatibility, permission filters, hybrid retrieval, reranking, citation validation, and stage-specific evaluation. A vector database is only one component. The production standard is a citation-first system that can abstain, respects access controls before the model call, and tracks quality, latency, and cost as the corpus changes.

## Further reading

- [RAG paper](https://arxiv.org/abs/2005.11401)
- [FAISS](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Weaviate documentation](https://weaviate.io/developers/weaviate)

---

**Next chapter:** Part 10 — AI Infrastructure: GPUs, CUDA, inference, serving, vLLM, Ollama, TensorRT, Docker, Kubernetes, and FastAPI.
