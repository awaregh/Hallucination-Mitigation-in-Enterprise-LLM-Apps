# Enterprise Recommendations: Hallucination Mitigation Strategy Selection

---

## Enterprise Adoption Recommendations

Based on our systematic evaluation, we provide the following tiered recommendations for enterprise LLM deployments.

### Tier 1: Minimum Viable Mitigation (Low Risk Applications)

**Recommended for:** Internal tools, draft generation, brainstorming assistants

**Strategy:** Naive RAG + Grounded Prompting

- Implement naive RAG with a well-maintained knowledge base
- Use the `GROUNDED_TEMPLATE` prompt with citation requirements
- Enable basic `CitationRequiredConstraint`
- Expected hallucination rate: 20–30%
- Setup time: 1–2 engineer-weeks

### Tier 2: Production Baseline (Medium Risk Applications)

**Recommended for:** Customer-facing knowledge bases, product documentation Q&A, employee FAQ systems

**Strategy:** Hybrid RAG + Citation Enforcement + Schema Validation

- Implement hybrid BM25 + dense retrieval
- Enforce citation verification against retrieved chunks
- Validate response structure via `StructuredResponseValidator`
- Enable `DenyListConstraint` for domain-inappropriate terms
- Expected hallucination rate: 15–22%
- Setup time: 3–5 engineer-weeks

### Tier 3: High-Assurance (High Risk Applications)

**Recommended for:** Legal research assistants, medical information systems, financial advisory tools, compliance engines

**Strategy:** Full Pipeline — Hybrid RAG + Reranking + Citation Enforcement + Self-Critique + System Roles

- All Tier 2 components plus:
- Cross-encoder reranking for precision retrieval
- Self-critique validation loop (1–2 rounds)
- Role enforcement via `RoleEnforcer` with domain-specific persona
- Conservative prompt template
- Human review for responses below confidence threshold
- Expected hallucination rate: 8–15%
- Setup time: 8–12 engineer-weeks

---

## Strategy Selection Guide

### Decision Tree

```
Is the application customer-facing?
├── NO → Is internal accuracy critical?
│   ├── NO → Use: Baseline + Grounded Prompting
│   └── YES → Use: Naive RAG + Citation Check
│
└── YES → Is the domain high-stakes? (legal/medical/financial)
    ├── NO → Use: Hybrid RAG + Citation Enforcement
    └── YES → Can users tolerate > 2s latency?
        ├── NO → Use: Hybrid RAG + Reranking + Validation
        └── YES → Use: Full Pipeline + Human Review
```

### Strategy Selection Matrix

| Criteria | Naive RAG | Hybrid RAG | Reranking | Multi-Query | Full Pipeline |
|---|---|---|---|---|---|
| Low latency (<1.5s) | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| Low cost | ✅ | ✅ | ⚠️ | ⚠️ | ❌ |
| Simple deployment | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ |
| High accuracy needs | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Adversarial robustness | ❌ | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Long context fidelity | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Citation enforcement | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| Domain-specific roles | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ |

Legend: ✅ = Strong fit, ⚠️ = Acceptable, ❌ = Not recommended

---

## When to Use Each Mitigation Strategy

### Grounded Prompting Only
**Use when:** Rapid prototyping, non-critical internal tools, knowledge base is small and well-curated  
**Avoid when:** High-stakes decisions, adversarial users, large or noisy knowledge bases

### Naive RAG
**Use when:** Knowledge base is well-structured, queries are factual and specific, latency is critical  
**Avoid when:** Vocabulary mismatch between queries and docs, multi-hop reasoning required

### Hybrid RAG (BM25 + Dense)
**Use when:** Mixed query types (keyword and semantic), diverse user vocabulary, production deployments  
**Avoid when:** Real-time requirements (<500ms), extremely cost-sensitive applications

### Reranking RAG
**Use when:** Precision is more important than recall, context window is limited, quality is premium  
**Avoid when:** Latency budget is tight, compute resources are constrained

### Multi-Query RAG
**Use when:** Users phrase queries inconsistently, vocabulary mismatch is a known problem, high recall needed  
**Avoid when:** Cost is a constraint (3x LLM calls), response time < 2s required

### Self-Critique Validation
**Use when:** High accuracy is required and latency budget permits, domain is high-stakes  
**Avoid when:** Latency-sensitive applications, cost must be minimized

### Citation Enforcement
**Use when:** Auditability is required, regulatory compliance, users need to verify claims  
**Almost always recommended** as it adds minimal latency and significantly improves trust

### System Roles
**Use when:** Different user types need different behavior, domain specialization required  
**Use with:** Grounded template + deny/allow lists for the role's domain

---

## Implementation Checklist

### Phase 1: Foundation (Week 1–2)

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Configure `SystemConfig` for your deployment environment
- [ ] Set up document ingestion pipeline with `IngestionPipeline`
- [ ] Choose chunking strategy: `FixedSizeChunker` (default) or `SentenceChunker` for prose
- [ ] Initialize vector store: `InMemoryVectorStore` (dev), `ChromaVectorStore` (prod)
- [ ] Implement `MockEmbedder` tests → replace with `OpenAIEmbedder` or `SentenceTransformerEmbedder`
- [ ] Run baseline evaluation: `pytest tests/ -v`

### Phase 2: Retrieval (Week 2–4)

- [ ] Index knowledge base documents with chosen chunking strategy
- [ ] Evaluate retrieval quality: manual inspection of top-K results for 20 sample queries
- [ ] Choose retrieval strategy: start with `NaiveRAGRetriever`, upgrade to `HybridSearchRetriever`
- [ ] Tune `chunk_size`, `overlap`, and `top_k` for your corpus
- [ ] Set up evaluation harness with your domain-specific test queries

### Phase 3: Guardrails (Week 3–5)

- [ ] Select appropriate `PromptTemplate` for your use case
- [ ] Configure `GuardrailConfig.deny_list` with domain-inappropriate terms
- [ ] Configure `GuardrailConfig.allow_list` for domain-restricted deployments
- [ ] Apply `RoleEnforcer` if deploying multiple personas
- [ ] Test guardrails against `adversarial_prompts.jsonl`

### Phase 4: Validation (Week 4–6)

- [ ] Enable `CitationRequiredConstraint` and `CitationVerifier`
- [ ] Add `StructuredResponseValidator` for structured output use cases
- [ ] Evaluate whether self-critique is worth the latency for your SLA
- [ ] Configure `ValidationPipeline` with appropriate step weights
- [ ] Define acceptable `validation_pass_rate` threshold for your domain

### Phase 5: Production Hardening (Week 6–8)

- [ ] Add end-to-end observability: log `latency_ms`, `tokens_used`, `citation_rate` per query
- [ ] Set up hallucination monitoring dashboard with alerting
- [ ] Define and document escalation paths for low-confidence responses
- [ ] Conduct adversarial red-teaming with domain experts
- [ ] Establish a human-in-the-loop review process for high-stakes outputs
- [ ] Document the knowledge base refresh and re-indexing process

---

## Production Readiness Criteria

A deployment is considered production-ready when it meets **all** of the following criteria:

### Accuracy Criteria
- [ ] Hallucination rate < 20% on domain-specific factual test set (minimum 100 queries)
- [ ] Citation accuracy > 70% (citations verified against retrieved chunks)
- [ ] False refusal rate < 10% on legitimate factual queries
- [ ] Appropriate refusal rate > 80% on adversarial and missing-knowledge queries

### Performance Criteria
- [ ] P95 latency within acceptable SLA for the application (typically <3s for interactive use)
- [ ] System handles expected query concurrency without degradation
- [ ] Token costs within approved budget per query

### Operational Criteria
- [ ] All queries and responses logged with structured metadata
- [ ] Hallucination rate monitored in real-time with alerting thresholds defined
- [ ] Knowledge base has a defined refresh cadence with re-indexing procedures
- [ ] Runbook exists for common failure modes
- [ ] Rollback procedure tested and documented

### Security Criteria
- [ ] Prompt injection resistance validated against adversarial test set
- [ ] PII detection and redaction in place for sensitive deployments
- [ ] API key rotation procedures documented
- [ ] Rate limiting and abuse detection configured

---

## Monitoring and Observability Recommendations

### Metrics to Track (Per Query)

```python
{
    "query_id": str,
    "strategy": str,
    "latency_ms": float,
    "tokens_used": int,
    "citation_rate": float,
    "validation_passed": bool,
    "hallucination_detected": bool,  # from heuristic or classifier
    "refusal_triggered": bool,
    "confidence": float,
    "retrieval_chunks_used": int,
    "timestamp": str,
}
```

### Alert Thresholds (Recommended)

| Metric | Warning Threshold | Critical Threshold |
|---|---|---|
| Hallucination rate (rolling 1h) | > 25% | > 40% |
| P95 latency | > 3,000ms | > 6,000ms |
| Citation accuracy | < 60% | < 40% |
| Validation pass rate | < 70% | < 50% |
| Error rate (system errors) | > 2% | > 5% |

### Dashboards to Build

1. **Real-time quality dashboard**: Hallucination rate, citation accuracy, validation pass rate (rolling 1h, 24h, 7d)
2. **Latency breakdown dashboard**: P50/P95/P99 per component (retrieval, generation, validation)
3. **Cost tracking dashboard**: Token usage and estimated cost per day/week by strategy
4. **Failure mode dashboard**: Categorized failures, worst-performing query types

---

## Cost Optimization Strategies

### Reduce Token Usage

1. **Compress retrieved context:** Use `ContextCompressionRetriever` to filter irrelevant chunks before LLM call.
2. **Tune chunk size:** Smaller chunks (256–384 chars) reduce context tokens; larger chunks (512–768) improve coherence.
3. **Limit top-K:** Start with `top_k=3` and increase only if quality requires it.
4. **Cache frequent queries:** Implement semantic similarity caching for repeated queries (>0.95 cosine similarity).

### Reduce LLM Calls

1. **Disable self-critique by default:** Only enable for queries above a complexity threshold or below a confidence threshold.
2. **Use smaller models for query generation:** In multi-query RAG, generate variants with a smaller model (GPT-4o-mini vs GPT-4o).
3. **Batch embedding calls:** Use `embed_batch` instead of per-chunk embedding for indexing.
4. **Model routing:** Route simple queries to cheaper models, complex queries to more capable models.

### Infrastructure Optimization

1. **Cache embeddings:** Cache document embeddings and only recompute on knowledge base updates.
2. **Persistent vector store:** Use `ChromaVectorStore` with persistence to avoid re-indexing on restart.
3. **Pre-warm retriever:** Load BM25 index at startup, not on first query.
4. **Async generation:** For batch evaluation, use async LLM clients to parallelize requests.

---

## Risk Assessment Framework

### Risk Scoring Matrix

For each planned deployment, score the following dimensions (1 = Low Risk, 5 = High Risk):

| Dimension | Questions to Answer | Score (1–5) |
|---|---|---|
| **Consequence severity** | What is the worst-case impact of a hallucination? | |
| **Query volume** | How many hallucinations occur at expected query volume? | |
| **User sophistication** | Will users detect hallucinations? | |
| **Domain sensitivity** | Is domain legal/medical/financial/compliance-related? | |
| **Reversibility** | Can hallucination damage be undone? | |

**Total Score Interpretation:**
- **5–10:** Low risk — Tier 1 mitigation sufficient
- **11–16:** Medium risk — Tier 2 mitigation required
- **17–22:** High risk — Tier 3 mitigation + human review required
- **23–25:** Critical risk — Consider whether LLM automation is appropriate at all

### Red Lines (Do Not Deploy Without These)

The following non-negotiable requirements apply to all enterprise deployments regardless of risk score:

1. **No medical diagnosis or treatment recommendations** without licensed practitioner review
2. **No legal advice or contract interpretation** without attorney review
3. **No financial advice** (investment recommendations, tax guidance) without registered advisor review
4. **No PII generation or inference** without privacy review
5. **No security configurations** (firewall rules, access controls) generated without security review

---

## Contact and Support

For questions about implementing these recommendations, consult:
- Internal ML Platform team for infrastructure questions
- Legal/Compliance team for domain-specific risk assessment
- Security team for adversarial robustness requirements
