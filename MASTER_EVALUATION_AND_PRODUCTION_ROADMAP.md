# AURA – Master Evaluation & Production Roadmap

**Version:** 6.0.0  
**Repository:** [github.com/SuhailSha/cloudrag-agent-chatbot](https://github.com/SuhailSha/cloudrag-agent-chatbot)  
**Verification Status:** Part 1 (`C1–C7`, `H1–H11`, `M1–M8`), Part 2 (`N1–N12`), and Part 3 (`V1–V6`) Remediated & Tested via `pytest tests/test_acceptance_suite.py -q` (16/16 Passed)

---

## 1. Summary of Audit Rounds & Current State

| Audit Round | Scope | Key Remediations |
| :--- | :--- | :--- |
| **Part 1 (`C1–C7`, `H1–H11`, `M1–M8`)** | Initial 26-item review | Raw dense/sparse relevance gate, deterministic intent router, citation & number critic, mandatory OAuth `state` + `HttpOnly` cookie, fail-closed `JWT_SECRET` + DB token revocation, truthful sub-processor disclosure, word-counted table & row-aware CSV chunking, consent-gated pending memories, `DatabaseUnavailableError` -> 503. |
| **Part 2 (`N1–N12`)** | Re-evaluation of v2 | Rightmost `X-Forwarded-For` hop + expired bucket pruning (`N1`), lightweight text-only `get_all_user_chunks` for BM25 (`N2`), startup schema & constraint migration for existing SQLite/Postgres DBs (`N3`), `DatabaseIntegrityError` -> 409 (`N4`), 9-table atomic account deletion (`N5`), partial split-brain DB embedding merger (`N6`), 30+ chunk `eval_golden_set.json` + domain near-miss tests (`N7`), cookie revocation on account deletion (`N8`), single-doc comparison routing & utility bill classification (`N9`), tiered quotas & metered comparisons (`N10`), strict `.csv` detection (`N11`), `source_doc_id` memory linkage & 2-model x 8s LLM timeout (`N12`). |
| **Part 3 (`V1–V6`)** | Re-evaluation of v3 | Concept synonym expansion (`CONCEPT_SYNONYM_GROUPS`) so paraphrased questions pass at `d_score = 0.45` and via BM25 while near-misses abstain (`V1`), strict derived-number verification requiring explicit calculation cues + same-chunk operands and phrase-level swapped-attribute detection (`V2`), Postgres migration ordering deleting duplicate docs before orphan chunks (`V3`), Docker-aligned `render.yaml` with secret env declarations (`V4`), parameterized PostgreSQL `LIKE` queries, `.github/workflows/ci.yml`, and production-sized (`chunk_size=180`) 32+ chunk evaluation corpus (`V6`). |

---

## 2. Reproducible Test Execution

Run the automated 16-gate test suite locally or in GitHub Actions CI:

```bash
pytest tests/test_acceptance_suite.py -q
python3 tests/test_acceptance_suite.py
```

### Verified Test Output
```text
16 passed in 0.48s
```

---

## 3. Future Roadmap (Post-Launch Enhancements)
- **Authorization-Code Flow with PKCE & Cookie-Only Auth**: Transition from OAuth implicit `id_token` callback + `localStorage` fallback to authorization-code flow with PKCE and cookie-only sessions.
- **Compiled Tailwind & Nonce-Based CSP**: Pre-compile Tailwind CSS at build time to remove `'unsafe-inline'` and external CDN script dependencies.
- **Native `pgvector` + PostgreSQL `tsvector` FTS**: Consolidate dense vectors and sparse BM25 into Neon PostgreSQL `pgvector` and `GIN` full-text indexes.
- **Cross-Encoder Reranker & OCR Worker Queue**: Add background worker queue (`HTTP 202`) with OCR fallback for scanned image PDFs and a cross-encoder reranker stage.
