# Senior AI Engineering Review — Complete Remediation Tracker (Parts 1, 2 & 3)

This tracker records every finding from **Part 1 (`C1–C7`, `H1–H11`, `M1–M8`)**, **Part 2 (`N1–N12`)**, and **Part 3 (`V1–V6`)** of the Senior AI Engineering Review, the exact code changes implemented, and the automated test gates in `tests/test_acceptance_suite.py` that verify them.

---

## 1. Part 3 Findings (`V1 – V6`)

| ID | Finding | Code Remediation | Verification Gate |
| :--- | :--- | :--- | :--- |
| **V1** | Answerability gate refused paraphrased answerable questions at simulated similarity 0.45 | Added `CONCEPT_SYNONYM_GROUPS` in `retrieval_pipeline.py` and `RAGRetrievalAgent.passes_relevance_gate()` in `agent_swarm.py` (`renewal/end` <-> `expires`, `out of pocket` <-> `deductible`, `move out/get back` <-> `deposit/refundable`, `minimum stay` <-> `lock-in`, `breakdown help` <-> `roadside assistance`). All 8 paraphrased probes pass at `d_score = 0.45` and via BM25 while all 20 near-miss/trivia probes abstain. | `test_01`, `test_02` |
| **V2** | Critic "derived number" rule accepted random 2-to-3-digit sums/differences (`30 days`, `Rs 7,000`, `85 days`) and missed single-sentence attribute swaps | Restricted `_is_derived_from_evidence()` to require an explicit calculation cue (`total`, `combined`, `sum`, `plus`, `difference`, `minus`) in the answer AND both operands present in the same chunk and in the answer. Upgraded `_detect_swapped_attribute_numbers()` to split on phrase boundaries (commas, `and`, `while`, `but`, semicolons, periods). | `test_04` |
| **V3** | Postgres migration deleted orphan chunks *before* deleting duplicate documents, leaving new orphan chunks before `ADD CONSTRAINT` | Reordered `_migrate_existing_schema_and_constraints()` in `database.py` to: (1) backfill empty `file_hash`, (2) delete duplicate `(user_id, file_hash)` documents, (3) delete orphan `document_chunks` *after* duplicate removal, and (4) create `UNIQUE INDEX` and `FOREIGN KEY`. | `test_09` |
| **V4** | `render.yaml` selected native Python free plan instead of `Dockerfile` and omitted secret declarations | Updated `render.yaml` to `runtime: docker`, `dockerfilePath: ./Dockerfile`, and declared `JWT_SECRET`, `DATABASE_URL`, `GROQ_API_KEY`, `GOOGLE_CLIENT_ID`, and `PINECONE_API_KEY` (`sync: false`). | `test_07` |
| **V5** | Rate limiter DB writes & PostgreSQL `LIKE 'demo_%'` placeholder escaping | Parameterized `WHERE id LIKE ?` in `cleanup_expired_demo_users()` and escaped literal `%` before `?` -> `%s` conversion in `database.py`. | `test_14`, `test_16` |
| **V6** | Reproducibility: test corpus used 45-word chunks, no paraphrases in golden set, no `.github/workflows/ci.yml`, stale tracker/roadmap claims | Upgraded `test_02` to use production chunking (`chunk_size=180, chunk_overlap=25, max_chunk_size=250`) over a 32+ chunk multi-page corpus, added paraphrased questions to `tests/eval_golden_set.json`, added `.github/workflows/ci.yml`, and updated all docs with real `pytest` output. | `test_02`, CI workflow |

---

## 2. Part 2 Findings (`N1 – N12`)

| ID | Finding | Code Remediation | Verification Gate |
| :--- | :--- | :--- | :--- |
| **N1** | `get_real_client_ip()` took leftmost `X-Forwarded-For` (`hops[0]`) and rate limit rows grew unbounded | Changed `get_real_client_ip()` in `app.py` to use rightmost `hops[-1]` and added expired-window pruning in `check_and_increment_rate_limit()`. | `test_14` |
| **N2** | `get_all_user_chunks()` selected `embedding_json`, increasing BM25 payload ~5x | Removed `embedding_json` from `get_all_user_chunks()` and added dedicated `get_user_chunk_embeddings()` for dense fallback. | `test_11` |
| **N3** | Foreign key and `UNIQUE(user_id, file_hash)` only applied to newly created DBs | Added `_migrate_existing_schema_and_constraints()` in `database.py` for pre-existing Postgres and SQLite databases. | `test_09` |
| **N4** | Duplicate `(user_id, file_hash)` raised `DatabaseUnavailableError` (`503`) | Added `DatabaseIntegrityError` mapped to `HTTP 409 Conflict` and handled concurrent duplicate uploads as `"duplicate"`. | `test_09` |
| **N5** | `delete_user_account()` left rows in `upload_jobs`, `token_usage_daily`, and `rate_limit_counters` | Updated `delete_user_account()` to purge all 9 user-linked tables in a single transaction. | `test_12` |
| **N6** | DB embedding fallback only ran when Chroma returned zero hits | Updated `_dense_search()` in `retrieval_pipeline.py` to merge DB-persisted embeddings for any `allowed_doc_ids` missing from Chroma. | `test_13` |
| **N7** | Weak tests (`Recall@5` had 5 chunks for `top_k=5`, no near-misses, no OAuth nonce test) | Rebuilt `tests/test_acceptance_suite.py` with 32+ chunks at `chunk_size=180`, 10 domain near-misses, server-side OAuth nonce test, and `tests/eval_golden_set.json`. | `test_01`–`test_16` |
| **N8** | `delete_account` only revoked header token, not `aura_session` cookie | Updated `delete_account()` and `logout_session()` in `app.py` to revoke and clear `aura_session` cookie. | `test_12` |
| **N9** | Router single-doc comparison, critic swapped/spelled numbers, electricity bill category | Updated router, critic, and `Finance` vocabulary in `agent_swarm.py`. | `test_03`, `test_04`, `test_15` |
| **N10** | Quotas: 50k limit for all users, comparisons unmetered, multi-query rate limiter | Added tiered limits (`50,000` demo / `250,000` signed-in), metered `compare_documents()`, and single-call atomic SQL rate limiting. | `test_14` |
| **N11** | Hard-wrapped comma prose misread as CSV; CSV chunks exceeded 256 wordpieces | Restricted `_chunk_csv_text()` to `.csv` files and counted punctuation/numeric wordpieces (`<= 180` wordpieces per chunk). | `test_08` |
| **N12** | Demo expiry vector cleanup, `source_doc_id` memory linkage, LLM timeout cap, doc scrubbing | Added `source_doc_id` to `user_memory`, capped LLM fallback to 2 models x 8s, and scrubbed docs. | `test_15` |

---

## 3. Latest `pytest` Execution Output

```text
$ pytest tests/test_acceptance_suite.py -q
................                                                         [100%]
16 passed in 0.48s
```
