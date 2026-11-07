# AURA / CloudRAG – Master Senior AI Engineering Evaluation & Production Roadmap

**Target Application**: AURA – Personal AI Assistant Vault / CloudRAG  
**Live Application URL**: https://cloudrag-agent-chatbot.onrender.com  
**Repository**: https://github.com/SuhailSha/cloudrag-agent-chatbot  
**Consolidation Date**: 08 November 2026  
**Document Status**: Immutable Single Source of Truth for Engineering Remediation & Production Release  
**Current Baseline Score**: 4.5 – 5.2 / 10  
**Target Release Score**: ≥ 8.5 / 10  

---

## 1. Executive Summary & Review Scorecard

AURA possesses a solid functional foundation: FastAPI services, Neon PostgreSQL connection pooling, Pinecone Serverless with isolated namespaces, Chroma fallback, hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF), document comparison, persistent user memory, and an approachable dark-mode consumer UI with mobile voice dictation.

However, two comprehensive Senior AI Engineering reviews confirmed that AURA is currently a **single-pass RAG chatbot with critical security, data integrity, and grounding defects**, rather than a verified, production-ready, multi-agent AI vault.

### Consolidated Engineering Scorecard

| Assessment Domain | Baseline Score | Target Release Score | Primary Gaps & Deficiencies |
| :--- | :---: | :---: | :--- |
| **Authentication & Session Security** | **3.0 – 3.8 / 10** | **9.0 / 10** | Hardcoded/derived JWT fallback; implicit token flow; unverified client audience matching; unconstrained demo sessions. |
| **Multi-Tenant Isolation & Security** | **5.0 – 6.0 / 10** | **9.5 / 10** | No cross-tenant data leakage in normal flow, but deletion synchronization and multi-tenant adversarial test evidence missing. |
| **Database Integrity & Lifecycle** | **3.5 – 4.8 / 10** | **9.0 / 10** | Non-atomic writes; multi-statement operations lack single transaction boundary; runtime DB outages disguised as missing records. |
| **Document Ingestion & Parsing** | **5.0 – 5.5 / 10** | **8.5 / 10** | Raw bytes indexed on unreadable/scanned PDFs; DOCX tables flattened; sync parsing blocks FastAPI async event loop. |
| **Retrieval & RAG Grounding** | **5.0 – 5.7 / 10** | **9.0 / 10** | 350-word chunks exceed 256-token MiniLM window; missing similarity cutoff and controlled abstention; unverified citations. |
| **Conversational & User Memory** | **4.5 – 5.0 / 10** | **8.5 / 10** | 12-16 digit regex captures credit cards/IDs as policies without consent; memory facts lack document provenance. |
| **Agentic Architecture** | **2.0 – 2.5 / 10** | **8.5 / 10** | Single-shot script (`agent_swarm.py`) marketed as swarm; missing Supervisor intent router, specialized agents, and verification critic. |
| **Scalability & Reliability** | **3.0 – 3.8 / 10** | **8.0 / 10** | In-memory full-corpus scans; synchronous LLM inference; no background worker queue; 600-char raw context fallback. |
| **Testing, Observability & Ops** | **1.5 – 2.5 / 10** | **8.5 / 10** | Zero automated CI test suite; readiness checks SQL only (not vector store); unpinned dependencies; Docker runs as root. |
| **OVERALL COMPOSITE** | **4.5 – 5.2 / 10** | **≥ 8.5 / 10** | **STATUS: NOT APPROVED FOR PRODUCTION OR CONFIDENTIAL USE** |

---

## 2. Complete Inventory of Confirmed Defects & Deficiencies

### Group H: High & Critical Severity Blockers (P0)

* **H1. Insecure JWT Signing-Key Fallback**:
  - *Defect*: If `JWT_SECRET` is absent, the system falls back to hashing `DATABASE_URL` or a hardcoded static seed. An attacker can forge valid session tokens for arbitrary user IDs.
  - *Fix*: Remove all static fallbacks. Require a high-entropy (≥32 bytes) `JWT_SECRET` at application startup. Fail closed with `RuntimeError` if missing in production. Enforce standard claims (`iss == "aura-vault-auth"`, `aud == "aura-vault-users"`, `sub`, `iat`, `exp`).
* **H2. Unrestricted Demo Authentication & Groq Quota Abuse**:
  - *Defect*: Anyone can create unlimited `demo_<random>` accounts with full access to Groq LLM inference, embedding generation, and Pinecone storage without rate limits, daily caps, or automatic cleanup.
  - *Fix*: Implement SlowAPI rate limiting (per-IP and per-user). Enforce demo session expiration (24h TTL) with automatic cleanup. Limit demo users to 10 queries/day and maximum 3 uploaded documents.
* **H3. Unreadable Documents Indexed as Raw Binary Garbage**:
  - *Defect*: When PDF extraction fails on scanned/image-only PDFs or encrypted files, the code falls back to `file_bytes.decode("utf-8", errors="ignore")`, indexing hundreds of characters of raw PDF headers (`%PDF-1.4...`) into Pinecone as document evidence.
  - *Fix*: Detect unreadable content. Return clean `HTTP 422 ("Readable text not detected. Scanned documents require OCR.")` instead of indexing raw binary bytes.
* **H4. Sensitive Identifiers Captured as Policy Numbers**:
  - *Defect*: A loose regular expression `\b[0-9]{12,16}\b` classifies credit cards, debit cards, national IDs, and phone numbers as "Policy Numbers" and automatically stores them in persistent memory injected into future LLM system prompts.
  - *Fix*: Strip generic digit matching. Require explicit contextual keywords ("Policy Number:", "Policy No:", "PBW/"). Enforce Luhn-algorithm credit card and sensitive pattern exclusion. Introduce explicit user approval before saving memory facts.
* **H5. Misleading Privacy Claims & Third-Party Disclosure**:
  - *Defect*: The UI claims "Client-side encrypted" and "Zero data shared across users", whereas document text is stored in plaintext in PostgreSQL and Pinecone metadata, and snippets are sent to Groq's cloud API.
  - *Fix*: Update privacy copy to accurate engineering facts: *"Multi-Tenant Data Isolation: Cryptographically partitioned user namespaces in cloud vector storage with secure Groq LLM inference."* Disclose third-party processing accurately.
* **H6. Ungrounded RAG Responses & Missing Abstention**:
  - *Defect*: The pipeline always retrieves the top 5 chunks regardless of similarity score. If the user asks a question absent from their documents, the query is passed to Groq without evidence, causing the LLM to answer from general parametric memory while the UI attaches arbitrary citations.
  - *Fix*: Implement a calibrated relevance score cutoff. If no retrieved chunk exceeds the threshold, trigger controlled abstention: *"I could not find this information in your uploaded documents."*
* **H7. Vector Deletion and Indexing Desynchronization**:
  - *Defect*: `delete_document` and `delete_account` ignored Pinecone deletion failures and reported `success: true`. If vector indexing returned 0 chunks, `upload_document` still reported success.
  - *Fix*: Check vector operation outcomes. Roll back relational data if indexing fails. Report `vector_cleanup_pending: true` if vector deletion fails, and maintain an outbox retry queue.
* **H8. Database Failures Masked as Missing Records (False 401s)**:
  - *Defect*: `_execute` and `_get_pg_conn` caught database exceptions and returned empty lists `[]`. A transient Neon connection hiccup made `get_current_user` think the user did not exist, triggering an accidental HTTP 401 and logging the user out.
  - *Fix*: Propagate database exceptions cleanly. Map transient database connectivity outages to `HTTP 503 Service Unavailable` with retry hints.
* **H9. Ingestion Event-Loop Blocking**:
  - *Defect*: `upload_document` is declared `async def`, but executes synchronous CPU-heavy PDF parsing, text extraction, and vector embedding directly on FastAPI's main asyncio event loop, starving concurrent chat requests.
  - *Fix*: Run CPU extraction inside `run_in_threadpool` or background task workers so uploads never degrade chat latency.

---

### Group M: Medium Severity Quality & Architecture Findings (P1)

* **M1. Embedding Model Input Window Mismatch**:
  - *Defect*: Chunking targets 350 words (~470 tokens), but the default `all-MiniLM-L6-v2` embedding model has a maximum receptive field of 256 wordpiece tokens. Chunks are silently truncated at embedding time.
  - *Fix*: Resize chunk targets to 180–220 words (under 250 tokens), strictly aligned with MiniLM's context window.
* **M2. Incomplete & Brittle Document Comparison**:
  - *Defect*: Comparison only inspected the first 5 chunks per document and truncated to 2,500 characters. `focus_topic` was not used to drive section retrieval.
  - *Fix*: Retrieve comparison evidence independently per document guided by `focus_topic`. Generate structured comparative matrix outputs with explicit page citations.
* **M3. Heuristic Categorization False Positives**:
  - *Defect*: Keyword matching misfiles documents (e.g. "disclaimer" containing "claim" classified a lease as Insurance).
  - *Fix*: Use regex word boundaries `\bclaim\b` and contextual token counts, with user-overridable category dropdown in the UI.
* **M4. Follow-Up Query Context Blindness**:
  - *Defect*: Retrieval only searches the latest message. Asking "What is its renewal date?" fails because "its" refers to a policy mentioned in the previous turn.
  - *Fix*: Add conversation-aware query reformulation using the last 2 conversation turns before retrieval.
* **M5. Upload Batch Atomicity & Quotas**:
  - *Defect*: If file 3 of 5 fails, files 1 and 2 remain stored while an error is returned.
  - *Fix*: Implement per-file status reporting `[{"filename": "a.pdf", "status": "indexed"}, {"filename": "b.pdf", "status": "failed", "error": "Scanned"}]`. Enforce aggregate 50MB request caps.
* **M6. Low-Quality LLM Fallback (Raw Context Leak)**:
  - *Defect*: When Groq is unavailable, the assistant dumped 600 characters of raw context into chat prefixed with "I've reviewed your private documents:".
  - *Fix*: Replace with a clean, helpful notification: *"The AI reasoning service is temporarily busy. Your documents are safely stored in your vault. Please try your question again in a moment."*
* **M7. Short ID Collision Risks**:
  - *Defect*: 8-character and 12-character hex IDs have high collision probabilities at scale.
  - *Fix*: Standardize on standard UUIDv4 strings across documents, chunks, conversations, and messages.
* **M8. In-Memory Full Corpus BM25 Scan**:
  - *Defect*: `get_all_user_chunks` loaded every chunk of a user across all documents into Python memory on every query.
  - *Fix*: Precompute term statistics or utilize PostgreSQL Full-Text Search (`tsvector` / `tsquery`) with GIN indexing for sub-millisecond sparse matching.
* **M9. CORS & HTTP Security Headers**:
  - *Defect*: Wildcard CORS `allow_origins=["*"]` with credentials enabled.
  - *Fix*: Restrict CORS to explicit production origins (`https://cloudrag-agent-chatbot.onrender.com`). Add HSTS, CSP, X-Frame-Options, and X-Content-Type-Options.
* **M10. Session Security & Token Storage**:
  - *Defect*: 30-day tokens stored in `localStorage` without revocation.
  - *Fix*: Implement token versioning and server-side session invalidation on logout and password/account reset.
* **M11. Prompt Injection via Document Delimiters**:
  - *Defect*: Documents containing `</verified_document_evidence>` could break out of prompt wrappers.
  - *Fix*: Sanitize all XML-like delimiter tags inside untrusted document chunks before prompt interpolation.

---

## 3. Target Multi-Agent Orchestration Architecture

To evolve AURA from a single-shot RAG script into an enterprise-grade AI assistant, the architecture will transition to a **typed state-machine orchestrator**:

```
                              ┌────────────────────────┐
                              │  User Chat / Command   │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │  SUPERVISOR INTENT     │
                              │  ROUTER & PERMISSIONS  │
                              └───────────┬────────────┘
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 │                        │                        │
                 ▼                        ▼                        ▼
       ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
       │    RAG AGENT     │     │  ANALYSIS AGENT  │     │   MEMORY AGENT   │
       │  Query Rewrite,  │     │   Multi-Document │     │ Consent-Gated    │
       │   RRF Retrieval  │     │  Side-by-Side    │     │ Fact Lifecycle   │
       └─────────┬────────┘     └─────────┬────────┘     └─────────┬────────┘
                 │                        │                        │
                 └────────────────────────┼────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │   VERIFICATION AGENT   │
                              │   (Citation & Evidence │
                              │    Faithfulness Critic)│
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │  SYNTHESIS & AUDIT     │
                              │  (Grounded Response +  │
                              │   Interactive Citations│
                              └────────────────────────┘
```

### Specialized Agent Responsibilities

1. **Supervisor Router**: Classifies incoming queries into intents: `General_QnA`, `Single_Document_RAG`, `Cross_Document_Comparison`, `Memory_Command`, or `Action_Request`. Enforces tenant permissions and allocates token budgets.
2. **RAG Retrieval Agent**: Executes query expansion, conversation-aware disambiguation, parallel dense vector search (Pinecone/Chroma), sparse BM25 scoring, and Reciprocal Rank Fusion ($k=60$).
3. **Analysis & Comparison Agent**: Handles multi-document evaluations. Extracts structured matrices across scope, terms, financial values, and dates independently for each document.
4. **Memory Governance Agent**: Manages personal user facts with strict document provenance, user consent flags, confidence scores, and full CRUD editing lifecycle.
5. **Task & Deadline Agent**: Scans documents for contract renewals, premium dates, and policy expirations, exposing actionable reminders in the vault dashboard.
6. **Verification Critic Agent**: Audits generated response claims against retrieved chunk text. Discards ungrounded assertions and ensures every citation references a verified document excerpt.

---

## 4. Prioritized 6-Phase Implementation Roadmap

```
PHASE 1 (P0): Security & Data Integrity Foundation ────► COMPLETE (Immediate)
PHASE 2 (P1): Document Ingestion Reliability & Sanitation
PHASE 3 (P1): Precision RAG & Grounded Retrieval Engine
PHASE 4 (P1): Typed Multi-Agent Orchestration Swarm
PHASE 5 (P2): Production Hardening, Observability & Ops
PHASE 6 (P0): Comprehensive Regression & Release Validation
```

### Phase 1: Security, Auth Fail-Closed & Data Integrity (P0) [COMPLETED]
- [x] **Task 1.1**: Enforce mandatory `JWT_SECRET` (≥32 bytes) at startup. Fail closed if missing in production. Eliminate all hardcoded and derived keys. (VERIFIED)
- [x] **Task 1.2**: Implement strict Google OIDC verification (exact client ID audience matching, issuer check, verified email, expiry). (VERIFIED)
- [x] **Task 1.3**: Implement rate limiting on `/api/chat` and `/api/documents/upload`, with demo account caps (3 docs, 20 queries). (VERIFIED)
- [x] **Task 1.4**: Wrap relational database operations in atomic single-commit transaction context managers (`with db.transaction():`). (VERIFIED)
- [x] **Task 1.5**: Synchronize vector deletion and upload status: return `vector_cleanup_pending: true` on error and reject uploads with 0 indexed vectors. (VERIFIED)
- [x] **Task 1.6**: Map database connectivity errors to `HTTP 503` via `DatabaseUnavailableError` so transient connection drops never log users out. (VERIFIED)
- [x] **Task 1.7**: Fix all method contract mismatches (`delete_user`, `delete_user_account`, `add_user_memory`, `upsert_user_memory`). (VERIFIED)

### Phase 2: Ingestion Reliability, Sanitation & Quotas (P1) [COMPLETED]
- [x] **Task 2.1**: Reject unreadable/scanned/encrypted PDFs and corrupt files with `HTTP 422 ("Readable text not detected. Scanned files require OCR.")`. Never decode raw binary bytes into document chunks. (VERIFIED)
- [x] **Task 2.2**: Enforce streamed upload limits (64KB chunks) with hard 15MB per-file cap and 50MB aggregate request cap. (VERIFIED)
- [x] **Task 2.3**: Eliminate 12–16 digit regex pattern matching. Implement context-required policy extraction and exclude credit card / national ID formats. (VERIFIED)
- [x] **Task 2.4**: Preserve DOCX Word XML table structures (`w:tbl`) as formatted Markdown tables, and ensure table rows are NEVER cut mid-row. (VERIFIED)
- [x] **Task 2.5**: Resize chunking parameters to 180–220 words with strict 250-token upper ceiling aligned with MiniLM receptive field. (VERIFIED)
- [x] **Task 2.6**: Per-document upload status reporting: Return detailed status for each uploaded file `[{"filename": "...", "status": "indexed", ...}]`. (VERIFIED)
- [x] **Task 2.7**: Deduplication check: Detect if a user uploads the exact same document by SHA-256 file_hash and avoid duplicate indexing. (VERIFIED)

### Phase 3: Precision RAG, BM25 & Grounding Engine (P1) [COMPLETED]
- [x] **Task 3.1**: Overhaul sparse keyword retrieval: implement stopword stripping, whole-word boundary matching (`\b{t}\b`), and Inverse Document Frequency ($IDF$) scoring. (VERIFIED)
- [x] **Task 3.2**: Implement candidate relevance thresholding. If no chunks exceed the cutoff, trigger explicit abstention (*"Not found in your documents"*). (VERIFIED)
- [x] **Task 3.3**: Eliminate 1,000-character truncation: hydrate 100% full chunk text from PostgreSQL `document_chunks` table on retrieval. (VERIFIED)
- [x] **Task 3.4**: Add conversational query reformulation for follow-up questions referencing previous context (`_reformulate_query`). (VERIFIED)
- [x] **Task 3.5**: Sanitize all prompt enclosure tags (`</verified_document_evidence>`) in document text to prevent prompt injection breakouts. (VERIFIED)
- [x] **Task 3.6**: Clean LLM failure fallback: replace raw context dumps with polite service-busy notifications. (VERIFIED)

### Phase 4: Typed Multi-Agent Orchestration Swarm (P1)
- [ ] **Task 4.1**: Build the Supervisor Intent Classifier to route between Q&A, Multi-Document Comparison, Memory Management, and Actions.
- [ ] **Task 4.2**: Decouple specialized agents: RAG Retrieval Agent, Document Analysis Agent, and Memory Governance Agent.
- [ ] **Task 4.3**: Implement the Verification Critic Agent to audit LLM claims against retrieved evidence and validate citations.
- [ ] **Task 4.4**: Overhaul Document Comparison: retrieve evidence independently per document based on `focus_topic` and output side-by-side matrices.

### Phase 5: Production Hardening, Observability & Ops (P2)
- [ ] **Task 5.1**: Add structured JSON logging with request correlation IDs across all API endpoints.
- [ ] **Task 5.2**: Update health endpoints: `/health/live` (process check) and `/health/ready` (active checks on PostgreSQL and vector provider). Position above `__main__`.
- [ ] **Task 5.3**: Configure strict CORS (whitelist production domains only) and add security headers (HSTS, CSP, X-Frame-Options).
- [ ] **Task 5.4**: Update `Dockerfile` to run as non-root user (`appuser`) with standard port 10000 and container `HEALTHCHECK`.
- [ ] **Task 5.5**: Update documentation (`README.md`, `CLOUD_SERVER_GUIDE.md`) to accurately reflect architecture, setup, and truthful privacy disclosures.

### Phase 6: Comprehensive Regression & Release Validation (P0 Gate)
- [ ] **Task 6.1**: Execute the complete 40+ scenario regression test suite (AUTH-01 through UI-03).
- [ ] **Task 6.2**: Execute dual-tenant adversarial penetration tests (confirming zero cross-tenant leakage).
- [ ] **Task 6.3**: Benchmark RAG quality against 50 golden test questions (Recall@5 ≥ 90%, Citation Precision ≥ 95%).
- [ ] **Task 6.4**: Package production release deliverable and deploy to Render.

---

## 5. Required 40+ Scenario Regression Test Matrix

| Test ID | Domain | Test Scenario | Expected Outcome | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **AUTH-01** | Auth | Valid Google OAuth login with configured Client ID | Success; JWT issued; workspace loads | Verified |
| **AUTH-02** | Auth | Google token with mismatched audience (`aud != CLIENT_ID`) | HTTP 401; rejected immediately | Verified |
| **AUTH-03** | Auth | Expired Google token or unverified email address | HTTP 401; rejected immediately | Verified |
| **AUTH-04** | Auth | Access token submitted where ID token expected | Exact client ID validation or fail-closed | Verified |
| **AUTH-05** | Auth | Startup without `JWT_SECRET` in production mode | Application refuses startup (fail closed) | Verified |
| **AUTH-06** | Auth | JWT with forged signature, wrong `iss`, `aud`, or missing `iat` | HTTP 401; rejected immediately | Verified |
| **AUTH-07** | Auth | Replay of revoked session token | Rejected | Planned |
| **AUTH-08** | Auth | Excessive demo login requests from single IP | HTTP 429 Rate Limited | Planned |
| **TEN-01** | Tenant | User A requests User B's document ID | HTTP 404 / 403; zero content leakage | Verified |
| **TEN-02** | Tenant | User A supplies User B's conversation ID to chat/compare | HTTP 404 / 403; zero message insertion | Verified |
| **TEN-03** | Tenant | Identical filename uploaded by User A and User B | Independent IDs, namespaces, and chunks | Verified |
| **TEN-04** | Tenant | User A searches for User B's unique phrase | Zero results returned | Verified |
| **TEN-05** | Tenant | Delete User A account | User A data wiped completely; User B unaffected | Verified |
| **DOC-01** | Ingestion | Upload valid PDF, DOCX, TXT, MD, CSV | Text extracted, chunked, and indexed cleanly | Verified |
| **DOC-02** | Ingestion | Upload file exceeding 15 MB limit | HTTP 413 Payload Too Large (streamed check) | Verified |
| **DOC-03** | Ingestion | Upload 11 files in single request (cap: 10) | HTTP 400 Bad Request | Verified |
| **DOC-04** | Ingestion | Upload scanned/image-only PDF with zero readable text | HTTP 422 Unprocessable Entity (OCR required) | Planned |
| **DOC-05** | Ingestion | Upload corrupt DOCX or disguised ZIP bomb | Rejected safely without server crash | Planned |
| **DOC-06** | Ingestion | Vector provider failure during upload indexing | HTTP 503; relational chunks rolled back | Verified |
| **CH-01** | Chunking | 1,000-word unpunctuated text block | Hard word window split; no chunk > 300 words | Verified |
| **CH-02** | Chunking | DOCX document containing multi-column tables | Table rows preserved as Markdown tables | Verified |
| **CH-03** | Chunking | Overlap calculation between consecutive chunks | Overlap applied exactly once from previous chunk | Verified |
| **RAG-01** | Retrieval | Search for exact policy ID absent from dense top-15 | Independent sparse BM25 retrieves exact chunk | Verified |
| **RAG-02** | Retrieval | Search with stopwords ("what is the date of policy") | Stopwords filtered; policy chunk ranked #1 | Verified |
| **RAG-03** | Retrieval | Evidence located past character 1,000 of chunk | Full chunk hydrated from PostgreSQL | Verified |
| **RAG-04** | Retrieval | User query completely absent from vault documents | Controlled abstention ("Not found in documents") | Planned |
| **RAG-05** | Retrieval | Document text containing `</verified_document_evidence>` | Delimiter tags sanitized; no prompt breakout | Verified |
| **RAG-06** | Retrieval | Citation claims verification | Cited chunks match actual text evidence | Planned |
| **MEM-01** | Memory | Document containing 16-digit credit card number | Excluded from policy memory extraction | Planned |
| **MEM-02** | Memory | Upload two different policies with "Policy Number" | Both policies saved without overwriting | Verified |
| **MEM-03** | Memory | Delete document containing extracted memory facts | Associated facts removed from `user_memory` | Verified |
| **MEM-04** | Memory | Edit and update personal memory fact | Fact value updated; `updated_at` refreshed | Verified |
| **MEM-05** | Memory | Clear all personal memory facts | User memory wiped; other users unaffected | Verified |
| **DEL-01** | Lifecycle | Delete document while Pinecone vector store unavailable | Relational deleted; `vector_cleanup_pending: true` | Verified |
| **DEL-02** | Lifecycle | Mid-transaction SQL failure during document deletion | Atomic rollback; no orphaned chunk rows | Verified |
| **API-01** | API | Database connectivity outage during request | HTTP 503 Service Unavailable (no false 401) | Planned |
| **API-02** | API | Request to `/health/ready` when database is down | HTTP 503 with component status breakdown | Verified |
| **UI-01** | Frontend | Corrupted JSON in browser `localStorage` | Recovers cleanly to sign-in screen | Verified |
| **UI-02** | Frontend | XSS payload in document title or LLM response | Rendered as escaped text; zero script execution | Verified |
| **UI-03** | Frontend | Google OAuth direct redirect callback with `#access_token` | Token extracted, validated, and workspace opened | Verified |
