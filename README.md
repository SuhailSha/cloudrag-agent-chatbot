# AURA – Personal AI Assistant Vault / CloudRAG
**Production-Grade Multi-Tenant Document Intelligence Platform**

[![Render Status](https://img.shields.io/badge/Render-Live-brightgreen)](https://cloudrag-agent-chatbot.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Live Application: [cloudrag-agent-chatbot.onrender.com](https://cloudrag-agent-chatbot.onrender.com)  
Repository: [github.com/SuhailSha/cloudrag-agent-chatbot](https://github.com/SuhailSha/cloudrag-agent-chatbot)

---

## 1. System Overview

AURA is a modular, production-hardened document intelligence platform designed for personal documents (insurance policies, financial statements, utility bills, employment agreements, medical records, and resumes).

It combines **per-user database and vector namespace isolation**, **hybrid dense-sparse retrieval (BM25 + Chroma/Pinecone with DB-persisted vector fallback)**, and a **modular document QA and verification pipeline** to deliver grounded answers, side-by-side document comparisons, and consent-gated memory governance.

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

---

## 2. Core Capabilities & Architecture

### A. Production Security & Multi-Tenant Isolation
- **Strict Cryptographic JWT Authentication**: Validates algorithm (`HS256`), issuer (`aura-vault-auth`), audience (`aura-vault-users`), expiration, and mandatory `iat` issued-at timestamps, with `HttpOnly; Secure; SameSite=Lax` session cookies.
- **Fail-Closed Secrets Management**: In production, application halts startup if `JWT_SECRET` is missing or shorter than 32 characters.
- **Google OIDC Audience & Nonce Verification**: Verifies audience against configured Web Client ID (`277695292361-386p40ds1cptp6bf1f8v1sr8mr5tbv8v.apps.googleusercontent.com`) and validates OAuth state and nonce parameters.
- **Atomic Single-Transaction DB Mutations & Schema Migrations**: Document deletion, 9-table account deletion, and document + chunk uploads commit in single ACID transactions with foreign key cascading (`ON DELETE CASCADE`), unique `(user_id, file_hash)` indexing, and distinct `DatabaseIntegrityError` (`HTTP 409`) vs `DatabaseUnavailableError` (`HTTP 503`) handling.
- **Non-Spoofable Proxy Rate Limiting**: Extracts the rightmost `X-Forwarded-For` hop appended by Render's ingress proxy and enforces atomic single-statement DB rate limits with automatic expired-window pruning.

### B. Ingestion & Sanitation Engine
- **Scanned Document & Page-Cap Guard**: Enforces a 150-page PDF limit and rejects scanned image PDFs, encrypted files, and corrupt DOCX files with clean `HTTP 422` error messages.
- **Structure-Aware Chunking**:
  - **Prose & Markdown Tables**: 180-word target, 25-word overlap, 250-word hard ceiling, with Markdown table header repetition and zero prose overlap glued onto tables.
  - **CSV Files**: Strictly triggered on `.csv` files with row-aware splitting, repeated header rows, and an 85-token target (110-token ceiling) so numeric cells stay inside `all-MiniLM-L6-v2`'s 256-wordpiece window.

### C. Precision RAG & Grounding Engine
- **Lightweight O(1) Pre-Tokenized BM25 Sparse Search**: Queries text-only columns (`get_all_user_chunks`) without transferring vector JSON blobs, using pre-tokenized `Counter` maps and alphanumeric identifier boosts.
- **Split-Brain Proof Dense Vector Retrieval**: Stores embeddings in both Chroma/Pinecone and PostgreSQL/SQLite (`embedding_json`), automatically supplementing any document missing from ephemeral disk after container restarts.
- **Calibrated Relevance Gate & Controlled Abstention**: Evaluates raw cosine similarity (`dense_score >= 0.35`) and BM25 score (`sparse_score >= 1.60`) while excluding generic container words (`policy`, `document`, `cover`) from satisfying lexical overlap on their own.

### D. Modular Document QA & Verification Pipeline

**Sub-Processors & Data Handling Disclosure:** Document metadata and text chunks are stored in plaintext in Neon PostgreSQL and Pinecone/Chroma partitioned by `user_id`. During chat and comparison requests, relevant document excerpts are sent to the Groq Cloud API (`openai/gpt-oss-120b` / `openai/gpt-oss-20b`) over TLS to synthesize answers. Demo accounts automatically expire after 24 hours.
1. **Supervisor Router**: Classifies incoming queries into `DOCUMENT_RAG`, `DOCUMENT_COMPARISON`, `MEMORY_MANAGEMENT`, or `GENERAL_CHAT`, keeping single-document attribute comparisons on `DOCUMENT_RAG`.
2. **RAG Retrieval Agent**: Handles conversational pronoun resolution, hybrid search execution, and specific-term relevance gating.
3. **Document Analysis Agent**: Performs side-by-side multi-document comparisons with focus-topic retrieval and daily token metering.
4. **Memory Governance Agent**: Manages consent-gated user memory facts (`pending` -> `confirmed`) with `source_doc_id` linkage and Luhn credit-card filtering.
5. **Verification Critic Agent**: Audits generated claims against retrieved chunk text, normalizing spelled-out numbers, permitting valid derived sums/differences, catching swapped attribute numbers, and stripping fabricated citations.

---

## 3. Environment Variables Configuration

| Variable | Description | Required? | Default |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Neon PostgreSQL Connection URI (`postgres://...`) | Recommended | SQLite (`aura_vault.db`) |
| `JWT_SECRET` | Cryptographic secret for signing auth tokens (≥32 chars) | **YES (Prod)** | - |
| `GOOGLE_CLIENT_ID` | Google OAuth 2.0 Web Client ID | **YES (OAuth)** | `277695292361-...` |
| `GROQ_API_KEY` | Groq Cloud API Key for LLM inference | **YES (RAG)** | - |
| `PINECONE_API_KEY` | Optional Pinecone API Key for cloud vector storage | Optional | Chroma + DB Vector Fallback |
| `ALLOWED_ORIGINS` | Comma-separated list of whitelisted CORS origins | Optional | `https://cloudrag-agent-chatbot.onrender.com` |
| `ALLOW_SQLITE_FALLBACK` | Allow fallback to local SQLite when Postgres drops | Optional | `false` |
| `PORT` | Web server listening port | Optional | `10000` |
