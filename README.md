# AURA – Personal AI Assistant Vault / CloudRAG
**Production-Grade Multi-Tenant Document Intelligence Platform**

[![Render Status](https://img.shields.io/badge/Render-Live-brightgreen)](https://cloudrag-agent-chatbot.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Live Application: [cloudrag-agent-chatbot.onrender.com](https://cloudrag-agent-chatbot.onrender.com)  
Repository: [github.com/SuhailSha/cloudrag-agent-chatbot](https://github.com/SuhailSha/cloudrag-agent-chatbot)

---

## 1. System Overview

AURA is an enterprise-grade document intelligence platform designed for confidential personal documents (insurance policies, financial statements, employment agreements, medical records, and resumes). 

It combines **strict multi-tenant database isolation**, **hybrid dense-sparse retrieval (BM25 + Chroma/Pinecone)**, and a **typed multi-agent orchestration swarm** to deliver grounded answers, side-by-side document comparisons, and persistent memory governance with 100% tenant privacy.

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
- **Strict Cryptographic JWT Authentication**: Validates algorithm (`HS256`), issuer (`aura-vault-auth`), audience (`aura-vault-users`), expiration, and mandatory `iat` issued-at timestamps.
- **Fail-Closed Secrets Management**: In production, application halts startup if `JWT_SECRET` is missing or shorter than 32 characters.
- **Google OIDC Audience Verification**: Explicitly verifies audience against configured Web Client ID (`277695292361-386p40ds1cptp6bf1f8v1sr8mr5tbv8v.apps.googleusercontent.com`).
- **Atomic Single-Transaction DB Mutations**: Document deletion, account deletion, and uploads commit in a single ACID transaction, with foreign key cascading and rollback.
- **Hardened CORS & Security Headers**: Strict origin whitelisting (`https://cloudrag-agent-chatbot.onrender.com`), with `HSTS`, `X-Content-Type-Options: nosniff`, and `X-Frame-Options: DENY`.

### B. Ingestion & Sanitation Engine
- **Scanned Document Rejection**: Rejects scanned image PDFs, encrypted files, and corrupt DOCX files with clean `HTTP 422` error messages rather than indexing raw binary garbage.
- **Receptive-Field Aligned Chunking**: Text is chunked into 180 words base, 25 words overlap, and 250 words hard ceiling, fitting the 256-wordpiece receptive window of `all-MiniLM-L6-v2`.
- **Markdown Table Integrity**: Markdown table rows (`| ... |`) are kept atomically intact to prevent splitting mid-row.
- **Deduplication**: SHA-256 content hashing blocks duplicate indexing and saves storage and embedding costs.

### C. Precision RAG & Grounding Engine
- **True BM25 Sparse Scoring**: Inverse Document Frequency ($IDF$) with whole-word boundaries, stopword filtering, and alphanumeric token boosts.
- **Reciprocal Rank Fusion**: Merges dense and sparse rankings with $k=60$.
- **Relevance Cutoff & Controlled Abstention**: Requires minimum evidence threshold ($RRF \ge 0.012$); abstains with a clear message when relevant evidence is absent.
- **Prompt Injection Delimiter Defense**: Encloses document evidence in `<verified_document_evidence>` and escapes delimiter tags.

### D. Typed Multi-Agent Swarm
1. **Supervisor Router**: Classifies incoming queries into `DOCUMENT_RAG`, `DOCUMENT_COMPARISON`, `MEMORY_MANAGEMENT`, or `GENERAL_CHAT`.
2. **RAG Retrieval Agent**: Handles conversational pronoun resolution and hybrid search execution.
3. **Document Analysis Agent**: Performs deep side-by-side contract/document comparisons with focus-topic driven retrieval.
4. **Memory Governance Agent**: Manages user memory facts with consent tracking, provenance, and Luhn credit-card filtering.
5. **Verification Critic Agent**: Audits generated claims against retrieved chunk text to catch unsupported assertions and validate citations.

---

## 3. Environment Variables Configuration

| Variable | Description | Required? | Default |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Neon PostgreSQL Connection URI (`postgres://...`) | Recommended | SQLite (`aura_vault.db`) |
| `JWT_SECRET` | Cryptographic secret for signing auth tokens (≥32 chars) | **YES (Prod)** | - |
| `GOOGLE_CLIENT_ID` | Google OAuth 2.0 Web Client ID | **YES (OAuth)** | `277695292361-...` |
| `GROQ_API_KEY` | Groq Cloud API Key for Llama-3.3-70b inference | **YES (RAG)** | - |
| `PINECONE_API_KEY` | Optional Pinecone API Key for cloud vector storage | Optional | Chroma ONNX Local |
| `ALLOWED_ORIGINS` | Comma-separated list of whitelisted CORS origins | Optional | `https://cloudrag-agent-chatbot.onrender.com` |
| `ALLOW_SQLITE_FALLBACK` | Allow fallback to local SQLite when Postgres drops | Optional | `false` |
| `PORT` | Web server listening port | Optional | `10000` |

---

## 4. Production Deployment on Render

1. Fork or push changes to your GitHub repository:
   ```bash
   git push origin main
   ```
2. In the [Render Dashboard](https://dashboard.render.com):
   - Choose your Web Service (`cloudrag-agent-chatbot`).
   - Under **Environment Variables**, ensure the keys above are populated.
3. Click **Manual Deploy** &rarr; **Clear build cache & deploy**.
4. Monitor the deployment logs. When ready, `/health/live` and `/health/ready` will return `200 OK`.
