# AURA – Enterprise Personal AI Assistant Architecture, Evaluation & Roadmap

**Current Version:** 5.0.0  
**Target Architecture:** Multi-Tenant, Zero-Leakage Personal Intelligence Platform  
**Live Application:** [cloudrag-agent-chatbot.onrender.com](https://cloudrag-agent-chatbot.onrender.com/)  
**GitHub Repository:** [github.com/SuhailSha/rag-agent-chatbot](https://github.com/SuhailSha/rag-agent-chatbot)  
**Evaluation Standard:** Senior AI Engineer Production & Security Review

---

## 1. Executive Summary & Evaluation Analysis

An independent Senior AI Engineering review evaluated AURA across 15 technical dimensions. The assessment identified strong foundational progress (FastAPI modularity, Neon PostgreSQL connection pooling, Pinecone namespace isolation, and clean UI design), while identifying critical vulnerabilities and technical debt that must be resolved before production readiness.

### Key Evaluation Verdicts
1. **Security & Authentication (Highest Priority)**:
   - *Finding 4.1 (Demo Collision)*: Demo mode lacked per-session cryptographic isolation.
   - *Finding 4.2 (JWT Secret)*: Relied on a fallback secret rather than enforcing a cryptographically random environment key.
   - *Finding 4.3 (OAuth Validation)*: Audience mismatch was logged rather than returning an immediate `401 Unauthorized`.
   - *Finding 4.4 (Privacy Claims)*: UI marketing text claimed "end-to-end encryption", which must be revised to accurately state transport TLS + tenant-isolated cloud database encryption.
   - *Finding 4.5 (Incomplete Deletion)*: Account deletion purged PostgreSQL and Pinecone but omitted local fallback artifacts and session tokens.
   - *Finding 4.6 (Legacy Migration Risk)*: Auto-claiming vectors from default namespace `""` created potential ownership ambiguity.
2. **Retrieval & RAG Quality**:
   - *Chunking*: Fixed word-based slicing cuts clauses, tables, and headers in two.
   - *Scoring*: ChromaDB fallback assigned hardcoded 0.85 similarity scores instead of true cosine distance.
   - *Keyword Gaps*: Dense semantic search alone struggles with exact alphanumeric strings (e.g. policy `PBW/000000775561` or vehicle `PY02Y3375`). Requires Hybrid BM25 + Dense Vector fusion.
   - *Reranking*: Missing cross-encoder reranking stage (`bge-reranker`) prior to context injection.
3. **Conversational Memory**:
   - Limited to 6-message rolling history without long-term user profile or episodic memory.
4. **Agent Orchestration**:
   - Monolithic assistant execution without deterministic intent routing or tool delegation.

---

## 2. Comprehensive 5-Phase Master Roadmap

```
AURA PRODUCTION ROADMAP
│
├── PHASE 1: Security, Authentication & Isolation Hardening (CRITICAL)
│   ├── [1.1] Enforce Cryptographic JWT Secret (Fail startup if missing)
│   ├── [1.2] Strict Google OIDC Audience & Expiry Enforcement (Reject on mismatch)
│   ├── [1.3] Isolated Session-Scoped Demo Tenants (Zero user collisions)
│   ├── [1.4] Complete Atomic Account Purge (DB, Pinecone, Chroma, Session revocation)
│   ├── [1.5] Remove Legacy Unauthenticated Vector Migration
│   └── [1.6] UI Privacy & Encryption Transparency Alignment
│
├── PHASE 2: Advanced RAG Pipeline & Retrieval Quality (HIGH)
│   ├── [2.1] Structure-Aware Recursive Chunking (Page, Table, Header preservation)
│   ├── [2.2] Real Calibrated Cosine Similarity Scoring (Remove hardcoded scores)
│   ├── [2.3] Hybrid Dense Vector + BM25 Keyword Search
│   ├── [2.4] Cross-Encoder Reranking Layer (bge-reranker-base)
│   ├── [2.5] Grounded Fact-Checking & Citation Verifier
│   └── [2.6] Document Prompt Injection Defense Layer
│
├── PHASE 3: 3-Layer Personal Conversational Memory (HIGH)
│   ├── [3.1] Layer 1: Contextual Working Memory (Current conversation)
│   ├── [3.2] Layer 2: Long-Term User Fact & Preference Vault (PostgreSQL-backed)
│   ├── [3.3] Layer 3: Episodic Decision & Milestone Memory
│   └── [3.4] User-Controlled Memory Management UI (Inspect, Edit, Forget)
│
├── PHASE 4: Agentic Intent Routing & Specialized Capabilities (MEDIUM)
│   ├── [4.1] Deterministic Intent Router (Doc Search, Comparison, Math/Analysis, Chit-Chat)
│   ├── [4.2] Multi-Document Comparative Matrix Analyzer
│   ├── [4.3] Mobile Voice Dictation (Web Speech API)
│   └── [4.4] Optical Character Recognition (OCR for Scanned Bills & RC Cards)
│
└── PHASE 5: Testing, Evaluation & Operational Monitoring (PRODUCTION)
    ├── [5.1] Automated 100-Question RAG Evaluation Suite (Ragas / TruLens)
    ├── [5.2] Multi-Tenant Penetration & Leakage Test Suite
    ├── [5.3] OpenTelemetry & Structured JSON Logging
    └── [5.4] Database Backup & Disaster Recovery Runbook
```

---

## 3. Detailed Specifications by Phase

### Phase 1: Security & Authentication Hardening
- **JWT Key Enforcement**: `os.environ["JWT_SECRET"]`. If missing in production, backend halts with explicit configuration error.
- **Strict OIDC Validation**: Google `id_token` must strictly match `aud == GOOGLE_CLIENT_ID`, `iss in ['accounts.google.com', 'https://accounts.google.com']`, and verify `email_verified == True`. Any discrepancy immediately returns HTTP 401.
- **Cryptographic Demo Isolation**: Each demo sign-in generates a unique random UUID (`usr_demo_{uuid4().hex}`) with its own temporary isolated Pinecone namespace. Demo users cannot share data.
- **Atomic Deletion**: Cascade removes PostgreSQL rows, executes `pinecone.delete(delete_all=True, namespace=ns)`, purges Chroma collection, and blacklists JWT session tokens.

### Phase 2: Advanced RAG & Retrieval Quality
- **Hybrid Search Architecture**:
  ```
  Query ──┬── Dense Vector Search (Pinecone / BGE Embeddings) ──┐
          └── Sparse Keyword Search (BM25 / Alphanumeric)      ──┼── Reciprocal Rank Fusion (RRF) ──> Reranker (Top 5)
  ```
- **Structure-Aware Extraction**: Extract tables as Markdown tables; capture page numbers and section headers in chunk metadata.
- **Calibrated Scoring**: Use true cosine distance `1.0 - distance` without arbitrary score clamping.
- **Prompt Injection Guard**: Sanitize document text chunks with structural XML wrapping (`<document_evidence>` tags) so document text cannot hijack system instructions.

### Phase 3: 3-Layer Personal Memory System
- **Layer 1 (Short-term)**: Sliding window token-managed conversation history.
- **Layer 2 (User Facts)**: Stored in PostgreSQL `user_memory` table (e.g. "User owns Toyota Etios 2011", "Insurance expires November 2027"). Automatically extracted and updated with user consent.
- **Layer 3 (Episodic)**: Key decisions and questions indexed in vector memory for cross-conversation recall.
- **UI Management**: A dedicated "My Assistant Memory" tab where users can view, edit, or delete what the assistant has remembered.

### Phase 4: Agentic Capabilities
- **Capability Router**: Fast keyword/pattern classifier routes query to Document RAG, Comparison Matrix, or General Knowledge without redundant LLM calls.
- **Comparative Analysis**: Select 2+ documents in My Vault and render side-by-side scorecard Markdown tables.
- **Mobile Voice Dictation**: Hands-free 1-tap mic input in the chat bar using native Web Speech API.
- **OCR Engine**: Image and scanned PDF text extraction.

### Phase 5: Production Readiness & Quality Assurance
- **Benchmark Suite**: Automated tests measuring Recall@5, Answer Relevance, Citation Faithfulness, and Hallucination Rate.
- **Isolation Tests**: Continuous integration test asserting that User B cannot query or retrieve any vector or record belonging to User A.

---

## 4. Current Execution Status

| Milestone | Component | Status | Target Date |
| :--- | :--- | :--- | :--- |
| **Phase 1.1 - 1.6** | Security, OAuth Hardening & True Demo Isolation | **In Progress** | Current Sprint |
| **Phase 2.1 - 2.6** | Structure-Aware Chunking, Hybrid Search & Scoring | Pending | Sprint 2 |
| **Phase 3.1 - 3.4** | 3-Layer Persistent Personal Memory | Pending | Sprint 3 |
| **Phase 4.1 - 4.4** | Intent Router, Comparisons & Mobile Voice | Pending | Sprint 4 |
| **Phase 5.1 - 5.4** | Automated Eval, Load Testing & Monitoring | Pending | Sprint 5 |
