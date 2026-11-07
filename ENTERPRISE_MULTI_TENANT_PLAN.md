# Enterprise Multi-Tenant Personal AI Assistant Architecture & Roadmap

## 1. Executive Summary & Vision
Transform the single-tenant prototype into a secure, consumer-friendly **Enterprise Personal AI Assistant**.
The platform operates as a private digital vault where individual users upload sensitive personal documents (insurance policies, resumes, medical bills, legal contracts, property documents) and converse with an AI agent.

### Core Tenets
1. **Zero Cross-Tenant Leakage**: Strict physical and logical data isolation. No user can ever query, search, view, or delete another user's files or chats.
2. **Zero Technical Jargon in UI**: Consumer-grade simplicity (reminiscent of Apple Intelligence / ChatGPT). Developer telemetry (ONNX, ChromaDB, HNSW, cosine space, Swarm DAG nodes, raw chunk IDs) is entirely hidden.
3. **100% Free-Tier & 24/7 Cloud Resilient**: Zero infrastructure costs using best-of-breed free cloud tiers (Google OAuth, Supabase/Neon PostgreSQL, Pinecone Serverless Namespaces, Groq LPU, and Render hosting).

---

## 2. Technology Stack & Free-Tier Selection

| Layer | Selected Service / Tool | Free Tier Quota & Fit | Why Selected |
| :--- | :--- | :--- | :--- |
| **Authentication** | **Google OAuth 2.0 (OpenID Connect)** | 100% Free forever (Google Cloud Console OAuth Client) | No password leaks, instant 1-tap sign-in on mobile and desktop, cryptographically verified email identity (`sub`, `email`, `name`, `picture`). |
| **Database** | **Supabase (Managed PostgreSQL)** or **Neon PostgreSQL** | Supabase: 500MB DB, 50,000 monthly active users free.<br>Neon: 0.5 GB storage, serverless autoscaling. | True cloud persistence (survives all Render restarts), built-in Row-Level Security (RLS) policies, structured relational queries for users, documents, and chat sessions. |
| **Vector Engine** | **Pinecone Serverless Cloud** | 100,000 vectors free, 2GB storage, permanent 24/7. | Native **Namespaces** (`namespace=f"user_{user_id}"`) provide hardware-isolated search boundaries per user within a single index. |
| **LLM Inference** | **Groq LPU Cloud** | Free tier (up to 30 requests/min, 14,400 req/day). | Ultra-fast inference (sub-second response) using `llama-3.3-70b-versatile` and `openai/gpt-oss-20b`. |
| **Hosting & Compute** | **Render Web Service** | Free 750 hours/month (FastAPI on Port 10000). | Runs 24/7, automatic GitHub CI/CD deployment on every commit. |

---

## 3. Security & Multi-Tenant Data Isolation Model

```
                    ┌──────────────────────────────────────────────┐
                    │               Client Browser / Phone         │
                    └──────────────────────┬───────────────────────┘
                                           │ Google OAuth Bearer JWT
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │            FastAPI Authentication Guard      │
                    │        (Extracts validated user_id: UUID)    │
                    └──────────────┬───────────────────────────────┘
                                   │
             ┌─────────────────────┴──────────────────────┐
             ▼                                            ▼
┌───────────────────────────────┐            ┌────────────────────────────────┐
│   PostgreSQL (Supabase/Neon)  │            │     Pinecone Serverless Cloud  │
├───────────────────────────────┤            ├────────────────────────────────┤
│ WHERE user_id = :user_id      │            │ namespace = f"user_{user_id}"  │
│ (Users, Docs, Chats, Messages)│            │ (Strictly isolated vector space│
│ Zero cross-user visibility    │            │  zero cross-tenant search)     │
└───────────────────────────────┘            └────────────────────────────────┘
```

### 3.1 Strict Isolation Rules
1. **Pinecone Namespaces**:
   - Every chunk upserted: `index.upsert(vectors=batch, namespace=f"user_{user_id}")`.
   - Every similarity query: `index.query(vector=q_emb, top_k=5, namespace=f"user_{user_id}")`.
   - Result: Pinecone's indexing engine restricts nearest-neighbor lookups strictly to that user's partition.
2. **Database Row-Level Isolation**:
   - Every database query in FastAPI must include `WHERE user_id = current_user.id`.
   - Foreign key cascading deletes: Deleting an account or a document automatically purges all corresponding chunks, messages, and citations.
3. **Session Verification**:
   - Google ID Tokens (`id_token`) verified on the backend via `google-auth` (`google.oauth2.id_token.verify_oauth2_token`).
   - Short-lived application JWTs issued for rapid API authentication.

---

## 4. Database Schema Specification

```sql
-- 1. Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    google_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Documents Vault Table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL, -- pdf, docx, txt, md
    file_size_bytes INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_documents_user ON documents(user_id);

-- 3. Conversations / Chat Sessions Table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_conversations_user ON conversations(user_id);

-- 4. Messages Table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender VARCHAR(20) NOT NULL CHECK (sender IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations_json JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
```

---

## 5. End-User Interface Experience (Zero-Jargon)

### 5.1 Removed Telemetry & Technical Noise
- ❌ Remove: ChromaDB / Pinecone backend switcher, ONNX badges, vector dimensions, chunk overlap sliders.
- ❌ Remove: Swarm DAG execution graph, raw chunk IDs (e.g. `#doc_dbed5694_c6`), cosine similarity scores (`64%`).
- ❌ Remove: JSON debug panels, temperature parameters, system prompts.

### 5.2 Added Consumer Features
- ✅ **One-Tap Google Sign-In**: Clean overlay screen: *"Welcome to Your Personal Vault. Sign in with Google to access your private assistant."*
- ✅ **Sidebar Conversation History**: Left drawer showing past chat topics ("Auto Insurance Coverage", "Resume Review", "Property Tax Query") with rename and delete options.
- ✅ **Private Vault Tab ("My Files")**:
  - Drag-and-drop zone with instant PDF / Word document preview.
  - Document cards with document title, upload timestamp, page count, and a friendly status: **"Indexed & Ready"**.
  - A privacy guarantee badge: *"🔒 Protected by User-Isolated Cloud Encryption. Only accessible by your Google account."*
- ✅ **Human-Centric Citation Chips**:
  - Instead of `#doc_6b1522bd_c2`, citations display as clean clickable pills: `📄 Zurich Car Policy (Page 1)` or `📄 Mohamed Resume (Skills)`.
  - Tapping a citation opens a clean drawer displaying the cited passage.
- ✅ **Account Controls**:
  - User avatar and email display in bottom-left corner.
  - Storage meter (e.g., *"2 of 20 documents used"*).
  - One-click **Sign Out** and **"Delete All My Documents"** (GDPR compliance).

---

## 6. Implementation Milestones

```
Milestone 1: Auth & Data Foundation
├── Setup Supabase / Neon PostgreSQL connection pool
├── Implement Google OAuth ID token verification endpoint
└── Create user session management and JWT issuing

Milestone 2: Multi-Tenant Vector & Ingestion Pipeline
├── Refactor ingestion_pipeline to tag user_id on every chunk
├── Refactor retrieval_pipeline to route all Pinecone calls through namespace=f"user_{user_id}"
└── Implement isolated document upload and cascade-delete endpoints

Milestone 3: Conversational Memory & Multi-Chat Backend
├── Create /api/conversations CRUD endpoints
├── Implement message history loading and context assembly
└── Bind Groq LPU LLM with user-isolated RAG context

Milestone 4: Modern Zero-Jargon UI Redesign
├── Build responsive Apple/ChatGPT-style glassmorphic interface
├── Integrate Google Identity Services SDK (GSI button)
├── Build Vault file manager and sidebar conversation drawer
└── Implement friendly citation pills and source viewer

Milestone 5: Production Deployment & Verification
├── Deploy database migrations to Supabase/Neon
├── Configure Render environment variables (GOOGLE_CLIENT_ID, SUPABASE_DB_URL, PINECONE_API_KEY, GROQ_API_KEY)
└── Multi-user end-to-end verification (test User A cannot see or search User B's documents)
```

---

## 7. Change Log & Future Revisions
*This section will be updated incrementally as features are implemented.*

- **v1.0.0 (Current Base)**: Single-tenant prototype with Pinecone cloud backup and local fallback.
- **v2.0.0 (Planned)**: Full multi-tenant enterprise overhaul with Google OAuth, PostgreSQL database, Pinecone namespaces, and conversational memory.
