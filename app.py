import hashlib
import os
import uuid
import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import VaultDatabase, DatabaseUnavailableError
from auth_manager import AuthManager, get_configured_google_client_id
from ingestion_pipeline import DocumentIngestionPipeline, UnreadableDocumentError
from retrieval_pipeline import MultiTenantRetrievalPipeline
from agent_swarm import EnterprisePersonalAssistant

PORT = int(os.getenv("PORT", 10000))

app = FastAPI(
    title="AURA - Enterprise Multi-Tenant Personal AI Assistant",
    description="Secure, private, user-isolated document intelligence vault.",
    version="6.0.0"
)

import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logger = logging.getLogger("aura")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(JSONFormatter())
    logger.addHandler(_h)

ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://cloudrag-agent-chatbot.onrender.com,http://localhost:10000,http://127.0.0.1:10000"
    ).split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_and_correlation_middleware(request, call_next):
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

from fastapi.responses import JSONResponse

# --- In-Memory Rate Limiting (Resolves H2 & P1-07) ---
RATE_LIMIT_STORAGE = {}

def check_rate_limit(client_ip: str, action: str, max_requests: int = 30, window_seconds: int = 60):
    now = time.time()
    key = f"{client_ip}:{action}"
    timestamps = RATE_LIMIT_STORAGE.get(key, [])
    # Filter out timestamps outside window
    valid_timestamps = [t for t in timestamps if now - t < window_seconds]
    if len(valid_timestamps) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {action}. Please wait a moment before trying again."
        )
    valid_timestamps.append(now)
    RATE_LIMIT_STORAGE[key] = valid_timestamps

# --- Database & Ingestion Exception Handlers (Resolves H8 & H3) ---
@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Database infrastructure is temporarily unavailable. Please retry in a moment."},
        headers={"Retry-After": "5"}
    )

@app.exception_handler(UnreadableDocumentError)
async def unreadable_document_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

# Global service instances
db = VaultDatabase()
auth = AuthManager(db)
ingestion = DocumentIngestionPipeline(chunk_size=180, chunk_overlap=25)
retrieval = MultiTenantRetrievalPipeline("./chroma_cloud_data", database=db)
assistant = EnterprisePersonalAssistant(retrieval, db)

# --- Upload Limits & Security Configuration (Resolves P0 Finding 3) ---
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB per file
MAX_TOTAL_REQUEST_SIZE = 50 * 1024 * 1024  # 50 MB aggregate batch limit (Resolves Task 2.2)
MAX_FILES_PER_REQUEST = 10
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

# --- Pydantic Request Models ---
class GoogleAuthRequest(BaseModel):
    id_token: str

class ConversationCreate(BaseModel):
    title: str = "New Conversation"

class ChatRequest(BaseModel):
    conversation_id: str
    query: str
    category: Optional[str] = "All"
    user_groq_key: Optional[str] = None

class DocumentCompareRequest(BaseModel):
    doc_ids: List[str]
    focus_topic: Optional[str] = None
    conversation_id: Optional[str] = None
    user_groq_key: Optional[str] = None

class MemoryUpdateRequest(BaseModel):
    fact_key: str
    fact_value: str

def clean_utf8(text: str) -> str:
    if not text:
        return ""
    return text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')

# --- Authentication Dependency ---
def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    token = authorization.replace("Bearer ", "").strip()
    payload = auth.verify_session_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in again.")
    user_id = payload.get("sub")
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User account not found. Please sign in again.")
    return user

# --- Core Static & Public Routes ---
@app.get("/logo.png")
def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    return Response(status_code=404)

@app.get("/")
def get_ui():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)

# --- Authentication Routes ---
@app.get("/api/config")
def get_public_config():
    return {
        "google_client_id": get_configured_google_client_id(),
        "demo_mode_available": True
    }

@app.post("/api/auth/google")
def authenticate_google(req: GoogleAuthRequest):
    profile = auth.verify_google_id_token(req.id_token)
    if not profile:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed: Google token rejected (invalid signature, audience mismatch, or unverified email)."
        )
    
    user = db.upsert_google_user(
        google_id=profile["google_id"],
        email=profile["email"],
        full_name=profile["full_name"],
        avatar_url=profile["avatar_url"]
    )
    token = auth.create_session_token(user["id"], user["email"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "avatar_url": user["avatar_url"]
        }
    }

@app.post("/api/auth/delete-account")
def delete_account(user: dict = Depends(get_current_user)):
    user_id = user["id"]
    retrieval.delete_user_vault(user_id)
    db.delete_user_account(user_id)
    return {"message": "User account, conversation history, and private document vault permanently wiped."}

# --- Conversation Routes ---
@app.get("/api/conversations")
def list_conversations(user: dict = Depends(get_current_user)):
    return db.get_user_conversations(user["id"])

@app.post("/api/conversations")
def create_conversation(req: ConversationCreate, user: dict = Depends(get_current_user)):
    conv = db.create_conversation(user["id"], title=clean_utf8(req.title))
    return conv

@app.get("/api/conversations/{conv_id}/messages")
def get_messages(conv_id: str, user: dict = Depends(get_current_user)):
    conv = db.get_conversation(user["id"], conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
    return db.get_conversation_messages(user["id"], conv_id)

@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    success = db.delete_conversation(user["id"], conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": "Conversation deleted."}

# --- Document Vault Routes ---
@app.get("/api/documents")
def list_documents(category: Optional[str] = Query(None), user: dict = Depends(get_current_user)):
    # 100% Zero-Leakage: returns only documents belonging to user_id
    return db.get_user_documents(user["id"], category=category)

@app.post("/api/documents/upload")
async def upload_document(
    files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user)
):
    check_rate_limit(user["id"], "upload", max_requests=10, window_seconds=60)
    # Demo quota enforcement (Resolves H2)
    if user["id"].startswith("demo_"):
        existing_docs = db.get_user_documents(user["id"])
        if len(existing_docs) + len(files) > 3:
            raise HTTPException(
                status_code=403,
                detail="Demo account limit: maximum 3 documents allowed. Sign in with Google for unlimited document storage."
            )
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Exceeded maximum limit of {MAX_FILES_PER_REQUEST} files per upload.")

    saved_docs = []
    results = []
    total_batch_bytes = 0

    for file in files:
        filename = clean_utf8(file.filename or "uploaded_file.txt")
        file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        
        # Extension allowlisting (P0 Finding 3)
        if file_ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": filename, "status": "failed", "error": f"Extension '{file_ext}' not supported."})
            continue

        try:
            # Streamed chunked reading with individual and aggregate caps (Resolves P1-07 & Task 2.2)
            content_bytes = bytearray()
            while True:
                chunk_slice = await file.read(65536)
                if not chunk_slice:
                    break
                content_bytes.extend(chunk_slice)
                total_batch_bytes += len(chunk_slice)
                if len(content_bytes) > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"File '{filename}' exceeds maximum allowed size of 15MB.")
                if total_batch_bytes > MAX_TOTAL_REQUEST_SIZE:
                    raise HTTPException(status_code=413, detail="Batch upload exceeds maximum aggregate limit of 50MB.")

            # MIME / Magic Signature Verification
            if file_ext == ".pdf" and not content_bytes.startswith(b"%PDF-"):
                results.append({"filename": filename, "status": "failed", "error": "Invalid PDF magic signature."})
                continue
            elif file_ext == ".docx" and not content_bytes.startswith(b"PK\x03\x04"):
                results.append({"filename": filename, "status": "failed", "error": "Invalid Word document signature."})
                continue

            # Deduplication Check by SHA-256 Hash (Resolves Task 2.7)
            file_hash = hashlib.sha256(content_bytes).hexdigest()
            existing_doc = db.get_document_by_hash(user["id"], file_hash)
            if existing_doc:
                results.append({
                    "filename": filename,
                    "status": "duplicate",
                    "doc_id": existing_doc["id"],
                    "message": f"Document already in vault as '{existing_doc['title']}'."
                })
                continue

            # Text Extraction with Strict OCR / Scanned Detection (Resolves H3 & Task 2.1)
            raw_text = ingestion.extract_text_from_bytes(content_bytes, filename)
            if not raw_text.strip():
                results.append({"filename": filename, "status": "failed", "error": "Document contains empty text."})
                continue

            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            clean_title = clean_utf8(filename)

            # Metadata Highlights Extraction with Sensitive Number Exclusion (Resolves H4 & Task 2.3)
            highlights = assistant.extract_document_highlights(raw_text, clean_title)
            category = highlights.get("category", "General")

            # Structure-Aware Chunking: 180 words, atomic table rows, single overlap (Resolves Task 2.4 & 2.5)
            chunks = ingestion.recursive_chunk_text(
                raw_text,
                doc_id=doc_id,
                doc_title=clean_title,
                collection=category,
                tags=[category]
            )

            # 1. Store Full Chunks in PostgreSQL / SQLite
            db.store_document_chunks(user["id"], doc_id, chunks)

            # 2. Index Vectors into Isolated Pinecone Namespace
            indexed_count = retrieval.index_chunks(chunks, user_id=user["id"], category=category)
            if indexed_count == 0 and len(chunks) > 0:
                # Rollback stored chunks and fail closed (Resolves P0-04)
                db._execute_non_query("DELETE FROM document_chunks WHERE user_id = ? AND doc_id = ?", (user["id"], doc_id))
                results.append({"filename": filename, "status": "failed", "error": "Vector indexing failed."})
                continue

            # 3. Save Document Metadata with SHA-256 file_hash
            doc_record = db.add_document(
                user_id=user["id"],
                doc_id=doc_id,
                title=clean_title,
                original_filename=clean_title,
                file_type=file_ext.replace(".", ""),
                file_size_bytes=len(content_bytes),
                chunk_count=len(chunks),
                category=category,
                summary=highlights,
                file_hash=file_hash
            )
            saved_docs.append(doc_record)
            results.append({
                "filename": filename,
                "status": "indexed",
                "doc_id": doc_id,
                "chunks": len(chunks),
                "category": category
            })

            # Extract user-scoped memory facts from genuine policy identifiers
            for ent in highlights.get("key_entities", []):
                db.upsert_user_memory(
                    user_id=user["id"],
                    category=category,
                    fact_key=ent["label"],
                    fact_value=ent["value"],
                    source_doc=clean_title
                )

        except HTTPException:
            raise
        except UnreadableDocumentError as ue:
            results.append({"filename": filename, "status": "failed", "error": str(ue)})
        except Exception as e:
            print(f"[Upload Processing Error]: {e}")
            results.append({"filename": filename, "status": "failed", "error": f"Internal processing error: {e}"})

    # If all files in the batch failed with unreadable errors, raise HTTP 422
    if not saved_docs and any(r.get("status") == "failed" for r in results):
        first_err = next(r.get("error") for r in results if r.get("status") == "failed")
        raise HTTPException(status_code=422, detail=first_err)

    return {"uploaded_count": len(saved_docs), "results": results, "documents": saved_docs}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    # Verify ownership before deletion
    doc = db.get_document(user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or unauthorized.")

    # Execute vector and relational deletions with verified status (Resolves P0-01)
    vec_ok = retrieval.delete_document(user["id"], doc_id)
    db_ok = db.delete_document(user["id"], doc_id)
    
    if not vec_ok:
        return {
            "message": "Document relational record and chunks deleted, but vector provider returned error. Vector cleanup scheduled.",
            "success": db_ok,
            "vector_cleanup_pending": True
        }
    return {"message": "Document, full-text chunks, and vector embeddings permanently deleted.", "success": db_ok}

# --- Document Comparison Route (Resolves P0 Authorization Gap) ---
@app.post("/api/documents/compare")
def compare_documents_route(req: DocumentCompareRequest, user: dict = Depends(get_current_user)):
    if not req.doc_ids or len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Please select at least 2 documents to compare.")

    # 1. Strict Authorization: Verify all requested documents belong to this user (P0 Authorization)
    for d_id in req.doc_ids:
        doc = db.get_document(user["id"], d_id)
        if not doc:
            raise HTTPException(status_code=403, detail=f"Unauthorized access to document {d_id}.")

    # 2. Strict Authorization: Verify conversation_id if provided (P0 Authorization Gap Fixed)
    conv_id = req.conversation_id
    if conv_id:
        conv = db.get_conversation(user["id"], conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
    else:
        # Create a dedicated comparison conversation
        new_conv = db.create_conversation(user["id"], title="Side-by-Side Comparison")
        conv_id = new_conv["id"]

    # 3. Execute Independent Document Comparison
    result = assistant.compare_documents(
        user_id=user["id"],
        doc_ids=req.doc_ids,
        focus_topic=req.focus_topic,
        user_groq_key=req.user_groq_key
    )

    db.add_message(conv_id, sender="assistant", content=result["answer"])
    return {
        "conversation_id": conv_id,
        "answer": result["answer"],
        "docs": result.get("docs", [])
    }

# --- Assistant Chat Route ---
@app.post("/api/chat")
def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    # Rate limit and demo account caps (Resolves H2)
    check_rate_limit(user["id"], "chat", max_requests=40, window_seconds=60)
    if user["id"].startswith("demo_"):
        convs = db.get_user_conversations(user["id"])
        total_msgs = sum(len(db.get_conversation_messages(user["id"], c["id"])) for c in convs)
        if total_msgs >= 20:
            raise HTTPException(
                status_code=403,
                detail="Demo session limit reached (20 messages). Sign in with your Google account for unlimited assistant access."
            )
    # Verify conversation ownership
    conv = db.get_conversation(user["id"], req.conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")

    res = assistant.respond(
        user_id=user["id"],
        conversation_id=req.conversation_id,
        query=clean_utf8(req.query),
        user_groq_key=req.user_groq_key,
        category_filter=req.category
    )

    # Auto-title conversation on first message
    messages = db.get_conversation_messages(user["id"], req.conversation_id)
    if len(messages) <= 2:
        short_title = clean_utf8(req.query)[:32] + "..." if len(req.query) > 32 else req.query
        db.update_conversation_title(user["id"], req.conversation_id, short_title)

    return res

# --- Assistant Memory Lifecycle Routes (Resolves Finding 8) ---
@app.get("/api/memory")
def get_user_memory(user: dict = Depends(get_current_user)):
    return db.get_user_memories(user["id"])

@app.put("/api/memory/{memory_id}")
def update_user_memory(memory_id: str, req: MemoryUpdateRequest, user: dict = Depends(get_current_user)):
    success = db.update_user_memory(user["id"], memory_id, req.fact_key, req.fact_value)
    if not success:
        raise HTTPException(status_code=404, detail="Memory fact not found or unauthorized.")
    return {"message": "Memory fact updated."}

@app.delete("/api/memory/{memory_id}")
def delete_user_memory(memory_id: str, user: dict = Depends(get_current_user)):
    success = db.delete_user_memory(user["id"], memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory fact not found.")
    return {"message": "Memory fact deleted."}

@app.delete("/api/memory")
def clear_all_memory(user: dict = Depends(get_current_user)):
    db.clear_user_memories(user["id"])
    return {"message": "All personal memory facts cleared."}

# --- Operational Readiness & Health Checks ---
@app.get("/health/live")
def health_live():
    return {"status": "alive", "timestamp": time.time()}

@app.get("/health/ready")
def health_ready():
    # Verify database readiness
    db_ok = False
    try:
        row = db._execute_scalar("SELECT 1 as ping")
        db_ok = bool(row)
    except Exception:
        db_ok = False

    if not db_ok:
        raise HTTPException(status_code=503, detail="Database connection pool unavailable.")

    return {
        "status": "ready",
        "database": "postgres" if db.is_postgres else "sqlite",
        "vector_provider": retrieval.mode,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)