import hashlib
import os
import re
import uuid
import time
import json
import logging
import contextvars
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header, Query, Request, Cookie
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import VaultDatabase, DatabaseUnavailableError, DatabaseIntegrityError
from auth_manager import AuthManager, get_configured_google_client_id
from ingestion_pipeline import DocumentIngestionPipeline, UnreadableDocumentError
from retrieval_pipeline import MultiTenantRetrievalPipeline
from agent_swarm import EnterprisePersonalAssistant

PORT = int(os.getenv("PORT", 10000))
CORRELATION_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="system")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "correlation_id": CORRELATION_ID_CTX.get(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


root_logger = logging.getLogger("aura")
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(JSONFormatter())
    root_logger.addHandler(_handler)

logger = logging.getLogger("aura.api")

ENABLE_SWAGGER = os.getenv("ENABLE_SWAGGER_DOCS", "false").lower() in ("1", "true", "yes")

app = FastAPI(
    title="AURA - Personal Document Assistant Vault",
    description="Multi-tenant document QA and verification platform with isolated user namespaces.",
    version="6.0.0",
    docs_url="/docs" if ENABLE_SWAGGER else None,
    redoc_url="/redoc" if ENABLE_SWAGGER else None,
    openapi_url="/openapi.json" if ENABLE_SWAGGER else None
)

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

CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://accounts.google.com https://apis.google.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' https: data: blob:; "
    "connect-src 'self' https://accounts.google.com https://oauth2.googleapis.com https://www.googleapis.com; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


@app.middleware("http")
async def security_and_correlation_middleware(request: Request, call_next):
    raw_corr = request.headers.get("X-Correlation-ID", "")
    if raw_corr and re.match(r'^[a-zA-Z0-9_\-]{1,64}$', raw_corr):
        corr_id = raw_corr
    else:
        corr_id = str(uuid.uuid4())

    token = CORRELATION_ID_CTX.set(corr_id)
    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), payment=(), usb=()"
        response.headers["Content-Security-Policy"] = CSP_POLICY
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
    finally:
        CORRELATION_ID_CTX.reset(token)


# --- Proxy-Aware Sliding-Window Rate Limiter (Fixes H7, N1, N10) ---
RATE_LIMIT_STORAGE: Dict[str, List[float]] = {}
MAX_RATE_LIMIT_KEYS = 5000


def get_real_client_ip(request: Optional[Request]) -> str:
    """
    Extracts non-spoofable client IP behind Render / Cloudflare reverse proxies (Fixes H7 & N1):
    Uses the rightmost hop of X-Forwarded-For (appended by the trusted ingress proxy) rather than
    hops[0] (which can be spoofed by a client sending a fake X-Forwarded-For header).
    """
    if not request:
        return "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
        if hops:
            return hops[-1][:64]
    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip[:64]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def check_rate_limit(identifier: str, action: str, max_requests: int = 30, window_seconds: int = 60):
    """
    Single-check DB-backed atomic rate limiter with bounded in-memory fallback (Fixes N1 & N10).
    """
    key = f"{identifier}:{action}"
    if "db" in globals() and hasattr(db, "check_and_increment_rate_limit"):
        try:
            allowed = db.check_and_increment_rate_limit(key, max_requests, window_seconds)
            if not allowed:
                logger.warning(f"Rate limit exceeded for key={key}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {action}. Please wait a moment before trying again."
                )
            return
        except HTTPException:
            raise
        except Exception:
            pass

    now = time.time()
    if len(RATE_LIMIT_STORAGE) > MAX_RATE_LIMIT_KEYS:
        stale_keys = [k for k, ts_list in RATE_LIMIT_STORAGE.items() if not ts_list or now - ts_list[-1] > 300]
        for k in stale_keys:
            RATE_LIMIT_STORAGE.pop(k, None)

    timestamps = RATE_LIMIT_STORAGE.get(key, [])
    valid_timestamps = [t for t in timestamps if now - t < window_seconds]
    if len(valid_timestamps) >= max_requests:
        logger.warning(f"In-memory rate limit exceeded for key={key}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {action}. Please wait a moment before trying again."
        )
    valid_timestamps.append(now)
    RATE_LIMIT_STORAGE[key] = valid_timestamps


# --- Database & Ingestion Exception Handlers (Fixes H5 & N4) ---
@app.exception_handler(DatabaseIntegrityError)
async def database_integrity_handler(request: Request, exc: DatabaseIntegrityError):
    logger.warning(f"DatabaseIntegrityError mapped to HTTP 409: {exc}")
    return JSONResponse(
        status_code=409,
        content={"detail": "Conflict or duplicate resource detected."}
    )


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    logger.error(f"DatabaseUnavailableError mapped to HTTP 503: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Database service is temporarily unavailable. Please retry in a moment."},
        headers={"Retry-After": "5"}
    )


@app.exception_handler(RequestValidationError)
async def safe_validation_exception_handler(request: Request, exc: RequestValidationError):
    safe_errors = []
    for err in exc.errors():
        safe_err = {
            "loc": err.get("loc"),
            "msg": clean_utf8(str(err.get("msg", "Invalid input"))),
            "type": err.get("type", "value_error")
        }
        safe_errors.append(safe_err)
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.exception_handler(UnreadableDocumentError)
async def unreadable_document_handler(request: Request, exc: UnreadableDocumentError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


# Global service instances
db = VaultDatabase()
auth = AuthManager(db)
ingestion = DocumentIngestionPipeline(chunk_size=180, chunk_overlap=25, max_chunk_size=250)
retrieval = MultiTenantRetrievalPipeline("./chroma_cloud_data", database=db)
assistant = EnterprisePersonalAssistant(retrieval, db)

# --- Upload Limits & Security Configuration ---
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB per file
MAX_TOTAL_REQUEST_SIZE = 50 * 1024 * 1024  # 50 MB aggregate batch limit
MAX_FILES_PER_REQUEST = 10
DEMO_MAX_DOCS = 3
DEMO_MAX_QUERIES = 10
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


# --- Validated Pydantic Request Models with max_length (Fixes H5 & H7) ---
class GoogleAuthRequest(BaseModel):
    id_token: str = Field(..., min_length=4, max_length=4096)
    nonce: Optional[str] = Field(None, max_length=256)


class ConversationCreate(BaseModel):
    title: str = Field("New Conversation", min_length=1, max_length=200)


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=2000)
    category: Optional[str] = Field("All", max_length=50)
    user_groq_key: Optional[str] = Field(None, max_length=256)


class DocumentCompareRequest(BaseModel):
    doc_ids: List[str] = Field(..., min_length=2, max_length=10)
    focus_topic: Optional[str] = Field(None, max_length=500)
    conversation_id: Optional[str] = Field(None, max_length=128)
    user_groq_key: Optional[str] = Field(None, max_length=256)


class MemoryUpdateRequest(BaseModel):
    fact_key: str = Field(..., min_length=1, max_length=100)
    fact_value: str = Field(..., min_length=1, max_length=2000)


class DocumentCategoryRequest(BaseModel):
    category: str = Field(..., min_length=1, max_length=50)


def clean_utf8(text: str) -> str:
    if not text:
        return ""
    return str(text).encode('utf-8', 'replace').decode('utf-8', 'replace')


# --- Authentication Dependency ---
def get_current_user(
    authorization: Optional[str] = Header(None),
    aura_session: Optional[str] = Cookie(None)
) -> dict:
    token = ""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
    elif aura_session:
        token = aura_session.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    payload = auth.verify_session_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in again.")
    user_id = payload.get("sub")
    user = db.get_user_by_id(user_id)
    if not user:
        if user_id and str(user_id).startswith("demo_"):
            try:
                retrieval.delete_user_vault(str(user_id))
            except Exception:
                pass
        raise HTTPException(status_code=401, detail="User account not found or demo session expired. Please sign in again.")
    return user


# --- Core Static & Public Routes ---
@app.get("/logo.png")
@app.get("/favicon.ico")
def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    return Response(status_code=204)


@app.api_route("/", methods=["GET", "HEAD"])
def get_ui(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)


# --- Authentication Routes ---
@app.get("/api/config")
def get_public_config():
    return {
        "google_client_id": get_configured_google_client_id(),
        "demo_mode_available": True,
        "version": "6.0.0"
    }


@app.post("/api/auth/google")
def authenticate_google(req: GoogleAuthRequest, request: Request):
    client_ip = get_real_client_ip(request)
    if req.id_token and req.id_token.strip().startswith("demo_"):
        check_rate_limit(client_ip, "demo_login", max_requests=10, window_seconds=60)
        try:
            expired_demos = db.cleanup_expired_demo_users(ttl_seconds=86400)
            for exp_uid in expired_demos:
                retrieval.delete_user_vault(exp_uid)
        except Exception as e:
            logger.warning(f"Demo cleanup notice: {e}")
    else:
        check_rate_limit(client_ip, "google_login", max_requests=20, window_seconds=60)

    profile = auth.verify_google_id_token(req.id_token, expected_nonce=req.nonce)
    if not profile:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed: Google token rejected (invalid signature, audience/nonce mismatch, or unverified email)."
        )

    user = db.upsert_google_user(
        google_id=profile["google_id"],
        email=profile["email"],
        full_name=profile["full_name"],
        avatar_url=profile["avatar_url"]
    )
    ttl = 86400 if user["id"].startswith("demo_") else 86400 * 7
    token = auth.create_session_token(user["id"], user["email"], expires_in_seconds=ttl)
    resp = JSONResponse(content={
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "avatar_url": user["avatar_url"]
        }
    })
    resp.set_cookie(
        key="aura_session",
        value=token,
        max_age=ttl,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/"
    )
    return resp


@app.post("/api/auth/logout")
def logout_session(
    authorization: Optional[str] = Header(None),
    aura_session: Optional[str] = Cookie(None)
):
    if authorization:
        auth.revoke_session_token(authorization.replace("Bearer ", "").strip())
    if aura_session:
        auth.revoke_session_token(aura_session.strip())
    resp = JSONResponse(content={"message": "Session token revoked and persisted."})
    resp.delete_cookie(key="aura_session", path="/")
    return resp


@app.post("/api/auth/delete-account")
def delete_account(
    authorization: Optional[str] = Header(None),
    aura_session: Optional[str] = Cookie(None),
    user: dict = Depends(get_current_user)
):
    """
    Permanently deletes user account across all 9 relational tables, revokes both header and cookie tokens,
    clears the aura_session cookie, and wipes the user's vector store collection (Fixes N5 & N8).
    """
    user_id = user["id"]
    if authorization:
        auth.revoke_session_token(authorization.replace("Bearer ", "").strip())
    if aura_session:
        auth.revoke_session_token(aura_session.strip())
    vec_ok = retrieval.delete_user_vault(user_id)
    db_ok = db.delete_user_account(user_id)
    payload = {
        "message": (
            "User account, conversation history, and document vault permanently wiped."
            if vec_ok else
            "Relational account and document records wiped; vector store cleanup scheduled."
        ),
        "success": db_ok,
        "vector_cleanup_pending": not vec_ok
    }
    resp = JSONResponse(content=payload)
    resp.delete_cookie(key="aura_session", path="/")
    return resp


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
    return db.get_user_documents(user["id"], category=category)


def _process_and_store_single_document(
    user_id: str, filename: str, file_ext: str, content_bytes: bytes, extract_memory: bool
) -> Dict[str, Any]:
    """
    Executes synchronous CPU extraction, chunking, DB transaction (parent row FIRST, then chunks),
    and vector indexing inside a worker thread so the async event loop is never blocked (Fixes H6, N4, N12).
    """
    file_hash = hashlib.sha256(content_bytes).hexdigest()
    existing_doc = db.get_document_by_hash(user_id, file_hash)
    if existing_doc:
        return {
            "result": {
                "filename": filename,
                "status": "duplicate",
                "doc_id": existing_doc["id"],
                "message": f"Document already in vault as '{existing_doc['title']}'."
            },
            "doc_record": None
        }

    raw_text = ingestion.extract_text_from_bytes(content_bytes, filename)
    if not raw_text.strip():
        return {
            "result": {"filename": filename, "status": "failed", "error": "Document contains empty text."},
            "doc_record": None
        }

    doc_id = f"doc_{uuid.uuid4().hex}"
    clean_title = clean_utf8(filename)[:250]

    highlights = assistant.extract_document_highlights(raw_text, clean_title)
    category = highlights.get("category", "General")

    chunks = ingestion.recursive_chunk_text(
        raw_text,
        doc_id=doc_id,
        doc_title=clean_title,
        collection=category,
        tags=[category]
    )

    if not chunks:
        return {
            "result": {"filename": filename, "status": "failed", "error": "No valid chunks generated."},
            "doc_record": None
        }

    try:
        doc_record = db.add_document_with_chunks(
            user_id=user_id,
            doc_id=doc_id,
            title=clean_title,
            original_filename=clean_title,
            file_type=file_ext.replace(".", ""),
            file_size_bytes=len(content_bytes),
            chunks=chunks,
            category=category,
            summary=highlights,
            file_hash=file_hash
        )
    except DatabaseIntegrityError:
        existing_race = db.get_document_by_hash(user_id, file_hash)
        return {
            "result": {
                "filename": filename,
                "status": "duplicate",
                "doc_id": existing_race["id"] if existing_race else doc_id,
                "message": "Document already uploaded in concurrent request."
            },
            "doc_record": None
        }

    indexed_count = retrieval.index_chunks(chunks, user_id=user_id, category=category)
    if indexed_count == 0 and len(chunks) > 0:
        db.delete_document(user_id, doc_id)
        return {
            "result": {"filename": filename, "status": "failed", "error": "Vector store indexing failed; upload rolled back."},
            "doc_record": None
        }

    if extract_memory:
        for ent in highlights.get("key_entities", []):
            db.upsert_user_memory(
                user_id=user_id,
                category=category,
                fact_key=ent["label"],
                fact_value=ent["value"],
                source_doc=clean_title,
                status="pending",
                source_doc_id=doc_id
            )

    return {
        "result": {
            "filename": filename,
            "status": "indexed",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "category": category
        },
        "doc_record": doc_record
    }


@app.post("/api/documents/upload")
async def upload_document(
    files: List[UploadFile] = File(...),
    extract_memory: bool = Form(False),
    user: dict = Depends(get_current_user)
):
    check_rate_limit(user["id"], "upload", max_requests=10, window_seconds=60)
    if user["id"].startswith("demo_"):
        existing_docs = await run_in_threadpool(db.get_user_documents, user["id"])
        if len(existing_docs) + len(files) > DEMO_MAX_DOCS:
            raise HTTPException(
                status_code=403,
                detail=f"Demo account limit: maximum {DEMO_MAX_DOCS} documents allowed. Sign in with Google for full vault capacity."
            )
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Exceeded maximum limit of {MAX_FILES_PER_REQUEST} files per upload.")

    saved_docs = []
    results = []
    total_batch_bytes = 0

    for file in files:
        filename = clean_utf8(file.filename or "uploaded_file.txt")[:250]
        file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if file_ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": filename, "status": "failed", "error": f"Extension '{file_ext}' not supported."})
            continue

        try:
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

            if file_ext == ".pdf" and not content_bytes.startswith(b"%PDF-"):
                results.append({"filename": filename, "status": "failed", "error": "Invalid PDF magic signature."})
                continue
            elif file_ext == ".docx" and not content_bytes.startswith(b"PK\x03\x04"):
                results.append({"filename": filename, "status": "failed", "error": "Invalid Word document signature."})
                continue

            outcome = await run_in_threadpool(
                _process_and_store_single_document,
                user["id"],
                filename,
                file_ext,
                bytes(content_bytes),
                extract_memory
            )
            results.append(outcome["result"])
            if outcome["doc_record"]:
                saved_docs.append(outcome["doc_record"])

        except HTTPException:
            raise
        except (DatabaseUnavailableError, DatabaseIntegrityError):
            raise
        except UnreadableDocumentError as ue:
            results.append({"filename": filename, "status": "failed", "error": str(ue)})
        except Exception as e:
            logger.error(f"Upload processing error for '{filename}': {e}")
            results.append({"filename": filename, "status": "failed", "error": f"Processing error: {e}"})

    job_id = await run_in_threadpool(db.create_upload_job, user["id"], len(files))
    final_status = "completed" if saved_docs or all(r.get("status") == "duplicate" for r in results) else "failed"
    await run_in_threadpool(db.update_upload_job, job_id, user["id"], len(results), results, final_status)

    if not saved_docs and any(r.get("status") == "failed" for r in results):
        first_err = next(r.get("error") for r in results if r.get("status") == "failed")
        raise HTTPException(status_code=422, detail=first_err)

    return {"job_id": job_id, "uploaded_count": len(saved_docs), "results": results, "documents": saved_docs}


@app.get("/api/documents/upload-status/{job_id}")
def get_upload_job_status(job_id: str, user: dict = Depends(get_current_user)):
    job = db.get_upload_job(user["id"], job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    return job


@app.put("/api/documents/{doc_id}/category")
def update_document_category_route(doc_id: str, req: DocumentCategoryRequest, user: dict = Depends(get_current_user)):
    doc = db.get_document(user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or unauthorized.")
    success = db.update_document_category(user["id"], doc_id, req.category)
    return {"message": "Document category updated.", "category": req.category, "success": success}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = db.get_document(user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or unauthorized.")

    chunks = db.get_document_full_chunks(user["id"], doc_id)
    chunk_ids = [c["chunk_id"] for c in chunks if c.get("chunk_id")]

    vec_ok = retrieval.delete_document(user["id"], doc_id, chunk_ids=chunk_ids)
    db_ok = db.delete_document(user["id"], doc_id)

    if not vec_ok:
        return {
            "message": "Document relational record and chunks deleted, but vector provider returned error. Vector cleanup scheduled.",
            "success": db_ok,
            "vector_cleanup_pending": True
        }
    return {
        "message": "Document, full-text chunks, and vector embeddings permanently deleted.",
        "success": db_ok,
        "vector_cleanup_pending": False
    }


# --- Document Comparison Route (Fixes H7, M2, N10) ---
@app.post("/api/documents/compare")
def compare_documents_route(req: DocumentCompareRequest, user: dict = Depends(get_current_user)):
    check_rate_limit(user["id"], "compare", max_requests=10, window_seconds=60)

    if user["id"].startswith("demo_"):
        user_queries = db.count_user_messages(user["id"])
        if user_queries >= DEMO_MAX_QUERIES:
            raise HTTPException(
                status_code=403,
                detail=f"Demo session limit reached ({DEMO_MAX_QUERIES} queries). Sign in with Google for full access."
            )

    if not req.doc_ids or len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Please select at least 2 documents to compare.")

    for d_id in req.doc_ids:
        doc = db.get_document(user["id"], d_id)
        if not doc:
            raise HTTPException(status_code=403, detail=f"Unauthorized access to document {d_id}.")

    conv_id = req.conversation_id
    if conv_id:
        conv = db.get_conversation(user["id"], conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
    else:
        new_conv = db.create_conversation(user["id"], title="Side-by-Side Comparison")
        conv_id = new_conv["id"]

    db.add_message(conv_id, sender="user", content=f"Compare documents ({len(req.doc_ids)} selected)" + (f" focusing on: {req.focus_topic}" if req.focus_topic else ""))

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


# --- Chat & RAG Execution Route ---
@app.post("/api/chat")
def chat_with_assistant(req: ChatRequest, user: dict = Depends(get_current_user)):
    check_rate_limit(user["id"], "chat", max_requests=25, window_seconds=60)

    conv = db.get_conversation(user["id"], req.conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")

    if user["id"].startswith("demo_"):
        user_queries = db.count_user_messages(user["id"])
        if user_queries >= DEMO_MAX_QUERIES:
            return {
                "answer": f"You have reached the demo limit ({DEMO_MAX_QUERIES} questions). Sign in with your Google account for full access.",
                "citations": [],
                "is_grounded": True,
                "verification_flags": []
            }

    res = assistant.respond(
        user_id=user["id"],
        conversation_id=req.conversation_id,
        query=clean_utf8(req.query),
        user_groq_key=req.user_groq_key,
        category_filter=req.category
    )

    messages = db.get_conversation_messages(user["id"], req.conversation_id)
    if len(messages) <= 2:
        short_title = clean_utf8(req.query)[:32] + "..." if len(req.query) > 32 else req.query
        db.update_conversation_title(user["id"], req.conversation_id, short_title)

    return res


# --- Assistant Memory Lifecycle Routes ---
@app.get("/api/memory")
def get_user_memory(user: dict = Depends(get_current_user)):
    return db.get_user_memories(user["id"])


@app.post("/api/memory/{memory_id}/confirm")
def confirm_user_memory_route(memory_id: str, user: dict = Depends(get_current_user)):
    success = db.confirm_user_memory(user["id"], memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory fact not found.")
    return {"message": "Memory fact confirmed and active.", "status": "confirmed"}


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


# --- Operational Readiness & Health Checks (Fixes H8) ---
@app.get("/health/live")
def health_live():
    return {"status": "alive", "version": "6.0.0", "timestamp": time.time()}


@app.get("/health/ready")
def health_ready():
    db_ok = False
    try:
        row = db._execute_scalar("SELECT 1 as ping")
        db_ok = bool(row)
    except Exception:
        db_ok = False

    vec_ok, vec_provider = retrieval.check_vector_health()

    if not db_ok or not vec_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "database_ok": db_ok,
                "vector_ok": vec_ok,
                "vector_provider": vec_provider
            }
        )

    return {
        "status": "ready",
        "version": "6.0.0",
        "database": "postgres" if db.is_postgres else "sqlite",
        "vector_provider": vec_provider,
        "vector_ready": vec_ok,
        "llm_configured": bool(assistant.groq_api_key)
    }


@app.get("/health")
def health():
    vec_ok, vec_provider = retrieval.check_vector_health()
    return {
        "status": "healthy" if vec_ok else "degraded",
        "version": "6.0.0",
        "database": "postgres" if db.is_postgres else "sqlite",
        "vector_provider": vec_provider,
        "llm_configured": bool(assistant.groq_api_key)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)