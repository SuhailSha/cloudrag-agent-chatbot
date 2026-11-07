import os
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from database import VaultDatabase, clean_utf8
from auth_manager import AuthManager
from ingestion_pipeline import DocumentIngestionPipeline
from retrieval_pipeline import MultiTenantRetrievalPipeline
from agent_swarm import EnterprisePersonalAssistant

PORT = int(os.getenv("PORT", 10000))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

app = FastAPI(
    title="Enterprise Multi-Tenant Personal AI Assistant",
    description="Secure, private, user-isolated document intelligence platform.",
    version="4.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
db = VaultDatabase()
auth = AuthManager(db)
ingestion = DocumentIngestionPipeline(chunk_size=350, chunk_overlap=40)
retrieval = MultiTenantRetrievalPipeline(db_path="./chroma_cloud_data")
assistant = EnterprisePersonalAssistant(retrieval, db)

def make_safe_title(text: str, max_chars: int = 40) -> str:
    cleaned = clean_utf8(text)
    single_line = " ".join(cleaned.split())
    return single_line[:max_chars] if single_line else "New Conversation"

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
        raise HTTPException(status_code=401, detail="User account not found.")
    return user

# --- Pydantic Schemas ---
class GoogleAuthRequest(BaseModel):
    id_token: str

class ChatMessageRequest(BaseModel):
    conversation_id: str
    query: str
    category: Optional[str] = "All"
    api_key: Optional[str] = None

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Conversation"

class CompareDocumentsRequest(BaseModel):
    doc_ids: List[str]
    focus: Optional[str] = None
    conversation_id: Optional[str] = None

# --- Authentication Routes ---
@app.get("/api/config")
def get_public_config():
    return {
        "google_client_id": GOOGLE_CLIENT_ID,
        "demo_mode_available": True
    }

@app.post("/api/auth/google")
def authenticate_google(req: GoogleAuthRequest):
    profile = auth.verify_google_id_token(req.id_token)
    if not profile:
        raise HTTPException(status_code=400, detail="Invalid Google token.")
    
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

@app.get("/api/auth/me")
def get_me(user: dict = Depends(get_current_user)):
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "avatar_url": user["avatar_url"]
    }

@app.post("/api/auth/delete-account")
def delete_account(user: dict = Depends(get_current_user)):
    ns = f"user_{user['id']}"
    if retrieval.mode == "pinecone" and retrieval.pinecone_index:
        try:
            retrieval.pinecone_index.delete(delete_all=True, namespace=ns)
            print(f"[Pinecone] Fully wiped namespace '{ns}'")
        except Exception as e:
            print(f"[Pinecone Wipe Warning]: {e}")
    
    db.delete_user_account(user["id"])
    return {"message": "Account and all associated documents deleted successfully."}

# --- Conversation Routes ---
@app.get("/api/conversations")
def list_conversations(user: dict = Depends(get_current_user)):
    convs = db.get_user_conversations(user["id"])
    if not convs:
        default_conv = db.create_conversation(user["id"], title="Personal Document Assistant")
        convs = [default_conv]
    return convs

@app.post("/api/conversations")
def create_conversation(req: CreateConversationRequest, user: dict = Depends(get_current_user)):
    safe_title = make_safe_title(req.title or "New Conversation")
    return db.create_conversation(user["id"], title=safe_title)

@app.get("/api/conversations/{conv_id}/messages")
def get_conversation_history(conv_id: str, user: dict = Depends(get_current_user)):
    return db.get_conversation_messages(user["id"], conv_id)

@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    success = db.delete_conversation(user["id"], conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": "Conversation deleted."}

# --- Document Vault Routes with Category Support ---
@app.get("/api/documents")
def list_documents(category: Optional[str] = Query(None), user: dict = Depends(get_current_user)):
    docs = db.get_user_documents(user["id"], category=category)
    if not docs and not category:
        recovered = retrieval.migrate_legacy_documents_to_user(user["id"])
        for r in recovered:
            highlights = assistant.extract_document_highlights(r.get("content", ""), r["title"], fast_only=True)
            cat = highlights.get("category", r.get("category", "General"))
            file_ext = r["title"].rsplit(".", 1)[-1].lower() if "." in r["title"] else "pdf"
            db.add_document(
                user_id=user["id"],
                doc_id=r["id"],
                title=r["title"],
                original_filename=r["title"],
                file_type=file_ext,
                file_size_bytes=r["chunk_count"] * 1024,
                chunk_count=r["chunk_count"],
                category=cat,
                summary=highlights
            )
        docs = db.get_user_documents(user["id"], category=category)
    return docs

@app.post("/api/documents/upload")
async def upload_document(
    files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user)
):
    saved_docs = []
    for file in files:
        try:
            content_bytes = await file.read()
            filename = clean_utf8(file.filename or "uploaded_file.txt")
            
            raw_text = ingestion.extract_text_from_bytes(content_bytes, filename)
            if not raw_text.strip():
                continue

            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

            highlights = assistant.extract_document_highlights(raw_text, filename, fast_only=False)
            category = highlights.get("category", "General")

            chunks = ingestion.recursive_chunk_text(
                text=raw_text,
                doc_id=doc_id,
                doc_title=filename,
                collection="Personal",
                tags=[user["id"]]
            )
            retrieval.index_chunks(chunks, user_id=user["id"], category=category)

            saved = db.add_document(
                user_id=user["id"],
                doc_id=doc_id,
                title=filename,
                original_filename=filename,
                file_type=file_ext,
                file_size_bytes=len(content_bytes),
                chunk_count=len(chunks),
                category=category,
                summary=highlights
            )
            saved_docs.append(saved)
        except Exception as e:
            print(f"[Upload Error for {file.filename}]: {e}")

    if not saved_docs:
        raise HTTPException(status_code=400, detail="Could not process uploaded file(s).")
    
    return saved_docs[0] if len(saved_docs) == 1 else saved_docs

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = db.get_document(user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    retrieval.delete_document(user["id"], doc_id)
    db.delete_document(user["id"], doc_id)
    return {"message": "Document deleted successfully."}

# --- Cross-Document Comparison Route ---
@app.post("/api/documents/compare")
def compare_documents_route(req: CompareDocumentsRequest, user: dict = Depends(get_current_user)):
    if len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="Please select at least 2 documents to compare.")

    conv_id = req.conversation_id
    if not conv_id:
        conv = db.create_conversation(user["id"], title="Document Comparison")
        conv_id = conv["id"]

    result = assistant.compare_documents(
        user_id=user["id"],
        doc_ids=req.doc_ids,
        focus_query=req.focus
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    prompt_msg = f"Compare: {', '.join(result.get('compared_documents', []))}"
    if req.focus:
        prompt_msg += f" (Focus: {req.focus})"
    
    db.add_message(conv_id, sender="user", content=prompt_msg)
    saved_msg = db.add_message(
        conv_id,
        sender="assistant",
        content=result["answer"],
        citations=[{"clean_label": d, "doc_title": d, "snippet": "Document Scorecard Source"} for d in result.get("compared_documents", [])]
    )
    db.update_conversation_title(user["id"], conv_id, result["title"][:40])

    return {
        "conversation_id": conv_id,
        "title": result["title"],
        "answer": result["answer"],
        "compared_documents": result["compared_documents"]
    }

# --- Chat & Assistant Routes with Category-Scoped Retrieval ---
@app.post("/api/chat")
def chat(req: ChatMessageRequest, user: dict = Depends(get_current_user)):
    clean_query = clean_utf8(req.query)
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    conv = db.get_conversation(user["id"], req.conversation_id)
    if not conv:
        conv = db.create_conversation(user["id"], title=make_safe_title(clean_query))

    if conv["title"] == "New Conversation":
        db.update_conversation_title(user["id"], conv["id"], make_safe_title(clean_query))

    response = assistant.respond(
        user_id=user["id"],
        conversation_id=conv["id"],
        query=clean_query,
        user_groq_key=req.api_key,
        category_filter=req.category
    )
    return response

# --- Health Check & Frontend ---
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "platform": "Enterprise Multi-Tenant Personal AI Assistant",
        "database": "Neon PostgreSQL (Connection Pooled)" if db.is_postgres else "SQLite",
        "vector_backend": "Pinecone Namespaces" if retrieval.mode == "pinecone" else "ChromaDB (Local)",
        "isolation": "Strict per-user namespace"
    }

@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "online", "message": "Enterprise Assistant Backend is live!"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
