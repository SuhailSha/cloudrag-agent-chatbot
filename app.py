import os
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
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
    version="4.0.1"
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

# Dependency: Enforce authentication and extract tenant user
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
    api_key: Optional[str] = None

class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Conversation"

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

# --- Conversation Routes (Strict User Isolation) ---
@app.get("/api/conversations")
def list_conversations(user: dict = Depends(get_current_user)):
    return db.get_user_conversations(user["id"])

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

# --- Document Vault Routes (Strict User Isolation) ---
@app.get("/api/documents")
def list_documents(user: dict = Depends(get_current_user)):
    return db.get_user_documents(user["id"])

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    try:
        content_bytes = await file.read()
        filename = clean_utf8(file.filename or "uploaded_file.txt")
        
        raw_text = ingestion.extract_text_from_bytes(content_bytes, filename)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from document.")

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

        # Chunk and index strictly into the user's isolated namespace
        chunks = ingestion.recursive_chunk_text(
            text=raw_text,
            doc_id=doc_id,
            doc_title=filename,
            collection="Personal",
            tags=[user["id"]]
        )
        retrieval.index_chunks(chunks, user_id=user["id"])

        # Record in database
        saved_doc = db.add_document(
            user_id=user["id"],
            doc_id=doc_id,
            title=filename,
            original_filename=filename,
            file_type=file_ext,
            file_size_bytes=len(content_bytes),
            chunk_count=len(chunks)
        )
        return saved_doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    doc = db.get_document(user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    retrieval.delete_document(user["id"], doc_id)
    db.delete_document(user["id"], doc_id)
    return {"message": "Document deleted successfully."}

# --- Chat & Assistant Routes ---
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
        user_groq_key=req.api_key
    )
    return response

# --- Health Check & Frontend ---
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "platform": "Enterprise Multi-Tenant Personal AI Assistant",
        "auth_provider": "Google OAuth 2.0 (OpenID Connect)",
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
