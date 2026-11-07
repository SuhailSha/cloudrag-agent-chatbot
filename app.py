import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ingestion_pipeline import DocumentIngestionPipeline

# Security Configuration
SECURITY_PASSCODE = os.getenv("PERSONAL_PASSCODE", "my_secret_passcode_2026")
PORT = int(os.getenv("PORT", 10000))

app = FastAPI(
    title="24/7 Cloud Personal RAG Multi-Agent System",
    description="Fast, lightweight, 24/7 personal document RAG chatbot.",
    version="3.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances initialized lazily to ensure instant port 10000 binding
_ingestion = None
_retrieval = None
_orchestrator = None

def get_ingestion():
    global _ingestion
    if _ingestion is None:
        _ingestion = DocumentIngestionPipeline(chunk_size=350, chunk_overlap=40)
    return _ingestion

def get_retrieval():
    global _retrieval
    if _retrieval is None:
        from retrieval_pipeline import HybridRetrievalPipeline
        _retrieval = HybridRetrievalPipeline(db_path="./chroma_cloud_data")
    return _retrieval

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agent_swarm import MultiAgentSwarmOrchestrator
        _orchestrator = MultiAgentSwarmOrchestrator(get_retrieval())
    return _orchestrator

document_registry = {}
task_registry = []

class ChatRequest(BaseModel):
    query: str
    collection: Optional[str] = "all"
    top_k: int = 4
    api_key: Optional[str] = None

@app.get("/api/health")
def health():
    retrieval = get_retrieval()
    return {
        "status": "online",
        "service": "24/7 Cloud Personal RAG",
        "total_chunks": retrieval.count(),
        "total_documents": len(document_registry)
    }

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("Personal"),
    tags: str = Form("")
):
    try:
        content_bytes = await file.read()
        filename = file.filename or "uploaded_file.txt"
        
        ingestion = get_ingestion()
        retrieval = get_retrieval()

        raw_text = ingestion.extract_text_from_bytes(content_bytes, filename)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from document.")
        
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        chunks = ingestion.recursive_chunk_text(raw_text, doc_id, filename, collection, tag_list)
        retrieval.index_chunks(chunks)

        doc_meta = {
            "id": doc_id,
            "title": filename,
            "collection": collection,
            "tags": tag_list,
            "chunk_count": len(chunks),
            "size_bytes": len(content_bytes),
            "content": raw_text[:5000],
            "active": True
        }
        document_registry[doc_id] = doc_meta
        return doc_meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
def get_documents():
    return list(document_registry.values())

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    retrieval = get_retrieval()
    retrieval.delete_document(doc_id)
    document_registry.pop(doc_id, None)
    return {"message": "Document deleted successfully"}

@app.post("/api/chat")
def chat_with_agents(req: ChatRequest):
    orchestrator = get_orchestrator()
    result = orchestrator.run_swarm(
        query=req.query,
        top_k=req.top_k,
        collection=req.collection,
        user_groq_key=req.api_key
    )
    if result.get("extracted_tasks"):
        task_registry.extend(result["extracted_tasks"])
    return result

@app.get("/api/tasks")
def get_tasks():
    return task_registry

# Serve UI directly to phone and desktop browsers
@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "status": "online",
        "message": "Personal RAG API is live and listening on port 10000!"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
