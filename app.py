import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingestion_pipeline import DocumentIngestionPipeline
from retrieval_pipeline import HybridRetrievalPipeline
from agent_swarm import MultiAgentSwarmOrchestrator

# Security Configuration
SECURITY_PASSCODE = os.getenv("PERSONAL_PASSCODE", "my_secret_passcode_2026")
PORT = int(os.getenv("PORT", 7860))

app = FastAPI(
    title="24/7 Cloud Personal RAG Multi-Agent System",
    description="Zero-cost, secure, 24/7 personal document RAG chatbot accessible from any phone or computer.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipelines
ingestion = DocumentIngestionPipeline(chunk_size=350, chunk_overlap=40)
retrieval = HybridRetrievalPipeline(db_path="./chroma_cloud_data")
orchestrator = MultiAgentSwarmOrchestrator(retrieval)

# In-memory document metadata registry
document_registry = {}
task_registry = []

def verify_token(x_passcode: Optional[str] = Header(None, alias="X-Personal-Passcode")):
    if SECURITY_PASSCODE and SECURITY_PASSCODE != "my_secret_passcode_2026":
        if not x_passcode or x_passcode != SECURITY_PASSCODE:
            raise HTTPException(status_code=401, detail="Unauthorized access: Invalid Security Passcode.")
    return True

class ChatRequest(BaseModel):
    query: str
    collection: Optional[str] = "all"
    top_k: int = 4

@app.get("/api/health")
def health():
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
        
        # 1. Ingestion: extract text
        raw_text = ingestion.extract_text_from_bytes(content_bytes, filename)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from document.")
        
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # 2. Ingestion: chunking
        chunks = ingestion.recursive_chunk_text(raw_text, doc_id, filename, collection, tag_list)
        
        # 3. Retrieval: vector indexing
        retrieval.index_chunks(chunks)

        doc_meta = {
            "id": doc_id,
            "title": filename,
            "collection": collection,
            "tags": tag_list,
            "chunk_count": len(chunks),
            "size_bytes": len(content_bytes),
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
    retrieval.delete_document(doc_id)
    document_registry.pop(doc_id, None)
    return {"message": "Document deleted successfully"}

@app.post("/api/chat")
def chat_with_agents(req: ChatRequest):
    result = orchestrator.run_swarm(
        query=req.query,
        top_k=req.top_k,
        collection=req.collection
    )
    if result.get("extracted_tasks"):
        task_registry.extend(result["extracted_tasks"])
    return result

@app.get("/api/tasks")
def get_tasks():
    return task_registry

# Serve the responsive Web App UI directly to any mobile or desktop browser
@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Server online. Please place index.html into ./static/ directory."}

if __name__ == "__main__":
    import uvicorn
    print(f"Launching 24/7 Cloud Personal RAG Server on port {PORT}...")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
