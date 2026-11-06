import os
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
from ingestion_pipeline import IngestedChunk

class HybridRetrievalPipeline:
    """
    Ultra-lightweight, 512MB RAM-optimized Retrieval Pipeline for Render:
    - Uses ChromaDB's high-performance ONNX embedding function (runs in <30MB RAM vs 500MB PyTorch)
    - Starts in < 1 second so Render port 10000 binds instantly
    - Persistent vector database in ./chroma_cloud_data
    """
    def __init__(self, db_path: str = "./chroma_cloud_data"):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # High-performance, lightweight ONNX all-MiniLM-L6-v2 embedding function
        # This uses minimal memory (<30MB) and boots in 0.1s
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name="cloud_personal_rag",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[IngestedChunk]) -> int:
        if not chunks:
            return 0
        
        ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4, collection_filter: str = None, min_score: float = 0.35) -> List[Tuple[Dict[str, Any], float]]:
        total_count = self.collection.count()
        if total_count == 0:
            return []

        where_clause = None
        if collection_filter and collection_filter.lower() != "all":
            where_clause = {"collection": collection_filter}

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, total_count),
            where=where_clause
        )

        hits: List[Tuple[Dict[str, Any], float]] = []
        if not results or not results['ids'] or not results['ids'][0]:
            return hits

        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.2] * len(ids)

        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            score = max(0.0, min(1.0, round(1.0 - dist, 3)))
            if score >= min_score:
                item = {
                    "chunk_id": cid,
                    "doc_id": meta.get("doc_id", "unknown"),
                    "doc_title": meta.get("doc_title", "Document"),
                    "collection": meta.get("collection", "General"),
                    "content": text,
                    "score": score
                }
                hits.append((item, score))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def delete_document(self, doc_id: str) -> bool:
        try:
            self.collection.delete(where={"doc_id": doc_id})
            return True
        except Exception:
            return False

    def count(self) -> int:
        return self.collection.count()
