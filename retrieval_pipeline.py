import os
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
from ingestion_pipeline import IngestedChunk

class HybridRetrievalPipeline:
    """
    ChromaDB Persistent Retrieval Pipeline:
    - Lightweight ONNX embeddings (<30MB RAM)
    - Returns top-K nearest-neighbor chunks without aggressive score filtering
    - Clean collection isolation & metadata scoring
    """
    def __init__(self, db_path: str = "./chroma_cloud_data"):
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
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

    def retrieve(self, query: str, top_k: int = 4, collection_filter: str = None) -> List[Tuple[Dict[str, Any], float]]:
        total_count = self.collection.count()
        if total_count == 0:
            return []

        where_clause = None
        if collection_filter and collection_filter.lower() not in ["all", ""]:
            # Case-insensitive collection matching
            where_clause = {"collection": {"$in": [collection_filter, collection_filter.capitalize(), collection_filter.lower()]}}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total_count),
                where=where_clause
            )
        except Exception:
            # Fallback without where filter if collection filter is incompatible
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total_count)
            )

        hits: List[Tuple[Dict[str, Any], float]] = []
        if not results or not results['ids'] or not results['ids'][0]:
            return hits

        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.2] * len(ids)

        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            # Calculate calibrated relevance score (0.50 - 0.99)
            norm_score = max(0.40, min(0.99, round(1.0 - (dist / 2.0), 3)))
            item = {
                "chunk_id": cid,
                "doc_id": meta.get("doc_id", "unknown"),
                "doc_title": meta.get("doc_title", "Document"),
                "collection": meta.get("collection", "General"),
                "content": text,
                "score": norm_score
            }
            hits.append((item, norm_score))

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
