import os
import json
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
from ingestion_pipeline import IngestedChunk

class HybridRetrievalPipeline:
    """
    Cloud-Permanent Vector Database Pipeline (Option 3):
    - Primary: Pinecone Serverless Cloud Vector Database (Permanent 24/7, 0 data loss across Render restarts)
    - Fallback: Local ChromaDB Persistent Index (<30MB ONNX)
    - Auto-detects PINECONE_API_KEY from environment or settings
    """
    def __init__(self, db_path: str = "./chroma_cloud_data"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "personal-rag").strip()
        self.pinecone_client = None
        self.pinecone_index = None
        self.mode = "chromadb"

        # 384-dimensional ONNX embedding function (Fast, CPU-optimized, <30MB RAM)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize Pinecone if API key is present
        if self.pinecone_api_key:
            try:
                from pinecone import Pinecone, ServerlessSpec
                self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
                
                # Check or create serverless index in AWS us-east-1
                existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
                if self.pinecone_index_name not in existing_indexes:
                    print(f"[Pinecone] Creating serverless cloud index '{self.pinecone_index_name}' (384-D, cosine)...")
                    self.pinecone_client.create_index(
                        name=self.pinecone_index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
                self.mode = "pinecone"
                print(f"[Pinecone] Connected to permanent cloud vector index '{self.pinecone_index_name}' successfully!")
            except Exception as e:
                print(f"[Pinecone Warning] Could not connect to Pinecone cloud: {e}. Falling back to ChromaDB.")
                self.mode = "chromadb"

        # Always initialize ChromaDB as backup
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="cloud_personal_rag",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[IngestedChunk]) -> int:
        if not chunks:
            return 0

        # 1. Index into Pinecone Cloud if active (Option 3)
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                # Generate embeddings for all chunks
                texts = [c.content for c in chunks]
                embeddings = self.embed_fn(texts)
                
                vectors_to_upsert = []
                for chunk, emb in zip(chunks, embeddings):
                    # Store text snippet directly in metadata
                    vectors_to_upsert.append({
                        "id": chunk.chunk_id,
                        "values": emb,
                        "metadata": {
                            "doc_id": chunk.doc_id,
                            "doc_title": chunk.doc_title,
                            "collection": chunk.metadata.get("collection", "Personal"),
                            "tags": chunk.metadata.get("tags", ""),
                            "content": chunk.content[:1000],  # Stored permanently in Pinecone
                            "chunk_index": chunk.chunk_index
                        }
                    })
                
                # Upsert in batches of 100
                for i in range(0, len(vectors_to_upsert), 100):
                    batch = vectors_to_upsert[i:i + 100]
                    self.pinecone_index.upsert(vectors=batch)
                
                print(f"[Pinecone] Upserted {len(chunks)} chunks permanently to cloud index.")
            except Exception as e:
                print(f"[Pinecone Error during indexing]: {e}")

        # 2. Always index to local ChromaDB as well
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
        # 1. Query Pinecone Cloud if active (Option 3)
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                q_emb = self.embed_fn([query])[0]
                filter_dict = None
                if collection_filter and collection_filter.lower() not in ["all", ""]:
                    filter_dict = {"collection": {"$eq": collection_filter}}

                res = self.pinecone_index.query(
                    vector=q_emb,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )

                hits: List[Tuple[Dict[str, Any], float]] = []
                for match in res.get("matches", []):
                    meta = match.get("metadata", {})
                    score = round(max(0.40, min(0.99, match.get("score", 0.8))), 3)
                    item = {
                        "chunk_id": match.get("id"),
                        "doc_id": meta.get("doc_id", "unknown"),
                        "doc_title": meta.get("doc_title", "Document"),
                        "collection": meta.get("collection", "Personal"),
                        "content": meta.get("content", ""),
                        "score": score
                    }
                    hits.append((item, score))
                
                if hits:
                    return hits
            except Exception as e:
                print(f"[Pinecone Query Error]: {e}. Falling back to ChromaDB.")

        # 2. Fallback to ChromaDB
        total_count = self.collection.count()
        if total_count == 0:
            return []

        where_clause = None
        if collection_filter and collection_filter.lower() not in ["all", ""]:
            where_clause = {"collection": {"$in": [collection_filter, collection_filter.capitalize(), collection_filter.lower()]}}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total_count),
                where=where_clause
            )
        except Exception:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total_count)
            )

        hits = []
        if not results or not results['ids'] or not results['ids'][0]:
            return hits

        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.2] * len(ids)

        for cid, text, meta, dist in zip(ids, docs, metas, distances):
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
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                self.pinecone_index.delete(filter={"doc_id": {"$eq": doc_id}})
            except Exception:
                pass
        try:
            self.collection.delete(where={"doc_id": doc_id})
            return True
        except Exception:
            return False

    def count(self) -> int:
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                stats = self.pinecone_index.describe_index_stats()
                return stats.get("total_vector_count", self.collection.count())
            except Exception:
                pass
        return self.collection.count()
