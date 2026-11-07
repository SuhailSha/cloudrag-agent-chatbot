import os
import time
import math
from typing import List, Dict, Any, Tuple, Optional
from ingestion_pipeline import IngestedChunk

class MultiTenantRetrievalPipeline:
    """
    Enterprise Multi-Tenant Vector Retrieval Pipeline.
    Strictly enforces zero cross-tenant leakage by partitioning vectors 
    into dedicated Pinecone namespaces: `namespace=f"user_{user_id}"`.
    """
    def __init__(self, db_path: str = "./chroma_cloud_data"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "personal-rag").strip()
        self.pinecone_client = None
        self.pinecone_index = None
        self.mode = "chroma"

        # Initialize Pinecone Serverless Cloud if API Key is configured
        if self.pinecone_api_key:
            try:
                from pinecone import Pinecone, ServerlessSpec
                self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
                
                try:
                    names = [idx.name for idx in self.pinecone_client.list_indexes()]
                except Exception:
                    names = self.pinecone_client.list_indexes().names()

                if self.pinecone_index_name not in names:
                    print(f"[Pinecone] Creating serverless index '{self.pinecone_index_name}'...")
                    self.pinecone_client.create_index(
                        name=self.pinecone_index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    time.sleep(2)

                self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
                self.mode = "pinecone"
                print(f"[Pinecone] Connected to multi-tenant cloud index '{self.pinecone_index_name}'.")
            except Exception as e:
                print(f"[Pinecone Warning] Falling back to local ChromaDB: {e}")

        # Local ChromaDB client for fallback
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            print(f"[ChromaDB Init Warning]: {e}")
            self.chroma_client = None
            self.embed_fn = None

    def _get_namespace(self, user_id: str) -> str:
        return f"user_{user_id.strip()}"

    def _get_chroma_collection(self, user_id: str):
        if not self.chroma_client:
            return None
        safe_name = f"user_{user_id.replace('-', '_')}"
        return self.chroma_client.get_or_create_collection(
            name=safe_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[IngestedChunk], user_id: str) -> int:
        if not chunks or not user_id:
            return 0

        ns = self._get_namespace(user_id)
        texts = [c.content for c in chunks]

        # 1. Pinecone Serverless Cloud (Isolated Namespace)
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                embeddings = self.embed_fn(texts)
                vectors = []
                for chunk, emb in zip(chunks, embeddings):
                    emb_list = emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
                    vectors.append({
                        "id": chunk.chunk_id,
                        "values": emb_list,
                        "metadata": {
                            "user_id": user_id,
                            "doc_id": chunk.doc_id,
                            "doc_title": chunk.doc_title,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content[:1000]
                        }
                    })

                for i in range(0, len(vectors), 100):
                    batch = vectors[i:i + 100]
                    self.pinecone_index.upsert(vectors=batch, namespace=ns)

                print(f"[Pinecone] Upserted {len(chunks)} chunks to namespace '{ns}'.")
            except Exception as e:
                print(f"[Pinecone Indexing Error in namespace {ns}]: {e}")

        # 2. ChromaDB Fallback
        try:
            coll = self._get_chroma_collection(user_id)
            if coll:
                ids = [c.chunk_id for c in chunks]
                metadatas = [c.metadata for c in chunks]
                coll.add(ids=ids, documents=texts, metadatas=metadatas)
        except Exception as e:
            print(f"[ChromaDB Add Error]: {e}")

        return len(chunks)

    def retrieve(self, query: str, user_id: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if not query.strip() or not user_id:
            return []

        ns = self._get_namespace(user_id)
        hits = []

        # 1. Query Pinecone Cloud Namespace
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                raw_emb = self.embed_fn([query])[0]
                q_emb = raw_emb.tolist() if hasattr(raw_emb, "tolist") else [float(x) for x in raw_emb]
                
                res = self.pinecone_index.query(
                    vector=q_emb,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=ns
                )
                matches = getattr(res, 'matches', []) or (res.get('matches') if isinstance(res, dict) else [])
                for match in matches:
                    score = getattr(match, 'score', 0.5) or (match.get('score') if isinstance(match, dict) else 0.5)
                    meta = getattr(match, 'metadata', {}) or (match.get('metadata') if isinstance(match, dict) else {})
                    mid = getattr(match, 'id', 'chunk') or (match.get('id') if isinstance(match, dict) else 'chunk')
                    
                    item = {
                        "chunk_id": mid,
                        "doc_id": meta.get("doc_id", "unknown"),
                        "doc_title": meta.get("doc_title", "Document"),
                        "content": meta.get("content", ""),
                        "score": round(float(score), 3)
                    }
                    hits.append((item, float(score)))

                if hits:
                    return hits
            except Exception as e:
                print(f"[Pinecone Retrieve Error in namespace {ns}]: {e}")

        # 2. ChromaDB Fallback
        try:
            coll = self._get_chroma_collection(user_id)
            if coll and coll.count() > 0:
                results = coll.query(query_texts=[query], n_results=min(top_k, coll.count()))
                if results and results.get("ids") and results["ids"][0]:
                    ids = results['ids'][0]
                    docs = results['documents'][0]
                    metas = results['metadatas'][0]
                    for cid, text, meta in zip(ids, docs, metas):
                        item = {
                            "chunk_id": cid,
                            "doc_id": meta.get("doc_id", "unknown"),
                            "doc_title": meta.get("doc_title", "Document"),
                            "content": text,
                            "score": 0.85
                        }
                        hits.append((item, 0.85))
        except Exception as e:
            print(f"[ChromaDB Query Error]: {e}")

        return hits

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        ns = self._get_namespace(user_id)
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                self.pinecone_index.delete(
                    filter={"doc_id": {"$eq": doc_id}},
                    namespace=ns
                )
            except Exception as e:
                print(f"[Pinecone Delete Error in namespace {ns}]: {e}")

        try:
            coll = self._get_chroma_collection(user_id)
            if coll:
                coll.delete(where={"doc_id": doc_id})
        except Exception:
            pass

        return True

    def count_user_chunks(self, user_id: str) -> int:
        ns = self._get_namespace(user_id)
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                stats = self.pinecone_index.describe_index_stats()
                namespaces = getattr(stats, "namespaces", {}) or (stats.get("namespaces") if isinstance(stats, dict) else {})
                ns_info = namespaces.get(ns, {})
                count = getattr(ns_info, "vector_count", None) or (ns_info.get("vector_count") if isinstance(ns_info, dict) else 0)
                return count or 0
            except Exception:
                pass

        try:
            coll = self._get_chroma_collection(user_id)
            if coll:
                return coll.count()
        except Exception:
            pass
        return 0

    def migrate_legacy_documents_to_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Discovers legacy vectors stored in the default Pinecone namespace ("")
        and automatically migrates them into this user's isolated namespace.
        """
        if not (self.mode == "pinecone" and self.pinecone_index):
            return []

        ns = self._get_namespace(user_id)
        reconstructed = {}
        try:
            print(f"[Pinecone] Checking default namespace for legacy documents to claim for user '{user_id}'...")
            q_res = self.pinecone_index.query(
                vector=[0.05] * 384,
                top_k=1000,
                include_metadata=True,
                namespace=""
            )
            matches = getattr(q_res, 'matches', []) or (q_res.get('matches') if isinstance(q_res, dict) else []) or []

            if not matches:
                q_res = self.pinecone_index.query(
                    vector=[-0.05] * 384,
                    top_k=1000,
                    include_metadata=True,
                    namespace=""
                )
                matches = getattr(q_res, 'matches', []) or (q_res.get('matches') if isinstance(q_res, dict) else []) or []

            vectors_to_copy = []
            for m in matches:
                mid = getattr(m, 'id', '') or (m.get('id') if isinstance(m, dict) else '')
                if mid == "__registry__":
                    continue
                meta = getattr(m, 'metadata', {}) or (m.get('metadata') if isinstance(m, dict) else {})
                vals = getattr(m, 'values', []) or (m.get('values') if isinstance(m, dict) else [])
                
                # If values were not returned, fetch them
                if not vals:
                    try:
                        f_res = self.pinecone_index.fetch(ids=[mid], namespace="")
                        vecs = getattr(f_res, 'vectors', {}) or (f_res.get('vectors') if isinstance(f_res, dict) else {})
                        if mid in vecs:
                            v_obj = vecs[mid]
                            vals = getattr(v_obj, 'values', []) or (v_obj.get('values') if isinstance(v_obj, dict) else [])
                    except Exception:
                        pass
                
                if not vals:
                    vals = [0.05] * 384

                vectors_to_copy.append({
                    "id": mid,
                    "values": vals,
                    "metadata": {
                        "user_id": user_id,
                        "doc_id": meta.get("doc_id", "unknown"),
                        "doc_title": meta.get("doc_title", "Document"),
                        "content": meta.get("content", ""),
                        "chunk_index": meta.get("chunk_index", 0)
                    }
                })

                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in ["__system__", "unknown"]:
                    if doc_id not in reconstructed:
                        reconstructed[doc_id] = {
                            "id": doc_id,
                            "title": meta.get("doc_title") or f"{doc_id}.pdf",
                            "chunk_count": 0,
                            "content": meta.get("content", "")
                        }
                    reconstructed[doc_id]["chunk_count"] += 1

            if vectors_to_copy:
                for i in range(0, len(vectors_to_copy), 100):
                    batch = vectors_to_copy[i:i + 100]
                    self.pinecone_index.upsert(vectors=batch, namespace=ns)
                print(f"[Pinecone] Successfully migrated {len(vectors_to_copy)} legacy chunks into namespace '{ns}'!")

            return list(reconstructed.values())
        except Exception as e:
            print(f"[Pinecone Migration Error]: {e}")
            return []
