import json
import os
import re
import time
import math
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from ingestion_pipeline import IngestedChunk

logger = logging.getLogger("aura.retrieval")

# Universal English stopwords for standard lexical BM25 tokenization
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

TOKEN_PATTERN = re.compile(r'\b[a-zA-Z0-9_\-\./]{2,}\b')


def tokenize_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS and len(t) >= 2]


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class MultiTenantRetrievalPipeline:
    """
    Modular Multi-Tenant Hybrid Retrieval Pipeline (Pure Semantic Dense Vector + BM25):
    - True Dense Semantic Vector Search via all-MiniLM-L6-v2 embeddings (Chroma / Pinecone).
    - True Sparse BM25 Search with alphanumeric identifier and title boosting for exact codes/dates/terms.
    - Zero hardcoded keyword or synonym tables: semantic intent and paraphrasing are handled natively by embeddings.
    - Reciprocal Rank Fusion (RRF) preserves raw calibrated dense_score and sparse_score alongside fused ranks.
    - Automatic DB-persisted vector fallback for container restarts with zero split-brain.
    """
    def __init__(self, db_path: str = "./chroma_cloud_data", database=None):
        self.db_path = db_path
        self.db = database
        os.makedirs(db_path, exist_ok=True)

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "personal-rag").strip()
        self.pinecone_client = None
        self.pinecone_index = None
        self.mode = "chroma"

        if self.pinecone_api_key:
            try:
                from pinecone import Pinecone, ServerlessSpec
                self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)

                existing_indexes = [idx["name"] for idx in self.pinecone_client.list_indexes()]
                if self.pinecone_index_name not in existing_indexes:
                    logger.info(f"Creating Pinecone index '{self.pinecone_index_name}'...")
                    try:
                        self.pinecone_client.create_index(
                            name=self.pinecone_index_name,
                            dimension=384,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                        time.sleep(2)
                    except Exception as e:
                        if "409" not in str(e) and "ALREADY_EXISTS" not in str(e).upper():
                            raise e

                self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
                self.mode = "pinecone"
                logger.info(f"Connected to Pinecone serverless index '{self.pinecone_index_name}'.")
            except Exception as e:
                logger.warning(f"Pinecone init failed ({e}); using local ChromaDB.")
                self.mode = "chroma"

        try:
            import chromadb
            from chromadb.utils import embedding_functions
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            logger.warning(f"ChromaDB unavailable ({e}); running in sparse_only mode if Pinecone is not active.")
            self.chroma_client = None
            self.embed_fn = None
            if self.mode == "chroma":
                self.mode = "sparse_only"

    def check_vector_health(self) -> Tuple[bool, str]:
        """Performs a live connectivity check against the active vector store (Fixes H8)."""
        if self.mode == "pinecone":
            if not self.pinecone_index:
                return False, "pinecone_uninitialized"
            try:
                self.pinecone_index.describe_index_stats()
                return True, "pinecone"
            except Exception as e:
                logger.error(f"Pinecone readiness check failed: {e}")
                return False, "pinecone_unreachable"
        elif self.mode == "chroma":
            if not self.chroma_client:
                return False, "chroma_uninitialized"
            try:
                self.chroma_client.heartbeat()
                return True, "chroma"
            except Exception as e:
                logger.error(f"Chroma readiness check failed: {e}")
                return False, "chroma_unreachable"
        return True, "sparse_only"

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

    def _compute_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []
        if self.embed_fn:
            try:
                raw_embs = self.embed_fn(texts)
                return [
                    (e.tolist() if hasattr(e, "tolist") else [float(x) for x in e])
                    for e in raw_embs
                ]
            except Exception as e:
                logger.warning(f"Embedding generation notice: {e}")
        return None

    def index_chunks(self, chunks: List[IngestedChunk], user_id: str, category: str = "General") -> int:
        if not chunks or not user_id:
            return 0

        ns = self._get_namespace(user_id)
        texts = [c.content for c in chunks]
        computed_embs = self._compute_embeddings(texts)
        if computed_embs and self.db and hasattr(self.db, "update_chunk_embeddings"):
            try:
                emb_map = {c.chunk_id: emb for c, emb in zip(chunks, computed_embs)}
                self.db.update_chunk_embeddings(user_id, emb_map)
            except Exception as e:
                logger.warning(f"Could not persist embeddings to DB: {e}")

        if self.mode == "pinecone":
            if not self.pinecone_index or not self.embed_fn:
                logger.error("Pinecone mode active but index or embedding function is missing.")
                return 0
            try:
                embeddings = computed_embs or self.embed_fn(texts)
                vectors = []
                for chunk, emb in zip(chunks, embeddings):
                    emb_list = emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
                    page_num = chunk.metadata.get("page_number", 1)
                    vectors.append({
                        "id": chunk.chunk_id,
                        "values": emb_list,
                        "metadata": {
                            "user_id": user_id,
                            "doc_id": chunk.doc_id,
                            "doc_title": chunk.doc_title,
                            "category": category,
                            "page_number": page_num,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content[:3000]
                        }
                    })

                for i in range(0, len(vectors), 100):
                    batch = vectors[i:i + 100]
                    self.pinecone_index.upsert(vectors=batch, namespace=ns)

                logger.info(f"Upserted {len(vectors)} chunks to Pinecone namespace '{ns}'.")
                return len(vectors)
            except Exception as e:
                logger.error(f"Pinecone upsert failed for namespace '{ns}': {e}")
                return 0

        if self.mode == "chroma":
            col = self._get_chroma_collection(user_id)
            if not col:
                return 0
            try:
                ids = [c.chunk_id for c in chunks]
                metas = [{
                    "user_id": user_id,
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "category": category,
                    "page_number": c.metadata.get("page_number", 1),
                    "chunk_index": c.chunk_index
                } for c in chunks]
                col.upsert(ids=ids, documents=texts, metadatas=metas)
                return len(chunks)
            except Exception as e:
                logger.error(f"Chroma upsert failed for user '{user_id}': {e}")
                return 0

        return len(chunks)

    def _dense_search(
        self, query: str, user_id: str, top_k: int = 15, allowed_doc_ids: Optional[set] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        dense_hits = []
        ns = self._get_namespace(user_id)

        if self.mode == "pinecone" and self.pinecone_index and self.embed_fn:
            try:
                raw_emb = self.embed_fn([query])[0]
                q_emb = raw_emb.tolist() if hasattr(raw_emb, "tolist") else [float(x) for x in raw_emb]

                query_kwargs = {
                    "vector": q_emb,
                    "top_k": top_k * 2 if allowed_doc_ids is not None else top_k,
                    "include_metadata": True,
                    "namespace": ns
                }
                if allowed_doc_ids is not None and len(allowed_doc_ids) > 0:
                    query_kwargs["filter"] = {"doc_id": {"$in": list(allowed_doc_ids)}}

                res = self.pinecone_index.query(**query_kwargs)
                matches = getattr(res, "matches", []) or (res.get("matches") if isinstance(res, dict) else []) or []

                for m in matches:
                    score = getattr(m, "score", 0.0) or (m.get("score") if isinstance(m, dict) else 0.0) or 0.0
                    meta = dict(getattr(m, "metadata", {}) or (m.get("metadata") if isinstance(m, dict) else {}) or {})
                    mid = getattr(m, "id", "") or (m.get("id") if isinstance(m, dict) else "")
                    if allowed_doc_ids is not None and meta.get("doc_id") not in allowed_doc_ids:
                        continue
                    meta["chunk_id"] = mid
                    dense_hits.append((meta, float(score)))
                return dense_hits[:top_k]
            except Exception as e:
                logger.warning(f"Pinecone dense search warning: {e}")
                return []

        chroma_doc_ids = set()
        col = self._get_chroma_collection(user_id)
        if col:
            try:
                where_filter = None
                if allowed_doc_ids is not None and len(allowed_doc_ids) > 0:
                    doc_list = list(allowed_doc_ids)
                    where_filter = {"doc_id": doc_list[0]} if len(doc_list) == 1 else {"doc_id": {"$in": doc_list}}

                existing_in_chroma = col.get(where=where_filter, include=["metadatas"])
                if existing_in_chroma and existing_in_chroma.get("metadatas"):
                    for m_obj in existing_in_chroma["metadatas"]:
                        if m_obj and m_obj.get("doc_id"):
                            chroma_doc_ids.add(m_obj["doc_id"])

                res = col.query(query_texts=[query], n_results=top_k, where=where_filter)
                if res and res.get("ids") and res["ids"][0]:
                    ids = res["ids"][0]
                    docs = res["documents"][0]
                    metas = res["metadatas"][0]
                    dists = res.get("distances", [[]])[0]
                    for cid, text, meta, dist in zip(ids, docs, metas, dists):
                        meta_dict = dict(meta) if meta else {}
                        if allowed_doc_ids is not None and meta_dict.get("doc_id") not in allowed_doc_ids:
                            continue
                        meta_dict["content"] = text
                        meta_dict["chunk_id"] = cid
                        cos_sim = max(0.0, min(1.0, 1.0 - float(dist)))
                        dense_hits.append((meta_dict, cos_sim))
            except Exception as e:
                logger.warning(f"Chroma dense search warning: {e}")

        missing_doc_ids = (allowed_doc_ids - chroma_doc_ids) if allowed_doc_ids is not None else None
        needs_db_fallback = (not dense_hits) or (missing_doc_ids is not None and len(missing_doc_ids) > 0)

        if needs_db_fallback and self.db:
            q_embs = self._compute_embeddings([query])
            if q_embs and q_embs[0]:
                q_vec = q_embs[0]
                target_doc_ids = missing_doc_ids if (dense_hits and missing_doc_ids) else allowed_doc_ids
                if hasattr(self.db, "get_user_chunk_embeddings"):
                    db_emb_rows = self.db.get_user_chunk_embeddings(user_id, allowed_doc_ids=target_doc_ids)
                elif hasattr(self.db, "get_all_user_chunks"):
                    db_emb_rows = self.db.get_all_user_chunks(user_id)
                else:
                    db_emb_rows = []

                seen_cids = {m.get("chunk_id") for m, _ in dense_hits if m.get("chunk_id")}
                for row in db_emb_rows:
                    cid = row.get("chunk_id")
                    if not cid or cid in seen_cids:
                        continue
                    if allowed_doc_ids is not None and row.get("doc_id") not in allowed_doc_ids:
                        continue
                    emb_str = row.get("embedding_json") or "[]"
                    if emb_str and emb_str != "[]":
                        try:
                            c_vec = json.loads(emb_str)
                            if c_vec:
                                sim = compute_cosine_similarity(q_vec, c_vec)
                                row_meta = {
                                    "chunk_id": cid,
                                    "doc_id": row.get("doc_id"),
                                    "doc_title": row.get("doc_title"),
                                    "category": row.get("category", "General"),
                                    "page_number": row.get("page_number", 1),
                                    "chunk_index": row.get("chunk_index", 1),
                                }
                                if row.get("content"):
                                    row_meta["content"] = row["content"]
                                dense_hits.append((row_meta, sim))
                        except Exception:
                            pass
                dense_hits.sort(key=lambda x: x[1], reverse=True)

        return dense_hits[:top_k]

    def _sparse_search(
        self, query: str, user_id: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 15
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        High-speed O(1) Pre-Tokenized BM25 Sparse Retrieval (Pure Information Retrieval):
        - Tokenizes query and candidates into Counter maps.
        - Exact term matches receive full BM25 TF-IDF scoring.
        - Substring matches for compound identifiers (e.g. policy codes 'PBW/00001001', dates, alphanumeric tags)
          receive a high-weight multiplier to guarantee exact-code recall.
        - Zero hardcoded concept synonym maps: semantics are carried natively by dense vector embeddings.
        """
        query_terms = list(dict.fromkeys(tokenize_for_bm25(query)))
        if not query_terms or not candidate_chunks:
            return []

        N = len(candidate_chunks)
        chunk_counters = []
        chunk_lengths = []
        doc_frequencies = {t: 0 for t in query_terms}

        for item in candidate_chunks:
            content_low = (item.get("content", "") or "").lower()
            tokens = tokenize_for_bm25(content_low)
            counts = Counter(tokens)
            chunk_counters.append(counts)
            chunk_lengths.append(max(1, len(tokens)))
            for t in query_terms:
                if counts.get(t, 0) > 0:
                    doc_frequencies[t] += 1
                elif ("/" in t or "-" in t or "." in t) and t in content_low:
                    counts[t] = 1
                    doc_frequencies[t] += 1

        avg_dl = sum(chunk_lengths) / float(N) if N > 0 else 100.0
        k1 = 1.5
        b = 0.75

        sparse_hits = []
        for idx, item in enumerate(candidate_chunks):
            counts = chunk_counters[idx]
            dl = chunk_lengths[idx]
            doc_title_lower = (item.get("doc_title") or "").lower()
            content_low = (item.get("content") or "").lower()
            score = 0.0
            matched_query_terms = 0

            for term in query_terms:
                tf = counts.get(term, 0)
                in_content = tf > 0 or (("/" in term or "-" in term or "." in term) and term in content_low)
                in_title = term in doc_title_lower and len(term) >= 2

                if in_content or in_title:
                    matched_query_terms += 1
                    effective_tf = max(1, tf)
                    df = doc_frequencies.get(term, 1)
                    idf = max(0.35, math.log(1.0 + (N - df + 0.5) / (df + 0.5)))
                    tf_norm = (effective_tf * (k1 + 1.0)) / (effective_tf + k1 * (1.0 - b + b * (dl / max(1.0, avg_dl))))
                    has_digit = any(c.isdigit() for c in term)
                    multiplier = 4.0 if has_digit else 1.0
                    score += multiplier * idf * tf_norm
                    if in_title:
                        score += 2.0

            if score > 0.0 and matched_query_terms > 0:
                enriched = dict(item)
                enriched["matched_terms"] = matched_query_terms
                enriched["query_term_coverage"] = matched_query_terms / float(len(query_terms))
                sparse_hits.append((enriched, score))

        sparse_hits.sort(key=lambda x: x[1], reverse=True)
        return sparse_hits[:top_k]

    def retrieve(
        self, query: str, user_id: str, top_k: int = 5, category: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not query.strip() or not user_id:
            return []

        allowed_doc_ids = None
        if self.db and hasattr(self.db, "get_user_documents"):
            if category and category.lower() not in ["all", ""]:
                cat_docs = self.db.get_user_documents(user_id, category=category.strip())
                allowed_doc_ids = {d["id"] for d in cat_docs}
                if not allowed_doc_ids:
                    return []
            else:
                all_docs = self.db.get_user_documents(user_id)
                allowed_doc_ids = {d["id"] for d in all_docs}
                if not allowed_doc_ids:
                    return []

        pool_k = max(15, top_k * 3)
        dense_results = self._dense_search(query, user_id=user_id, top_k=pool_k, allowed_doc_ids=allowed_doc_ids)

        candidate_pool = []
        if self.db and hasattr(self.db, "get_all_user_chunks"):
            candidate_pool = self.db.get_all_user_chunks(user_id)
            if allowed_doc_ids is not None:
                candidate_pool = [c for c in candidate_pool if c.get("doc_id") in allowed_doc_ids]

        if not candidate_pool:
            dense_chunk_map = {item["chunk_id"]: item for item, _ in dense_results if "chunk_id" in item}
            candidate_pool = list(dense_chunk_map.values())

        sparse_results = self._sparse_search(query, user_id=user_id, candidate_chunks=candidate_pool, top_k=pool_k)

        rrf_scores: Dict[str, float] = {}
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}
        item_pool: Dict[str, Dict[str, Any]] = {}

        for rank, (item, d_score) in enumerate(dense_results):
            cid = item.get("chunk_id")
            if cid:
                item_pool[cid] = dict(item)
                dense_scores[cid] = float(d_score)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank + 1.0))

        for rank, (item, s_score) in enumerate(sparse_results):
            cid = item.get("chunk_id")
            if cid:
                if cid in item_pool:
                    item_pool[cid]["matched_terms"] = item.get("matched_terms", 0)
                    item_pool[cid]["query_term_coverage"] = item.get("query_term_coverage", 0.0)
                    if not item_pool[cid].get("content") and item.get("content"):
                        item_pool[cid]["content"] = item["content"]
                else:
                    item_pool[cid] = dict(item)
                sparse_scores[cid] = float(s_score)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank + 1.0))

        for cid in rrf_scores:
            if dense_scores.get(cid, 0.0) >= 0.35 and sparse_scores.get(cid, 0.0) >= 1.0:
                rrf_scores[cid] += 0.015

        ranked_cids = sorted(
            rrf_scores.keys(),
            key=lambda k: (rrf_scores[k], sparse_scores.get(k, 0.0), dense_scores.get(k, 0.0)),
            reverse=True
        )

        final_hits = []
        for cid in ranked_cids[:top_k]:
            item = dict(item_pool[cid])
            d_sc = dense_scores.get(cid, 0.0)
            s_sc = sparse_scores.get(cid, 0.0)
            r_sc = rrf_scores[cid]

            item["dense_score"] = round(d_sc, 4)
            item["sparse_score"] = round(s_sc, 4)
            item["rrf_score"] = round(r_sc, 5)

            if self.db and hasattr(self.db, "get_full_chunk_text"):
                full_text = self.db.get_full_chunk_text(cid, user_id)
                if full_text:
                    item["content"] = full_text

            final_hits.append((item, r_sc))

        return final_hits

    def delete_document(self, user_id: str, doc_id: str, chunk_ids: Optional[List[str]] = None) -> bool:
        ns = self._get_namespace(user_id)
        success = True

        if not chunk_ids and self.db and hasattr(self.db, "get_document_full_chunks"):
            try:
                rows = self.db.get_document_full_chunks(user_id, doc_id)
                chunk_ids = [r["chunk_id"] for r in rows if r.get("chunk_id")]
            except Exception:
                chunk_ids = []

        if self.mode == "pinecone" and self.pinecone_index:
            try:
                if chunk_ids:
                    for i in range(0, len(chunk_ids), 100):
                        self.pinecone_index.delete(ids=chunk_ids[i:i + 100], namespace=ns)
                else:
                    self.pinecone_index.delete(filter={"doc_id": doc_id}, namespace=ns)
            except Exception as e:
                logger.error(f"Pinecone delete error for doc '{doc_id}': {e}")
                success = False

        col = self._get_chroma_collection(user_id)
        if col:
            try:
                if chunk_ids:
                    col.delete(ids=chunk_ids)
                else:
                    col.delete(where={"doc_id": doc_id})
            except Exception as e:
                logger.error(f"Chroma delete error for doc '{doc_id}': {e}")
                if self.mode == "chroma":
                    success = False

        return success

    def delete_user_vault(self, user_id: str) -> bool:
        ns = self._get_namespace(user_id)
        success = True
        if self.mode == "pinecone" and self.pinecone_index:
            try:
                self.pinecone_index.delete(delete_all=True, namespace=ns)
            except Exception as e:
                logger.error(f"Pinecone vault wipe error for '{user_id}': {e}")
                success = False

        if self.chroma_client:
            safe_name = f"user_{user_id.replace('-', '_')}"
            try:
                self.chroma_client.delete_collection(name=safe_name)
            except Exception as e:
                if "does not exist" not in str(e).lower():
                    logger.warning(f"Chroma collection delete notice for '{user_id}': {e}")
                    if self.mode == "chroma":
                        success = False
        return success
