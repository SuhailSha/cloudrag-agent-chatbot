from contextlib import contextmanager
import os
import sqlite3
import json
import uuid
import time
from typing import Dict, List, Optional, Any

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.lower().startswith("postgresql://") or DATABASE_URL.lower().startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL.split("://", 1)[1]

if "channel_binding=" in DATABASE_URL:
    parts = DATABASE_URL.split("?")
    base = parts[0]
    if len(parts) > 1:
        query_params = [p for p in parts[1].split("&") if not p.startswith("channel_binding=")]
        DATABASE_URL = base + ("?" + "&".join(query_params) if query_params else "")

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./chroma_cloud_data/assistant_vault.db")

def clean_utf8(val: Any) -> str:
    if val is None:
        return ""
    s = str(val)
    return s.encode('utf-8', 'ignore').decode('utf-8', 'ignore').strip()

class VaultDatabase:
    """
    Enterprise Multi-Tenant Database Engine with Connection Pooling,
    Smart Vault Category Filtering, and 3-Layer Persistent User Memory.
    """
    @contextmanager
    def transaction(self):
        """
        True Single-Commit Transaction Scope (Resolves Finding 1 & P0-02):
        Executes multi-statement operations on a single dedicated connection.
        Commits once upon success, rolls back atomically on any failure.
        """
        if self.is_postgres and self.pg_pool:
            conn = self._get_pg_conn()
            if not conn:
                raise RuntimeError("Database Transaction Error: Unable to acquire connection from pool.")
            try:
                yield conn
                conn.commit()
            except Exception:
                try:
                    if not getattr(conn, 'closed', 0):
                        conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                if conn:
                    try:
                        self.pg_pool.putconn(conn)
                    except Exception:
                        pass
        else:
            conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def __init__(self, db_url: str = DATABASE_URL, sqlite_path: str = SQLITE_DB_PATH):
        self.db_url = db_url
        self.sqlite_path = sqlite_path
        self.is_postgres = False
        self.pg_pool = None

        if self.db_url:
            try:
                import psycopg2
                from psycopg2 import pool
                self.pg_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.db_url,
                    sslmode="require"
                )
                self.is_postgres = True
                print("[Database] Successfully connected to Neon PostgreSQL Connection Pool (Warm 10ms queries)!")
            except Exception as e:
                allow_sqlite = os.getenv("ALLOW_SQLITE_FALLBACK", "").strip().lower() in ["1", "true"]
                if not allow_sqlite:
                    print(f"[FATAL DATABASE ERROR] Could not connect to production PostgreSQL: {e}")
                    raise RuntimeError(f"Database Initialization Failed: Unable to establish connection to DATABASE_URL ({e}). Failing closed to prevent silent data split-brain.")
                print(f"[Database Warning] Local development fallback to SQLite: {e}")
                self.is_postgres = False

        if not self.is_postgres:
            os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)
            print(f"[Database] Using SQLite storage at '{self.sqlite_path}'.")

        self._init_db()

    def _get_pg_conn(self):
        if not (self.is_postgres and self.pg_pool):
            return None
        import psycopg2
        for _ in range(3):
            conn = None
            try:
                conn = self.pg_pool.getconn()
                if not conn or getattr(conn, 'closed', 0) != 0:
                    if conn:
                        try:
                            self.pg_pool.putconn(conn, close=True)
                        except Exception:
                            pass
                    continue
                # Quick health check for serverless Neon pooler
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                return conn
            except Exception:
                if conn:
                    try:
                        self.pg_pool.putconn(conn, close=True)
                    except Exception:
                        pass
        try:
            return psycopg2.connect(self.db_url, sslmode="require")
        except Exception as e:
            print(f"[Database Connection Error]: {e}")
            return None

    def _execute(self, query: str, params: tuple = ()):
        if self.is_postgres and self.pg_pool:
            import psycopg2.extras
            pg_query = query.replace("?", "%s")
            conn = None
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(pg_query, params)
                    conn.commit()
                    try:
                        rows = cur.fetchall()
                        return [dict(r) for r in rows]
                    except Exception:
                        return []
            except Exception as e:
                if conn:
                    try:
                        if not getattr(conn, 'closed', 0):
                            conn.rollback()
                    except Exception:
                        pass
                    try:
                        self.pg_pool.putconn(conn, close=True)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    conn = None
                print(f"[PostgreSQL Execute Error]: {e}")
                return []
            finally:
                if conn:
                    try:
                        self.pg_pool.putconn(conn)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
        else:
            conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                cur = conn.execute(query, params)
                conn.commit()
                try:
                    rows = cur.fetchall()
                    return [dict(r) for r in rows]
                except Exception:
                    return []
            finally:
                conn.close()

    def _execute_scalar(self, query: str, params: tuple = ()):
        rows = self._execute(query, params)
        return rows[0] if rows else None

    def _execute_non_query(self, query: str, params: tuple = ()):
        if self.is_postgres and self.pg_pool:
            pg_query = query.replace("?", "%s")
            conn = None
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return 0
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                    conn.commit()
                    return cur.rowcount
            except Exception as e:
                if conn:
                    try:
                        if not getattr(conn, 'closed', 0):
                            conn.rollback()
                    except Exception:
                        pass
                    try:
                        self.pg_pool.putconn(conn, close=True)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    conn = None
                print(f"[PostgreSQL NonQuery Error]: {e}")
                return 0
            finally:
                if conn:
                    try:
                        self.pg_pool.putconn(conn)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
        else:
            conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                cur = conn.execute(query, params)
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def _init_db(self):
        real_type = "DOUBLE PRECISION" if self.is_postgres else "REAL"

        # 1. Users Table
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(255) PRIMARY KEY,
            google_id VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(255),
            avatar_url TEXT,
            created_at {real_type} NOT NULL,
            last_login {real_type} NOT NULL
        );
        """)

        # 2. Document Chunks Table (Preserves 100% full chunk evidence)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id VARCHAR(100) PRIMARY KEY,
            doc_id VARCHAR(50) NOT NULL,
            user_id VARCHAR(50) NOT NULL,
            chunk_index INTEGER DEFAULT 1,
            page_number INTEGER DEFAULT 1,
            content TEXT NOT NULL,
            created_at {real_type}
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);")
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks(user_id);")

        # 3. Documents Table (with category support)
        self._execute_non_query(f"""
CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            original_filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            category VARCHAR(50) DEFAULT 'General',
            summary_json TEXT DEFAULT '{{}}',
            created_at {real_type} NOT NULL
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);")

        # Column migrations
        if self.is_postgres:
            try:
                self._execute_non_query("ALTER TABLE documents ADD COLUMN IF NOT EXISTS category VARCHAR(50) DEFAULT 'General';")
            except Exception:
                pass
            try:
                self._execute_non_query("ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_json TEXT DEFAULT '{}';")
            except Exception:
                pass
        else:
            try:
                self._execute_non_query("ALTER TABLE documents ADD COLUMN category VARCHAR(50) DEFAULT 'General';")
            except Exception:
                pass
            try:
                self._execute_non_query("ALTER TABLE documents ADD COLUMN summary_json TEXT DEFAULT '{}';")
            except Exception:
                pass

        # 3. Conversations Table
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            created_at {real_type} NOT NULL,
            updated_at {real_type} NOT NULL
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);")

        # 4. Chat Messages Table
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS messages (
            id VARCHAR(255) PRIMARY KEY,
            conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            sender VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT DEFAULT '[]',
            created_at {real_type} NOT NULL
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);")

        # 5. Persistent Personal User Memory (Phase 3)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS user_memory (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            category VARCHAR(50) NOT NULL,
            fact_key VARCHAR(100) NOT NULL,
            fact_value TEXT NOT NULL,
            source_doc_title VARCHAR(255) DEFAULT '',
            created_at {real_type} NOT NULL,
            updated_at {real_type} NOT NULL
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(user_id);")

    # --- User Management ---
    def upsert_google_user(self, google_id: str, email: str, full_name: str, avatar_url: str) -> Dict[str, Any]:
        now = time.time()
        c_email = clean_utf8(email)
        c_name = clean_utf8(full_name)
        c_avatar = clean_utf8(avatar_url)

        existing = self._execute_scalar("SELECT * FROM users WHERE google_id = ? OR email = ?", (google_id, c_email))
        if existing:
            user_id = existing["id"]
            self._execute_non_query("""
                UPDATE users 
                SET google_id = ?, email = ?, full_name = ?, avatar_url = ?, last_login = ?
                WHERE id = ?
            """, (google_id, c_email, c_name, c_avatar, now, user_id))
        else:
            if google_id.startswith("gid_demo_"):
                user_id = f"demo_{uuid.uuid4().hex[:10]}"
            else:
                user_id = f"usr_{uuid.uuid4().hex[:12]}"
            self._execute_non_query("""
                INSERT INTO users (id, google_id, email, full_name, avatar_url, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, google_id, c_email, c_name, c_avatar, now, now))
        
        return self.get_user_by_id(user_id)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._execute_scalar("SELECT * FROM users WHERE id = ?", (user_id,))

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return self._execute_scalar("SELECT * FROM users WHERE email = ?", (clean_utf8(email).lower(),))

    def delete_user_account(self, user_id: str) -> bool:
        """Atomic single-transaction account wipe across all relational tables (Resolves P0-01, P0-02)."""
        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"DELETE FROM document_chunks WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM documents WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM user_memory WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = {p})", (user_id,))
            cur.execute(f"DELETE FROM conversations WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM users WHERE id = {p}", (user_id,))
            return cur.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """Alias for delete_user_account to maintain API contract (Resolves P0-01)."""
        return self.delete_user_account(user_id)

    # --- Document Management with Category Support ---
    def add_document(self, user_id: str, doc_id: str, title: str, original_filename: str,
                     file_type: str, file_size_bytes: int, chunk_count: int,
                     category: str = "General", summary: Dict[str, Any] = None) -> Dict[str, Any]:
        now = time.time()
        c_title = clean_utf8(title)
        c_orig = clean_utf8(original_filename)
        c_cat = clean_utf8(category) or "General"
        s_json = json.dumps(summary or {})

        self._execute_non_query("""
            INSERT INTO documents (id, user_id, title, original_filename, file_type, file_size_bytes, chunk_count, category, summary_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, user_id, c_title, c_orig, file_type, file_size_bytes, chunk_count, c_cat, s_json, now))

        return self.get_document(user_id, doc_id)

    def update_document_summary(self, user_id: str, doc_id: str, summary: Dict[str, Any]):
        s_json = json.dumps(summary or {})
        self._execute_non_query("""
            UPDATE documents SET summary_json = ?
            WHERE user_id = ? AND id = ?
        """, (s_json, user_id, doc_id))

    def get_user_documents(self, user_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if category and category.lower() not in ["all", ""]:
            rows = self._execute("""
                SELECT * FROM documents WHERE user_id = ? AND LOWER(category) = LOWER(?) ORDER BY created_at DESC
            """, (user_id, category.strip()))
        else:
            rows = self._execute("""
                SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC
            """, (user_id,))

        for r in rows:
            try:
                r["summary"] = json.loads(r.get("summary_json") or "{}")
            except Exception:
                r["summary"] = {}
        return rows

    def get_document(self, user_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        d = self._execute_scalar("""
            SELECT * FROM documents WHERE user_id = ? AND id = ?
        """, (user_id, doc_id))
        if d:
            try:
                d["summary"] = json.loads(d.get("summary_json") or "{}")
            except Exception:
                d["summary"] = {}
        return d

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        """Atomic transactional deletion of document, chunks, and extracted facts (Resolves P0-02, P0-05)."""
        with self.transaction() as conn:
            cur = conn.cursor()
            q_chunks = "DELETE FROM document_chunks WHERE user_id = %s AND doc_id = %s" if self.is_postgres else "DELETE FROM document_chunks WHERE user_id = ? AND doc_id = ?"
            cur.execute(q_chunks, (user_id, doc_id))
            
            q_mem = """
                DELETE FROM user_memory 
                WHERE user_id = %s AND source_doc_title = (SELECT title FROM documents WHERE id = %s AND user_id = %s)
            """ if self.is_postgres else """
                DELETE FROM user_memory 
                WHERE user_id = ? AND source_doc_title = (SELECT title FROM documents WHERE id = ? AND user_id = ?)
            """
            cur.execute(q_mem, (user_id, doc_id, user_id))

            q_doc = "DELETE FROM documents WHERE user_id = %s AND id = %s" if self.is_postgres else "DELETE FROM documents WHERE user_id = ? AND id = ?"
            cur.execute(q_doc, (user_id, doc_id))
            return cur.rowcount > 0

    # --- Conversation Sessions ---
    def create_conversation(self, user_id: str, title: str = "New Conversation") -> Dict[str, Any]:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = time.time()
        c_title = clean_utf8(title) or "New Conversation"
        self._execute_non_query("""
            INSERT INTO conversations (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (conv_id, user_id, c_title, now, now))
        return self.get_conversation(user_id, conv_id)

    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        return self._execute("""
            SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC
        """, (user_id,))

    def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._execute_scalar("""
            SELECT * FROM conversations WHERE user_id = ? AND id = ?
        """, (user_id, conversation_id))

    def update_conversation_title(self, user_id: str, conversation_id: str, title: str):
        c_title = clean_utf8(title) or "New Conversation"
        self._execute_non_query("""
            UPDATE conversations SET title = ?, updated_at = ?
            WHERE user_id = ? AND id = ?
        """, (c_title, time.time(), user_id, conversation_id))

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        count = self._execute_non_query("""
            DELETE FROM conversations WHERE user_id = ? AND id = ?
        """, (user_id, conversation_id))
        return count > 0

    # --- Chat Messages ---
    def add_message(self, conversation_id: str, sender: str, content: str, citations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = time.time()
        c_content = clean_utf8(content)
        cit_json = json.dumps(citations or [])

        self._execute_non_query("""
            INSERT INTO messages (id, conversation_id, sender, content, citations_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (msg_id, conversation_id, sender, c_content, cit_json, now))

        self._execute_non_query("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "sender": sender,
            "content": c_content,
            "citations": citations or [],
            "created_at": now
        }

    def get_conversation_messages(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        if not self.get_conversation(user_id, conversation_id):
            return []
        rows = self._execute("""
            SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC
        """, (conversation_id,))
        for r in rows:
            try:
                r["citations"] = json.loads(r.get("citations_json", "[]"))
            except Exception:
                r["citations"] = []
        return rows

    # --- 3-Layer Persistent User Memory (Phase 3) ---
    def add_user_memory(self, user_id: str, fact_key: str, fact_value: str, source_doc_title: str = "", category: str = "General") -> Dict[str, Any]:
        """Alias for upsert_user_memory ensuring API contract compatibility (Resolves P0-02)."""
        return self.upsert_user_memory(user_id=user_id, category=category, fact_key=fact_key, fact_value=fact_value, source_doc=source_doc_title)

    def upsert_user_memory(self, user_id: str, category: str, fact_key: str, fact_value: str, source_doc: str = "") -> Dict[str, Any]:
        now = time.time()
        c_cat = clean_utf8(category)
        c_key = clean_utf8(fact_key)
        c_val = clean_utf8(fact_value)
        c_src = clean_utf8(source_doc)

        existing = self._execute_scalar("""
            SELECT * FROM user_memory WHERE user_id = ? AND fact_key = ? AND source_doc_title = ?
        """, (user_id, c_key, c_src))

        if existing:
            mem_id = existing["id"]
            self._execute_non_query("""
                UPDATE user_memory 
                SET fact_value = ?, category = ?, source_doc_title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
            """, (c_val, c_cat, c_src, now, mem_id, user_id))
        else:
            mem_id = f"mem_{uuid.uuid4().hex[:10]}"
            self._execute_non_query("""
                INSERT INTO user_memory (id, user_id, category, fact_key, fact_value, source_doc_title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (mem_id, user_id, c_cat, c_key, c_val, c_src, now, now))

        return {
            "id": mem_id,
            "category": c_cat,
            "fact_key": c_key,
            "fact_value": c_val,
            "source_doc_title": c_src,
            "updated_at": now
        }

    def get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        return self._execute("""
            SELECT * FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC
        """, (user_id,))

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        count = self._execute_non_query("""
            DELETE FROM user_memory WHERE user_id = ? AND id = ?
        """, (user_id, memory_id))
        return count > 0

    # --- Strict Authorization & Ownership Helpers (P0 Security) ---
    def get_document(self, user_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._execute_scalar("""
            SELECT * FROM documents WHERE id = ? AND user_id = ?
        """, (doc_id, user_id))

    def get_conversation(self, user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
        return self._execute_scalar("""
            SELECT * FROM conversations WHERE id = ? AND user_id = ?
        """, (conv_id, user_id))

    def store_document_chunks(self, user_id: str, doc_id: str, chunks) -> int:
        now = time.time()
        stored = 0
        for c in chunks:
            page = c.metadata.get("page_number", 1)
            self._execute_non_query("""
                INSERT INTO document_chunks (chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO UPDATE SET content = EXCLUDED.content
            """ if self.is_postgres else """
                INSERT OR REPLACE INTO document_chunks (chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (c.chunk_id, doc_id, user_id, c.chunk_index, page, c.content, now))
            stored += 1
        return stored

    def get_document_full_chunks(self, user_id: str, doc_id: str) -> List[Dict[str, Any]]:
        return self._execute("""
            SELECT * FROM document_chunks WHERE doc_id = ? AND user_id = ? ORDER BY chunk_index ASC
        """, (doc_id, user_id))

    def get_full_chunk_text(self, chunk_id: str, user_id: str) -> Optional[str]:
        row = self._execute_scalar("""
            SELECT content FROM document_chunks WHERE chunk_id = ? AND user_id = ?
        """, (chunk_id, user_id))
        return row["content"] if row else None

    def update_user_memory(self, user_id: str, memory_id: str, fact_key: str, fact_value: str) -> bool:
        now = time.time()
        c_k = clean_utf8(fact_key)
        c_v = clean_utf8(fact_value)
        count = self._execute_non_query("""
            UPDATE user_memory SET fact_key = ?, fact_value = ?, created_at = ? WHERE id = ? AND user_id = ?
        """, (c_k, c_v, now, memory_id, user_id))
        return count > 0

    def clear_user_memories(self, user_id: str) -> bool:
        count = self._execute_non_query("""
            DELETE FROM user_memory WHERE user_id = ?
        """, (user_id,))
        return True

    def get_all_user_chunks(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves all authorized chunks for true independent sparse BM25 search (Resolves P1-03)."""
        return self._execute("""
            SELECT c.chunk_id, c.doc_id, c.page_number, c.content, COALESCE(d.title, 'Document') as doc_title, COALESCE(d.category, 'General') as category
            FROM document_chunks c
            LEFT JOIN documents d ON c.doc_id = d.id AND c.user_id = d.user_id
            WHERE c.user_id = ?
        """, (user_id,))
