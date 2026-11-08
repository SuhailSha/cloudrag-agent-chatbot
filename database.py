from contextlib import contextmanager
import os
import sqlite3
import json
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("aura.database")

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


class DatabaseUnavailableError(Exception):
    """Raised when database connection pool or query fails due to infrastructure or operational error (Fixes H5)."""
    pass


class VaultDatabase:
    """
    Production Multi-Tenant Database Layer (Fixes C5, H5, H6, H9, M2):
    - Raises DatabaseUnavailableError (mapped to HTTP 503) whenever connection acquisition or SQL execution fails;
      never masks errors by returning [] or 0 (Fixes H5).
    - Persists revoked JWT tokens in `revoked_tokens` table so revocations survive restarts and scale across workers (Fixes C5).
    - Enforces FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE on `document_chunks` and UNIQUE(user_id, file_hash) (Fixes H6).
    - Tracks connection checkout timestamps to avoid running redundant `SELECT 1` pings on warm connections (Fixes H9).
    - Enforces 24-hour TTL expiry for `demo_` accounts and strict `google_id` matching without overwriting on email alone (Fixes M2).
    """
    @contextmanager
    def transaction(self):
        if self.is_postgres and self.pg_pool:
            conn = self._get_pg_conn()
            if not conn:
                raise DatabaseUnavailableError("Unable to acquire PostgreSQL connection from pool for transaction.")
            try:
                yield conn
                conn.commit()
                self._conn_last_used[id(conn)] = time.time()
            except Exception as e:
                try:
                    if not getattr(conn, 'closed', 0):
                        conn.rollback()
                except Exception:
                    pass
                if isinstance(e, DatabaseUnavailableError):
                    raise
                raise DatabaseUnavailableError(f"Transaction execution failed: {e}") from e
            finally:
                if conn:
                    try:
                        self.pg_pool.putconn(conn)
                    except Exception:
                        pass
        else:
            try:
                conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("PRAGMA journal_mode = WAL;")
            except Exception as e:
                raise DatabaseUnavailableError(f"SQLite connection failed: {e}") from e
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                if isinstance(e, (RuntimeError, ValueError, sqlite3.IntegrityError)):
                    raise
                raise DatabaseUnavailableError(f"SQLite transaction error: {e}") from e
            finally:
                conn.close()

    def __init__(self, db_url: str = DATABASE_URL, sqlite_path: str = SQLITE_DB_PATH):
        self.db_url = db_url
        self.sqlite_path = sqlite_path
        self.is_postgres = False
        self.pg_pool = None
        self._conn_last_used: Dict[int, float] = {}

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
                logger.info("Connected to Neon PostgreSQL connection pool.")
            except Exception as e:
                allow_sqlite = os.getenv("ALLOW_SQLITE_FALLBACK", "").strip().lower() in ["1", "true"]
                if not allow_sqlite:
                    logger.critical(f"Could not connect to production PostgreSQL: {e}")
                    raise RuntimeError(
                        f"Database Initialization Failed: Unable to establish connection to DATABASE_URL ({e}). "
                        "Failing closed to prevent silent data split-brain."
                    )
                logger.warning(f"Local development fallback to SQLite: {e}")
                self.is_postgres = False

        if not self.is_postgres:
            os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)
            logger.info(f"Using SQLite storage at '{self.sqlite_path}'.")

        self._init_db()

    def _get_pg_conn(self):
        if not (self.is_postgres and self.pg_pool):
            raise DatabaseUnavailableError("PostgreSQL connection pool is not initialized.")
        import psycopg2
        now = time.time()
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
                # Only ping with SELECT 1 if connection was idle for > 30 seconds (Fixes H9)
                last_ok = self._conn_last_used.get(id(conn), 0.0)
                if now - last_ok > 30.0:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                self._conn_last_used[id(conn)] = now
                return conn
            except Exception:
                if conn:
                    self._conn_last_used.pop(id(conn), None)
                    try:
                        self.pg_pool.putconn(conn, close=True)
                    except Exception:
                        pass
        try:
            direct_conn = psycopg2.connect(self.db_url, sslmode="require")
            self._conn_last_used[id(direct_conn)] = time.time()
            return direct_conn
        except Exception as e:
            logger.error(f"Database connection pool exhausted / unreachable: {e}")
            raise DatabaseUnavailableError(f"Database connection unavailable: {e}") from e

    def _execute(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        if self.is_postgres and self.pg_pool:
            import psycopg2.extras
            pg_query = query.replace("?", "%s")
            conn = None
            try:
                conn = self._get_pg_conn()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(pg_query, params)
                    conn.commit()
                    self._conn_last_used[id(conn)] = time.time()
                    if cur.description is None:
                        return []
                    rows = cur.fetchall()
                    return [dict(r) for r in rows]
            except DatabaseUnavailableError:
                raise
            except Exception as e:
                if conn:
                    self._conn_last_used.pop(id(conn), None)
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
                logger.error(f"PostgreSQL query error: {e}")
                raise DatabaseUnavailableError(f"Database query error: {e}") from e
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
            try:
                conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("PRAGMA journal_mode = WAL;")
            except Exception as e:
                raise DatabaseUnavailableError(f"SQLite connection error: {e}") from e
            try:
                cur = conn.execute(query, params)
                conn.commit()
                if cur.description is None:
                    return []
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                conn.rollback()
                logger.error(f"SQLite query error: {e}")
                raise DatabaseUnavailableError(f"Database query error: {e}") from e
            finally:
                conn.close()

    def _execute_scalar(self, query: str, params: tuple = ()):
        rows = self._execute(query, params)
        if not rows:
            return None
        first_row = rows[0]
        if isinstance(first_row, dict) and len(first_row) == 1:
            return list(first_row.values())[0]
        return first_row

    def _execute_non_query(self, query: str, params: tuple = ()) -> int:
        if self.is_postgres and self.pg_pool:
            pg_query = query.replace("?", "%s")
            conn = None
            try:
                conn = self._get_pg_conn()
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                    conn.commit()
                    self._conn_last_used[id(conn)] = time.time()
                    return cur.rowcount
            except DatabaseUnavailableError:
                raise
            except Exception as e:
                if conn:
                    self._conn_last_used.pop(id(conn), None)
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
                logger.error(f"PostgreSQL write error: {e}")
                raise DatabaseUnavailableError(f"Database write error: {e}") from e
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
            try:
                conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("PRAGMA journal_mode = WAL;")
            except Exception as e:
                raise DatabaseUnavailableError(f"SQLite connection error: {e}") from e
            try:
                cur = conn.execute(query, params)
                conn.commit()
                return cur.rowcount
            except Exception as e:
                conn.rollback()
                logger.error(f"SQLite write error: {e}")
                raise DatabaseUnavailableError(f"Database write error: {e}") from e
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

        # 2. Documents Table (Created BEFORE document_chunks so FOREIGN KEY constraint succeeds - Fixes H6)
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
            file_hash VARCHAR(64) DEFAULT '',
            created_at {real_type} NOT NULL,
            UNIQUE(user_id, file_hash)
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);")

        # 3. Document Chunks Table with Foreign Key Cascade (Fixes H6)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            doc_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            user_id VARCHAR(255) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            chunk_index INTEGER DEFAULT 1,
            page_number INTEGER DEFAULT 1,
            content TEXT NOT NULL,
            embedding_json TEXT DEFAULT '[]',
            created_at {real_type}
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);")
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks(user_id);")

        # Column migrations for existing PostgreSQL tables (SQLite CREATE TABLE already includes all columns)
        if self.is_postgres:
            for col_sql in [
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS category VARCHAR(50) DEFAULT 'General';",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_json TEXT DEFAULT '{}';",
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) DEFAULT '';"
            ]:
                try:
                    self._execute_non_query(col_sql)
                except Exception:
                    pass

        # 4. Conversations Table
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

        # 5. Chat Messages Table
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

        # 6. Persistent Personal User Memory
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS user_memory (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            category VARCHAR(50) NOT NULL,
            fact_key VARCHAR(100) NOT NULL,
            fact_value TEXT NOT NULL,
            source_doc_title VARCHAR(255) DEFAULT '',
            status VARCHAR(20) DEFAULT 'confirmed',
            created_at {real_type} NOT NULL,
            updated_at {real_type} NOT NULL
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(user_id);")

        # 7. Persistent Revoked Session Tokens Table (Fixes C5)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS revoked_tokens (
            token_hash VARCHAR(64) PRIMARY KEY,
            expires_at {real_type} NOT NULL,
            revoked_at {real_type} NOT NULL
        );
        """)

        # 8. Persistent Rate Limit Counters (Fixes H7)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS rate_limit_counters (
            bucket_key VARCHAR(255) PRIMARY KEY,
            window_start {real_type} NOT NULL,
            req_count INTEGER NOT NULL DEFAULT 1
        );
        """)

        # 9. Daily Per-User Token Budget Table (Fixes H7)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS token_usage_daily (
            usage_key VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            day_str VARCHAR(20) NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            updated_at {real_type} NOT NULL
        );
        """)

        # 10. Upload Job Status Tracking (Fixes H6)
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS upload_jobs (
            job_id VARCHAR(128) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'processing',
            total_files INTEGER NOT NULL DEFAULT 0,
            completed_files INTEGER NOT NULL DEFAULT 0,
            results_json TEXT DEFAULT '[]',
            created_at {real_type} NOT NULL,
            updated_at {real_type} NOT NULL
        );
        """)

        if self.is_postgres:
            for m_sql in [
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_json TEXT DEFAULT '[]';",
                "ALTER TABLE user_memory ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'confirmed';"
            ]:
                try:
                    self._execute_non_query(m_sql)
                except Exception:
                    pass

    # --- Token Revocation Persistence (Fixes C5) ---
    def revoke_token_hash(self, token_hash: str, expires_at: float) -> bool:
        now = time.time()
        q = """
            INSERT INTO revoked_tokens (token_hash, expires_at, revoked_at)
            VALUES (?, ?, ?)
            ON CONFLICT (token_hash) DO NOTHING
        """ if self.is_postgres else """
            INSERT OR IGNORE INTO revoked_tokens (token_hash, expires_at, revoked_at)
            VALUES (?, ?, ?)
        """
        self._execute_non_query(q, (token_hash, expires_at, now))
        # Prune expired revocations opportunistically
        self._execute_non_query("DELETE FROM revoked_tokens WHERE expires_at < ?", (now,))
        return True

    def is_token_hash_revoked(self, token_hash: str) -> bool:
        row = self._execute_scalar(
            "SELECT token_hash FROM revoked_tokens WHERE token_hash = ? AND expires_at > ?",
            (token_hash, time.time())
        )
        return row is not None

    # --- User Management & Demo Expiry (Fixes M2) ---
    def cleanup_expired_demo_users(self, ttl_seconds: int = 86400) -> List[str]:
        """Finds and deletes demo accounts older than 24 hours (Fixes M2)."""
        cutoff = time.time() - ttl_seconds
        expired_rows = self._execute(
            "SELECT id FROM users WHERE id LIKE 'demo_%' AND created_at < ?",
            (cutoff,)
        )
        expired_ids = [r["id"] for r in expired_rows if r.get("id")]
        for uid in expired_ids:
            try:
                self.delete_user_account(uid)
            except Exception as e:
                logger.warning(f"Failed cleaning expired demo user {uid}: {e}")
        return expired_ids

    def upsert_google_user(self, google_id: str, email: str, full_name: str, avatar_url: str) -> Dict[str, Any]:
        """
        Strictly matches by google_id first so an unverified/different Google sub cannot overwrite an existing
        user's google_id (Fixes M2).
        """
        now = time.time()
        c_email = clean_utf8(email).lower()[:250]
        c_name = clean_utf8(full_name)[:250]
        c_avatar = clean_utf8(avatar_url)[:1000]

        existing_by_gid = self._execute_scalar("SELECT * FROM users WHERE google_id = ?", (google_id,))
        if existing_by_gid:
            user_id = existing_by_gid["id"]
            self._execute_non_query("""
                UPDATE users
                SET email = ?, full_name = ?, avatar_url = ?, last_login = ?
                WHERE id = ?
            """, (c_email, c_name, c_avatar, now, user_id))
            return self.get_user_by_id(user_id)

        existing_by_email = self._execute_scalar("SELECT * FROM users WHERE email = ?", (c_email,))
        if existing_by_email:
            # Do NOT overwrite an existing user's google_id with a different google_id
            user_id = existing_by_email["id"]
            self._execute_non_query("""
                UPDATE users
                SET full_name = ?, avatar_url = ?, last_login = ?
                WHERE id = ?
            """, (c_name, c_avatar, now, user_id))
            return self.get_user_by_id(user_id)

        if google_id.startswith("gid_demo_"):
            user_id = f"demo_{uuid.uuid4().hex}"
        else:
            user_id = f"usr_{uuid.uuid4().hex}"

        self._execute_non_query("""
            INSERT INTO users (id, google_id, email, full_name, avatar_url, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, google_id, c_email, c_name, c_avatar, now, now))

        return self.get_user_by_id(user_id)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self._execute_scalar("SELECT * FROM users WHERE id = ?", (user_id,))
        if user and user_id.startswith("demo_"):
            # Enforce 24h TTL on demo accounts (Fixes M2)
            if time.time() - float(user.get("created_at", 0)) > 86400:
                try:
                    self.delete_user_account(user_id)
                except Exception:
                    pass
                return None
        return user

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return self._execute_scalar("SELECT * FROM users WHERE email = ?", (clean_utf8(email).lower(),))

    def count_user_messages(self, user_id: str) -> int:
        """Counts user-submitted questions across all conversations for demo quota enforcement (Fixes M2)."""
        val = self._execute_scalar("""
            SELECT COUNT(*) FROM messages
            WHERE sender = 'user' AND conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)
        """, (user_id,))
        return int(val or 0)

    def delete_user_account(self, user_id: str) -> bool:
        """Atomic single-transaction account wipe across all relational tables."""
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
        return self.delete_user_account(user_id)

    # --- Document Management ---
    def get_document_by_hash(self, user_id: str, file_hash: str) -> Optional[Dict[str, Any]]:
        if not file_hash:
            return None
        return self._execute_scalar("""
            SELECT * FROM documents WHERE user_id = ? AND file_hash = ?
        """, (user_id, file_hash))

    def add_document(
        self, user_id: str, doc_id: str, title: str, original_filename: str,
        file_type: str, file_size_bytes: int, chunk_count: int,
        category: str = "General", summary: Dict[str, Any] = None,
        file_hash: str = ""
    ) -> Dict[str, Any]:
        now = time.time()
        c_title = clean_utf8(title)[:250] or "Document"
        c_orig = clean_utf8(original_filename)[:250] or "Document"
        c_cat = (clean_utf8(category) or "General")[:48]
        c_type = (clean_utf8(file_type) or "txt")[:48]
        c_hash = clean_utf8(file_hash)[:64] or uuid.uuid4().hex
        s_json = json.dumps(summary or {})

        self._execute_non_query("""
            INSERT INTO documents (id, user_id, title, original_filename, file_type, file_size_bytes, chunk_count, category, summary_json, file_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, user_id, c_title, c_orig, c_type, file_size_bytes, chunk_count, c_cat, s_json, c_hash, now))

        return self.get_document(user_id, doc_id)

    def add_document_with_chunks(
        self, user_id: str, doc_id: str, title: str, original_filename: str,
        file_type: str, file_size_bytes: int, chunks: list,
        category: str = "General", summary: Dict[str, Any] = None,
        file_hash: str = ""
    ) -> Dict[str, Any]:
        """
        Inserts the parent `documents` row FIRST, followed by all `document_chunks` rows inside
        a single atomic SQL transaction (Fixes H6).
        """
        now = time.time()
        c_title = clean_utf8(title)[:250] or "Document"
        c_orig = clean_utf8(original_filename)[:250] or "Document"
        c_cat = (clean_utf8(category) or "General")[:48]
        c_type = (clean_utf8(file_type) or "txt")[:48]
        c_hash = clean_utf8(file_hash)[:64] or uuid.uuid4().hex
        s_json = json.dumps(summary or {})

        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(
                f"""
                INSERT INTO documents (id, user_id, title, original_filename, file_type, file_size_bytes, chunk_count, category, summary_json, file_hash, created_at)
                VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
                """,
                (doc_id, user_id, c_title, c_orig, c_type, file_size_bytes, len(chunks), c_cat, s_json, c_hash, now)
            )
            q_chunk = """
                INSERT INTO document_chunks (chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET content = EXCLUDED.content
            """ if self.is_postgres else """
                INSERT OR REPLACE INTO document_chunks (chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            for c in chunks:
                if isinstance(c, dict):
                    chunk_id = c.get("chunk_id", str(uuid.uuid4()))
                    chunk_index = c.get("chunk_index", 0)
                    content = c.get("content", "")
                    page = c.get("page_number", c.get("metadata", {}).get("page_number", 1))
                else:
                    chunk_id = getattr(c, "chunk_id", str(uuid.uuid4()))
                    chunk_index = getattr(c, "chunk_index", 0)
                    content = getattr(c, "content", "")
                    metadata = getattr(c, "metadata", {})
                    page = metadata.get("page_number", 1) if isinstance(metadata, dict) else getattr(c, "page_number", 1)
                cur.execute(q_chunk, (chunk_id, doc_id, user_id, chunk_index, page, content, now))

        return self.get_document(user_id, doc_id)

    def update_document_summary(self, user_id: str, doc_id: str, summary: Dict[str, Any]):
        s_json = json.dumps(summary or {})
        self._execute_non_query("""
            UPDATE documents SET summary_json = ?
            WHERE user_id = ? AND id = ?
        """, (s_json, user_id, doc_id))

    def update_document_category(self, user_id: str, doc_id: str, category: str) -> bool:
        c_cat = (clean_utf8(category) or "General")[:48]
        count = self._execute_non_query("""
            UPDATE documents SET category = ? WHERE user_id = ? AND id = ?
        """, (c_cat, user_id, doc_id))
        return count > 0

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
        """Atomic transactional deletion of document, chunks, and extracted facts."""
        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"DELETE FROM document_chunks WHERE user_id = {p} AND doc_id = {p}", (user_id, doc_id))
            cur.execute(
                f"DELETE FROM user_memory WHERE user_id = {p} AND source_doc_title = (SELECT title FROM documents WHERE id = {p} AND user_id = {p})",
                (user_id, doc_id, user_id)
            )
            cur.execute(f"DELETE FROM documents WHERE user_id = {p} AND id = {p}", (user_id, doc_id))
            return cur.rowcount > 0

    # --- Conversation Sessions ---
    def create_conversation(self, user_id: str, title: str = "New Conversation") -> Dict[str, Any]:
        conv_id = f"conv_{uuid.uuid4().hex}"
        now = time.time()
        c_title = (clean_utf8(title) or "New Conversation")[:250]
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
        c_title = (clean_utf8(title) or "New Conversation")[:250]
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
        msg_id = f"msg_{uuid.uuid4().hex}"
        now = time.time()
        c_content = clean_utf8(content)
        cit_json = json.dumps(citations or [])

        self._execute_non_query("""
            INSERT INTO messages (id, conversation_id, sender, content, citations_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (msg_id, conversation_id, sender[:20], c_content, cit_json, now))

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

    # --- Persistent User Memory ---
    def add_user_memory(self, user_id: str, fact_key: str, fact_value: str, source_doc_title: str = "", category: str = "General") -> Dict[str, Any]:
        return self.upsert_user_memory(user_id=user_id, category=category, fact_key=fact_key, fact_value=fact_value, source_doc=source_doc_title)

    def upsert_user_memory(self, user_id: str, category: str, fact_key: str, fact_value: str, source_doc: str = "", status: str = "confirmed") -> Dict[str, Any]:
        now = time.time()
        c_cat = (clean_utf8(category) or "General")[:48]
        c_key = clean_utf8(fact_key)[:95]
        c_val = clean_utf8(fact_value)[:2000]
        c_src = clean_utf8(source_doc)[:250]
        c_status = "pending" if status == "pending" else "confirmed"

        existing = self._execute_scalar("""
            SELECT * FROM user_memory WHERE user_id = ? AND fact_key = ? AND source_doc_title = ?
        """, (user_id, c_key, c_src))

        if existing:
            mem_id = existing["id"]
            self._execute_non_query("""
                UPDATE user_memory
                SET fact_value = ?, category = ?, source_doc_title = ?, status = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
            """, (c_val, c_cat, c_src, c_status, now, mem_id, user_id))
        else:
            mem_id = f"mem_{uuid.uuid4().hex}"
            self._execute_non_query("""
                INSERT INTO user_memory (id, user_id, category, fact_key, fact_value, source_doc_title, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (mem_id, user_id, c_cat, c_key, c_val, c_src, c_status, now, now))

        return {
            "id": mem_id,
            "category": c_cat,
            "fact_key": c_key,
            "fact_value": c_val,
            "source_doc_title": c_src,
            "status": c_status,
            "updated_at": now
        }

    def confirm_user_memory(self, user_id: str, memory_id: str) -> bool:
        count = self._execute_non_query("""
            UPDATE user_memory SET status = 'confirmed', updated_at = ? WHERE id = ? AND user_id = ?
        """, (time.time(), memory_id, user_id))
        return count > 0

    def get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        return self._execute("""
            SELECT * FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC
        """, (user_id,))

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        count = self._execute_non_query("""
            DELETE FROM user_memory WHERE user_id = ? AND id = ?
        """, (user_id, memory_id))
        return count > 0

    def update_user_memory(self, user_id: str, memory_id: str, fact_key: str, fact_value: str) -> bool:
        now = time.time()
        c_k = clean_utf8(fact_key)[:95]
        c_v = clean_utf8(fact_value)[:2000]
        count = self._execute_non_query("""
            UPDATE user_memory SET fact_key = ?, fact_value = ?, updated_at = ? WHERE id = ? AND user_id = ?
        """, (c_k, c_v, now, memory_id, user_id))
        return count > 0

    def clear_user_memories(self, user_id: str) -> bool:
        self._execute_non_query("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
        return True

    def store_document_chunks(self, user_id: str, doc_id: str, chunks) -> int:
        now = time.time()
        stored = 0
        with self.transaction() as conn:
            cur = conn.cursor()
            q = """
                INSERT INTO document_chunks (chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET content = EXCLUDED.content
            """ if self.is_postgres else """
                INSERT OR REPLACE INTO document_chunks (chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            for c in chunks:
                if isinstance(c, dict):
                    chunk_id = c.get("chunk_id", str(uuid.uuid4()))
                    chunk_index = c.get("chunk_index", 0)
                    content = c.get("content", "")
                    page = c.get("page_number", c.get("metadata", {}).get("page_number", 1))
                else:
                    chunk_id = getattr(c, "chunk_id", str(uuid.uuid4()))
                    chunk_index = getattr(c, "chunk_index", 0)
                    content = getattr(c, "content", "")
                    metadata = getattr(c, "metadata", {})
                    page = metadata.get("page_number", 1) if isinstance(metadata, dict) else getattr(c, "page_number", 1)
                cur.execute(q, (chunk_id, doc_id, user_id, chunk_index, page, content, now))
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
        if row is None:
            return None
        return row["content"] if isinstance(row, dict) else str(row)

    def get_all_user_chunks(self, user_id: str) -> List[Dict[str, Any]]:
        return self._execute("""
            SELECT c.chunk_id, c.doc_id, c.page_number, c.content, c.embedding_json, COALESCE(d.title, 'Document') as doc_title, COALESCE(d.category, 'General') as category
            FROM document_chunks c
            LEFT JOIN documents d ON c.doc_id = d.id AND c.user_id = d.user_id
            WHERE c.user_id = ?
        """, (user_id,))

    def update_chunk_embeddings(self, user_id: str, chunk_embeddings: Dict[str, List[float]]) -> int:
        """Persists 384-dim dense vectors in PostgreSQL/SQLite so dense search survives container restarts (Fixes H8)."""
        if not chunk_embeddings:
            return 0
        updated = 0
        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            for cid, vec in chunk_embeddings.items():
                vec_json = json.dumps([round(float(x), 5) for x in vec])
                cur.execute(
                    f"UPDATE document_chunks SET embedding_json = {p} WHERE chunk_id = {p} AND user_id = {p}",
                    (vec_json, cid, user_id)
                )
                updated += cur.rowcount
        return updated

    def check_and_increment_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Database-persisted rate limiting across instances and restarts (Fixes H7)."""
        now = time.time()
        row = self._execute_scalar("SELECT * FROM rate_limit_counters WHERE bucket_key = ?", (key,))
        if not row or (now - float(row.get("window_start", 0))) >= window_seconds:
            q = """
                INSERT INTO rate_limit_counters (bucket_key, window_start, req_count)
                VALUES (?, ?, 1)
                ON CONFLICT (bucket_key) DO UPDATE SET window_start = EXCLUDED.window_start, req_count = 1
            """ if self.is_postgres else """
                INSERT OR REPLACE INTO rate_limit_counters (bucket_key, window_start, req_count)
                VALUES (?, ?, 1)
            """
            self._execute_non_query(q, (key, now))
            return True
        current_count = int(row.get("req_count", 0))
        if current_count >= max_requests:
            return False
        self._execute_non_query(
            "UPDATE rate_limit_counters SET req_count = req_count + 1 WHERE bucket_key = ?",
            (key,)
        )
        return True

    def add_and_check_daily_tokens(self, user_id: str, tokens_to_add: int = 0, daily_limit: int = 50000) -> Tuple[bool, int]:
        """Tracks daily per-user LLM token consumption in the database (Fixes H7)."""
        day_str = time.strftime("%Y-%m-%d", time.gmtime())
        usage_key = f"{user_id}:{day_str}"
        now = time.time()
        row = self._execute_scalar("SELECT * FROM token_usage_daily WHERE usage_key = ?", (usage_key,))
        current_used = int(row.get("tokens_used", 0)) if row else 0
        if current_used >= daily_limit and tokens_to_add == 0:
            return False, current_used
        new_total = current_used + max(0, int(tokens_to_add))
        if tokens_to_add > 0:
            q = """
                INSERT INTO token_usage_daily (usage_key, user_id, day_str, tokens_used, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (usage_key) DO UPDATE SET tokens_used = EXCLUDED.tokens_used, updated_at = EXCLUDED.updated_at
            """ if self.is_postgres else """
                INSERT OR REPLACE INTO token_usage_daily (usage_key, user_id, day_str, tokens_used, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """
            self._execute_non_query(q, (usage_key, user_id, day_str, new_total, now))
        return new_total <= daily_limit, new_total

    def create_upload_job(self, user_id: str, total_files: int) -> str:
        job_id = f"job_{uuid.uuid4().hex}"
        now = time.time()
        self._execute_non_query("""
            INSERT INTO upload_jobs (job_id, user_id, status, total_files, completed_files, results_json, created_at, updated_at)
            VALUES (?, ?, 'processing', ?, 0, '[]', ?, ?)
        """, (job_id, user_id, total_files, now, now))
        return job_id

    def update_upload_job(self, job_id: str, user_id: str, completed_files: int, results: list, status: str = "processing"):
        self._execute_non_query("""
            UPDATE upload_jobs
            SET completed_files = ?, results_json = ?, status = ?, updated_at = ?
            WHERE job_id = ? AND user_id = ?
        """, (completed_files, json.dumps(results), status, time.time(), job_id, user_id))

    def get_upload_job(self, user_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        row = self._execute_scalar("SELECT * FROM upload_jobs WHERE job_id = ? AND user_id = ?", (job_id, user_id))
        if row:
            try:
                row["results"] = json.loads(row.get("results_json") or "[]")
            except Exception:
                row["results"] = []
        return row
