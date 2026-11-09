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
    """Raised when database connection pool or query fails due to infrastructure outage (Mapped to HTTP 503)."""
    pass


class DatabaseIntegrityError(Exception):
    """Raised when a UNIQUE or FOREIGN KEY constraint is violated (Mapped to HTTP 409 Conflict - Fixes N4)."""
    pass


def _is_integrity_exception(exc: Exception) -> bool:
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    cls_name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(k in cls_name for k in ("integrityerror", "uniqueviolation", "foreignkeyviolation")):
        return True
    if any(k in msg for k in ("unique constraint", "duplicate key value", "foreign key constraint")):
        return True
    return False


class VaultDatabase:
    """
    Production Multi-Tenant Database Layer (Fixes N1, N2, N3, N4, N5, N10, N12):
    - Distinguishes DatabaseIntegrityError (409) from DatabaseUnavailableError (503) (Fixes N4).
    - Runs idempotent column, orphan-cleanup, deduplication, and constraint migrations on BOTH pre-existing
      SQLite and PostgreSQL databases (Fixes N3).
    - Keeps `get_all_user_chunks()` lightweight (text only, no 3.5 KB `embedding_json` per chunk) and exposes
      `get_user_chunk_embeddings()` separately for dense vector fallback (Fixes N2).
    - Wipes all 9 user tables (`upload_jobs`, `token_usage_daily`, `rate_limit_counters`, `document_chunks`,
      `documents`, `user_memory`, `messages`, `conversations`, `users`) in a single transaction on account deletion (Fixes N5).
    - Links `user_memory` to `source_doc_id` so deleting one document never deletes another document's facts with the same title (Fixes N12).
    - Uses single-statement atomic SQL upserts for rate limiting and daily token budgets, pruning expired rows (Fixes N1 & N10).
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
                if isinstance(e, (DatabaseUnavailableError, DatabaseIntegrityError, RuntimeError, ValueError)):
                    raise
                if _is_integrity_exception(e):
                    raise DatabaseIntegrityError(f"Database integrity constraint violated: {e}") from e
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
                if isinstance(e, (DatabaseUnavailableError, DatabaseIntegrityError, RuntimeError, ValueError)):
                    raise
                if _is_integrity_exception(e):
                    raise DatabaseIntegrityError(f"Database integrity constraint violated: {e}") from e
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
            pg_query = query.replace("%", "%%").replace("?", "%s")
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
            except (DatabaseUnavailableError, DatabaseIntegrityError):
                raise
            except Exception as e:
                if conn:
                    self._conn_last_used.pop(id(conn), None)
                    try:
                        if not getattr(conn, 'closed', 0):
                            conn.rollback()
                    except Exception:
                        pass
                    if _is_integrity_exception(e):
                        try:
                            self.pg_pool.putconn(conn)
                        except Exception:
                            pass
                        conn = None
                        raise DatabaseIntegrityError(f"Database integrity constraint violated: {e}") from e
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
                if _is_integrity_exception(e):
                    raise DatabaseIntegrityError(f"Database integrity constraint violated: {e}") from e
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
            pg_query = query.replace("%", "%%").replace("?", "%s")
            conn = None
            try:
                conn = self._get_pg_conn()
                with conn.cursor() as cur:
                    cur.execute(pg_query, params)
                    conn.commit()
                    self._conn_last_used[id(conn)] = time.time()
                    return cur.rowcount
            except (DatabaseUnavailableError, DatabaseIntegrityError):
                raise
            except Exception as e:
                if conn:
                    self._conn_last_used.pop(id(conn), None)
                    try:
                        if not getattr(conn, 'closed', 0):
                            conn.rollback()
                    except Exception:
                        pass
                    if _is_integrity_exception(e):
                        try:
                            self.pg_pool.putconn(conn)
                        except Exception:
                            pass
                        conn = None
                        raise DatabaseIntegrityError(f"Database integrity constraint violated: {e}") from e
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
                if _is_integrity_exception(e):
                    raise DatabaseIntegrityError(f"Database integrity constraint violated: {e}") from e
                logger.error(f"SQLite write error: {e}")
                raise DatabaseUnavailableError(f"Database write error: {e}") from e
            finally:
                conn.close()

    def _safe_add_column(self, table: str, col_name: str, col_def: str):
        """Idempotently adds a column on BOTH PostgreSQL and existing SQLite databases without logging false errors (Fixes N3)."""
        if self.is_postgres:
            try:
                self._execute_non_query(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col_name} {col_def};")
            except Exception:
                pass
        else:
            rows = self._execute(f"PRAGMA table_info({table});")
            existing_cols = {r.get("name") for r in rows}
            if col_name not in existing_cols:
                self._execute_non_query(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")

    def _migrate_existing_schema_and_constraints(self):
        """
        Upgrades pre-existing PostgreSQL and SQLite databases (Fixes N3):
        1. Adds any missing columns (`category`, `summary_json`, `file_hash`, `embedding_json`, `status`, `source_doc_id`).
        2. Deletes orphan rows in `document_chunks` whose `doc_id` no longer exists in `documents`.
        3. Backfills empty `file_hash` values with unique IDs and deduplicates any repeated `(user_id, file_hash)` rows.
        4. Enforces `UNIQUE INDEX idx_unique_user_file_hash ON documents(user_id, file_hash)` on both SQLite and Postgres,
           and adds the `fk_document_chunks_doc_id` FOREIGN KEY constraint on PostgreSQL if missing.
        """
        self._safe_add_column("users", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._safe_add_column("users", "last_login", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._safe_add_column("documents", "category", "VARCHAR(50) DEFAULT 'General'")
        self._safe_add_column("documents", "summary_json", "TEXT DEFAULT '{}'")
        self._safe_add_column("documents", "file_hash", "VARCHAR(64) DEFAULT ''")
        self._safe_add_column("documents", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._safe_add_column("document_chunks", "embedding_json", "TEXT DEFAULT '[]'")
        self._safe_add_column("document_chunks", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        self._safe_add_column("user_memory", "status", "VARCHAR(20) DEFAULT 'confirmed'")
        self._safe_add_column("user_memory", "source_doc_id", "VARCHAR(255) DEFAULT ''")
        self._safe_add_column("user_memory", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

        # 1. Clean up orphan document_chunks
        self._execute_non_query("""
            DELETE FROM document_chunks
            WHERE doc_id NOT IN (SELECT id FROM documents)
        """)

        # 2. Backfill empty file_hash with unique document id so UNIQUE(user_id, file_hash) never collides on empty strings
        self._execute_non_query("""
            UPDATE documents SET file_hash = id WHERE file_hash IS NULL OR file_hash = ''
        """)

        # 3. Remove duplicate (user_id, file_hash) rows deterministically
        self._execute_non_query("""
            DELETE FROM documents
            WHERE id IN (
                SELECT d1.id FROM documents d1
                JOIN documents d2
                  ON d1.user_id = d2.user_id
                 AND d1.file_hash = d2.file_hash
                 AND d1.id > d2.id
            )
        """)

        # 4. Enforce UNIQUE(user_id, file_hash) via unique index on both SQLite and PostgreSQL
        self._execute_non_query("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_user_file_hash ON documents(user_id, file_hash);
        """)

        if self.is_postgres:
            # Add foreign key constraint on pre-existing PostgreSQL document_chunks table if not already present
            try:
                self._execute_non_query("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint WHERE conname = 'fk_document_chunks_doc_id'
                        ) THEN
                            ALTER TABLE document_chunks
                            ADD CONSTRAINT fk_document_chunks_doc_id
                            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE;
                        END IF;
                    END $$;
                """)
            except Exception as e:
                logger.warning(f"PostgreSQL FK constraint check notice: {e}")
        else:
            # On pre-existing SQLite databases where document_chunks was created without a FOREIGN KEY,
            # rebuild document_chunks with REFERENCES documents(id) ON DELETE CASCADE (Fixes N3)
            fk_rows = self._execute("PRAGMA foreign_key_list(document_chunks);")
            has_doc_fk = any(r.get("table") == "documents" for r in fk_rows)
            if not has_doc_fk:
                with self.transaction() as conn:
                    conn.execute("""
                        CREATE TABLE document_chunks_new (
                            chunk_id VARCHAR(255) PRIMARY KEY,
                            doc_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                            user_id VARCHAR(255) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                            chunk_index INTEGER DEFAULT 1,
                            page_number INTEGER DEFAULT 1,
                            content TEXT NOT NULL,
                            embedding_json TEXT DEFAULT '[]',
                            created_at REAL
                        );
                    """)
                    conn.execute("""
                        INSERT INTO document_chunks_new (chunk_id, doc_id, user_id, chunk_index, page_number, content, embedding_json, created_at)
                        SELECT chunk_id, doc_id, user_id, chunk_index, page_number, content, COALESCE(embedding_json, '[]'), created_at
                        FROM document_chunks
                        WHERE doc_id IN (SELECT id FROM documents) AND user_id IN (SELECT id FROM users);
                    """)
                    conn.execute("DROP TABLE document_chunks;")
                    conn.execute("ALTER TABLE document_chunks_new RENAME TO document_chunks;")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_user ON document_chunks(user_id);")

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

        # 2. Documents Table
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

        # 3. Document Chunks Table with Foreign Key Cascade
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
            source_doc_id VARCHAR(255) DEFAULT '',
            source_doc_title VARCHAR(255) DEFAULT '',
            status VARCHAR(20) DEFAULT 'confirmed',
            created_at {real_type} NOT NULL,
            updated_at {real_type} NOT NULL
        );
        """)
        self._execute_non_query("CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(user_id);")

        # 7. Persistent Revoked Session Tokens Table
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS revoked_tokens (
            token_hash VARCHAR(64) PRIMARY KEY,
            expires_at {real_type} NOT NULL,
            revoked_at {real_type} NOT NULL
        );
        """)

        # 8. Persistent Rate Limit Counters
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS rate_limit_counters (
            bucket_key VARCHAR(255) PRIMARY KEY,
            window_start {real_type} NOT NULL,
            req_count INTEGER NOT NULL DEFAULT 1
        );
        """)

        # 9. Daily Per-User Token Budget Table
        self._execute_non_query(f"""
        CREATE TABLE IF NOT EXISTS token_usage_daily (
            usage_key VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            day_str VARCHAR(20) NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            updated_at {real_type} NOT NULL
        );
        """)

        # 10. Upload Job Status Tracking
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

        # Run idempotent schema & constraint migration for pre-existing tables (Fixes N3)
        self._migrate_existing_schema_and_constraints()

    # --- Token Revocation Persistence ---
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
        self._execute_non_query("DELETE FROM revoked_tokens WHERE expires_at < ?", (now,))
        return True

    def is_token_hash_revoked(self, token_hash: str) -> bool:
        row = self._execute_scalar(
            "SELECT token_hash FROM revoked_tokens WHERE token_hash = ? AND expires_at > ?",
            (token_hash, time.time())
        )
        return row is not None

    # --- User Management & Demo Expiry ---
    def cleanup_expired_demo_users(self, ttl_seconds: int = 86400) -> List[str]:
        cutoff = time.time() - ttl_seconds
        expired_rows = self._execute(
            "SELECT id FROM users WHERE id LIKE ? AND created_at < ?",
            ("demo_%", cutoff)
        )
        expired_ids = [r["id"] for r in expired_rows if r.get("id")]
        for uid in expired_ids:
            try:
                self.delete_user_account(uid)
            except Exception as e:
                logger.warning(f"Failed cleaning expired demo user {uid}: {e}")
        return expired_ids

    def upsert_google_user(self, google_id: str, email: str, full_name: str, avatar_url: str) -> Dict[str, Any]:
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
        val = self._execute_scalar("""
            SELECT COUNT(*) FROM messages
            WHERE sender = 'user' AND conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)
        """, (user_id,))
        return int(val or 0)

    def delete_user_account(self, user_id: str) -> bool:
        """
        Atomic single-transaction account wipe across ALL 9 user-linked tables (Fixes N5):
        Cleans `upload_jobs`, `token_usage_daily`, `rate_limit_counters`, `document_chunks`,
        `documents`, `user_memory`, `messages`, `conversations`, and `users`.
        """
        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"DELETE FROM upload_jobs WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM token_usage_daily WHERE user_id = {p}", (user_id,))
            cur.execute(f"DELETE FROM rate_limit_counters WHERE bucket_key LIKE {p}", (f"{user_id}:%",))
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
        c_hash = clean_utf8(file_hash)[:64] or doc_id
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
        now = time.time()
        c_title = clean_utf8(title)[:250] or "Document"
        c_orig = clean_utf8(original_filename)[:250] or "Document"
        c_cat = (clean_utf8(category) or "General")[:48]
        c_type = (clean_utf8(file_type) or "txt")[:48]
        c_hash = clean_utf8(file_hash)[:64] or doc_id
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
        """
        Atomic transactional deletion of document, chunks, and extracted facts.
        Matches user_memory by `source_doc_id` first (so two docs with the same title never delete each other's facts - Fixes N12).
        """
        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            cur.execute(f"DELETE FROM document_chunks WHERE user_id = {p} AND doc_id = {p}", (user_id, doc_id))
            cur.execute(
                f"""
                DELETE FROM user_memory
                WHERE user_id = {p}
                  AND (
                      source_doc_id = {p}
                      OR (
                          (source_doc_id IS NULL OR source_doc_id = '')
                          AND source_doc_title = (SELECT title FROM documents WHERE id = {p} AND user_id = {p})
                      )
                  )
                """,
                (user_id, doc_id, doc_id, user_id)
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
    def add_user_memory(self, user_id: str, fact_key: str, fact_value: str, source_doc_title: str = "", category: str = "General", source_doc_id: str = "") -> Dict[str, Any]:
        return self.upsert_user_memory(user_id=user_id, category=category, fact_key=fact_key, fact_value=fact_value, source_doc=source_doc_title, source_doc_id=source_doc_id)

    def upsert_user_memory(
        self, user_id: str, category: str, fact_key: str, fact_value: str,
        source_doc: str = "", status: str = "confirmed", source_doc_id: str = ""
    ) -> Dict[str, Any]:
        now = time.time()
        c_cat = (clean_utf8(category) or "General")[:48]
        c_key = clean_utf8(fact_key)[:95]
        c_val = clean_utf8(fact_value)[:2000]
        c_src = clean_utf8(source_doc)[:250]
        c_doc_id = clean_utf8(source_doc_id)[:250]
        c_status = "pending" if status == "pending" else "confirmed"

        if c_doc_id:
            existing = self._execute_scalar("""
                SELECT * FROM user_memory WHERE user_id = ? AND fact_key = ? AND source_doc_id = ?
            """, (user_id, c_key, c_doc_id))
        else:
            existing = self._execute_scalar("""
                SELECT * FROM user_memory WHERE user_id = ? AND fact_key = ? AND source_doc_title = ?
            """, (user_id, c_key, c_src))

        if existing:
            mem_id = existing["id"]
            self._execute_non_query("""
                UPDATE user_memory
                SET fact_value = ?, category = ?, source_doc_title = ?, source_doc_id = ?, status = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
            """, (c_val, c_cat, c_src, c_doc_id, c_status, now, mem_id, user_id))
        else:
            mem_id = f"mem_{uuid.uuid4().hex}"
            self._execute_non_query("""
                INSERT INTO user_memory (id, user_id, category, fact_key, fact_value, source_doc_id, source_doc_title, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (mem_id, user_id, c_cat, c_key, c_val, c_doc_id, c_src, c_status, now, now))

        return {
            "id": mem_id,
            "category": c_cat,
            "fact_key": c_key,
            "fact_value": c_val,
            "source_doc_id": c_doc_id,
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
            SELECT chunk_id, doc_id, user_id, chunk_index, page_number, content, created_at
            FROM document_chunks WHERE doc_id = ? AND user_id = ? ORDER BY chunk_index ASC
        """, (doc_id, user_id))

    def get_full_chunk_text(self, chunk_id: str, user_id: str) -> Optional[str]:
        row = self._execute_scalar("""
            SELECT content FROM document_chunks WHERE chunk_id = ? AND user_id = ?
        """, (chunk_id, user_id))
        if row is None:
            return None
        return row["content"] if isinstance(row, dict) else str(row)

    def get_all_user_chunks(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Fetches ONLY text and lightweight metadata for BM25 sparse search — NEVER selects `embedding_json`
        so per-question DB payload stays small (~0.9 KB/chunk instead of ~4.5 KB/chunk - Fixes N2).
        """
        return self._execute("""
            SELECT c.chunk_id, c.doc_id, c.page_number, c.content, COALESCE(d.title, 'Document') as doc_title, COALESCE(d.category, 'General') as category
            FROM document_chunks c
            LEFT JOIN documents d ON c.doc_id = d.id AND c.user_id = d.user_id
            WHERE c.user_id = ?
        """, (user_id,))

    def get_user_chunk_embeddings(self, user_id: str, allowed_doc_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        Dedicated query used ONLY when dense vector fallback is needed (Fixes N2 & N6).
        """
        rows = self._execute("""
            SELECT c.chunk_id, c.doc_id, c.page_number, c.content, c.embedding_json,
                   COALESCE(d.title, 'Document') as doc_title, COALESCE(d.category, 'General') as category
            FROM document_chunks c
            LEFT JOIN documents d ON c.doc_id = d.id AND c.user_id = d.user_id
            WHERE c.user_id = ? AND c.embedding_json IS NOT NULL AND c.embedding_json != '[]'
        """, (user_id,))
        if allowed_doc_ids is not None:
            rows = [r for r in rows if r.get("doc_id") in allowed_doc_ids]
        return rows

    def update_chunk_embeddings(self, user_id: str, chunk_embeddings: Dict[str, List[float]]) -> int:
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
        """
        Single-transaction atomic rate-limit check & increment with expired-row pruning (Fixes N1 & N10).
        """
        now = time.time()
        expiry_cutoff = now - max(window_seconds * 2, 300)
        with self.transaction() as conn:
            cur = conn.cursor()
            p = "%s" if self.is_postgres else "?"
            # Prune expired rate-limit rows so spoofed/old keys never accumulate (Fixes N1)
            cur.execute(f"DELETE FROM rate_limit_counters WHERE window_start < {p}", (expiry_cutoff,))

            if self.is_postgres:
                cur.execute(
                    """
                    INSERT INTO rate_limit_counters (bucket_key, window_start, req_count)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (bucket_key) DO UPDATE
                    SET req_count = CASE
                        WHEN (%s - rate_limit_counters.window_start) >= %s THEN 1
                        ELSE rate_limit_counters.req_count + 1
                    END,
                    window_start = CASE
                        WHEN (%s - rate_limit_counters.window_start) >= %s THEN %s
                        ELSE rate_limit_counters.window_start
                    END
                    RETURNING req_count;
                    """,
                    (key, now, now, window_seconds, now, window_seconds, now)
                )
                row = cur.fetchone()
                count = int(row[0]) if row else 1
                return count <= max_requests
            else:
                cur.execute(
                    """
                    INSERT INTO rate_limit_counters (bucket_key, window_start, req_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT (bucket_key) DO UPDATE
                    SET req_count = CASE
                        WHEN (? - rate_limit_counters.window_start) >= ? THEN 1
                        ELSE rate_limit_counters.req_count + 1
                    END,
                    window_start = CASE
                        WHEN (? - rate_limit_counters.window_start) >= ? THEN ?
                        ELSE rate_limit_counters.window_start
                    END;
                    """,
                    (key, now, now, window_seconds, now, window_seconds, now)
                )
                cur.execute("SELECT req_count FROM rate_limit_counters WHERE bucket_key = ?", (key,))
                row = cur.fetchone()
                count = int(row[0]) if row else 1
                return count <= max_requests

    def add_and_check_daily_tokens(self, user_id: str, tokens_to_add: int = 0, daily_limit: Optional[int] = None) -> Tuple[bool, int]:
        """
        Atomic daily token budget counter with tiered limits (50,000 for demo users, 250,000 for Google users - Fixes N10).
        """
        if daily_limit is None:
            daily_limit = 50000 if str(user_id).startswith("demo_") else 250000

        day_str = time.strftime("%Y-%m-%d", time.gmtime())
        usage_key = f"{user_id}:{day_str}"
        now = time.time()
        add_val = max(0, int(tokens_to_add))

        if add_val == 0:
            row = self._execute_scalar("SELECT tokens_used FROM token_usage_daily WHERE usage_key = ?", (usage_key,))
            used = int(row or 0) if not isinstance(row, dict) else int(row.get("tokens_used", 0))
            return used < daily_limit, used

        with self.transaction() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute(
                    """
                    INSERT INTO token_usage_daily (usage_key, user_id, day_str, tokens_used, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (usage_key) DO UPDATE
                    SET tokens_used = token_usage_daily.tokens_used + EXCLUDED.tokens_used,
                        updated_at = EXCLUDED.updated_at
                    RETURNING tokens_used;
                    """,
                    (usage_key, user_id, day_str, add_val, now)
                )
                row = cur.fetchone()
                total = int(row[0]) if row else add_val
            else:
                cur.execute(
                    """
                    INSERT INTO token_usage_daily (usage_key, user_id, day_str, tokens_used, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (usage_key) DO UPDATE
                    SET tokens_used = token_usage_daily.tokens_used + excluded.tokens_used,
                        updated_at = excluded.updated_at;
                    """,
                    (usage_key, user_id, day_str, add_val, now)
                )
                cur.execute("SELECT tokens_used FROM token_usage_daily WHERE usage_key = ?", (usage_key,))
                row = cur.fetchone()
                total = int(row[0]) if row else add_val

        return total <= daily_limit, total

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
