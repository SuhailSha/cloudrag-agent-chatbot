import os
import sqlite3
import json
import uuid
import time
from typing import Dict, List, Optional, Any

DB_PATH = os.getenv("SQLITE_DB_PATH", "./chroma_cloud_data/assistant_vault.db")

def clean_utf8(val: Any) -> str:
    """
    Sanitizes string by safely ignoring lone Unicode surrogate code points
    that would otherwise trigger SQLite 'UnicodeEncodeError: surrogates not allowed'.
    """
    if val is None:
        return ""
    s = str(val)
    return s.encode('utf-8', 'ignore').decode('utf-8', 'ignore').strip()

class VaultDatabase:
    """
    Multi-tenant enterprise database manager.
    Enforces strict tenant boundaries: every query requires user_id.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # 1. Users
            conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                google_id TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                avatar_url TEXT,
                created_at REAL NOT NULL,
                last_login REAL NOT NULL
            );
            """)

            # 2. User Documents
            conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                summary_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);")

            # Backward-compatible column migration
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN summary_json TEXT DEFAULT '{}';")
            except Exception:
                pass

            # 3. Conversations
            conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);")

            # 4. Chat Messages
            conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations_json TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);")
            conn.commit()

    # --- User Management ---
    def upsert_google_user(self, google_id: str, email: str, full_name: str, avatar_url: str) -> Dict[str, Any]:
        now = time.time()
        c_email = clean_utf8(email)
        c_name = clean_utf8(full_name)
        c_avatar = clean_utf8(avatar_url)
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,))
            row = cursor.fetchone()
            if row:
                user_id = row["id"]
                conn.execute("""
                    UPDATE users 
                    SET email = ?, full_name = ?, avatar_url = ?, last_login = ?
                    WHERE id = ?
                """, (c_email, c_name, c_avatar, now, user_id))
            else:
                user_id = f"usr_{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO users (id, google_id, email, full_name, avatar_url, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, google_id, c_email, c_name, c_avatar, now, now))
            conn.commit()
            
            user = self.get_user_by_id(user_id)
            return user

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE email = ?", (clean_utf8(email).lower(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def delete_user_account(self, user_id: str) -> bool:
        """Purges all user records, documents, and messages via cascading foreign keys."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Document Management (Strict User Isolation) ---
    def add_document(self, user_id: str, doc_id: str, title: str, original_filename: str,
                     file_type: str, file_size_bytes: int, chunk_count: int, summary: Dict[str, Any] = None) -> Dict[str, Any]:
        now = time.time()
        c_title = clean_utf8(title)
        c_orig = clean_utf8(original_filename)
        s_json = json.dumps(summary or {})
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (id, user_id, title, original_filename, file_type, file_size_bytes, chunk_count, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, user_id, c_title, c_orig, file_type, file_size_bytes, chunk_count, s_json, now))
            conn.commit()
        return self.get_document(user_id, doc_id)

    def update_document_summary(self, user_id: str, doc_id: str, summary: Dict[str, Any]):
        s_json = json.dumps(summary or {})
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE documents SET summary_json = ?
                WHERE user_id = ? AND id = ?
            """, (s_json, user_id, doc_id))
            conn.commit()

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC
            """, (user_id,))
            docs = []
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["summary"] = json.loads(d.get("summary_json") or "{}")
                except Exception:
                    d["summary"] = {}
                docs.append(d)
            return docs

    def get_document(self, user_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM documents WHERE user_id = ? AND id = ?
            """, (user_id, doc_id))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                try:
                    d["summary"] = json.loads(d.get("summary_json") or "{}")
                except Exception:
                    d["summary"] = {}
                return d
            return None

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM documents WHERE user_id = ? AND id = ?
            """, (user_id, doc_id))
            conn.commit()
            return cursor.rowcount > 0

    # --- Conversation Sessions (Strict User Isolation) ---
    def create_conversation(self, user_id: str, title: str = "New Conversation") -> Dict[str, Any]:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = time.time()
        c_title = clean_utf8(title) or "New Conversation"
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (conv_id, user_id, c_title, now, now))
            conn.commit()
        return self.get_conversation(user_id, conv_id)

    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC
            """, (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM conversations WHERE user_id = ? AND id = ?
            """, (user_id, conversation_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_conversation_title(self, user_id: str, conversation_id: str, title: str):
        c_title = clean_utf8(title) or "New Conversation"
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE conversations SET title = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
            """, (c_title, time.time(), user_id, conversation_id))
            conn.commit()

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM conversations WHERE user_id = ? AND id = ?
            """, (user_id, conversation_id))
            conn.commit()
            return cursor.rowcount > 0

    # --- Chat Messages ---
    def add_message(self, conversation_id: str, sender: str, content: str, citations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = time.time()
        c_content = clean_utf8(content)
        cit_json = json.dumps(citations or [])
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO messages (id, conversation_id, sender, content, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, conversation_id, sender, c_content, cit_json, now))
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            conn.commit()
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
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC
            """, (conversation_id,))
            rows = cursor.fetchall()
            messages = []
            for r in rows:
                m = dict(r)
                try:
                    m["citations"] = json.loads(m.get("citations_json", "[]"))
                except Exception:
                    m["citations"] = []
                messages.append(m)
            return messages
