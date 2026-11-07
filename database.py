import os
import sqlite3
import json
import uuid
import time
from typing import Dict, List, Optional, Any

DB_PATH = os.getenv("SQLITE_DB_PATH", "./chroma_cloud_data/assistant_vault.db")

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
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);")

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
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,))
            row = cursor.fetchone()
            if row:
                user_id = row["id"]
                conn.execute("""
                    UPDATE users 
                    SET email = ?, full_name = ?, avatar_url = ?, last_login = ?
                    WHERE id = ?
                """, (email, full_name, avatar_url, now, user_id))
            else:
                user_id = f"usr_{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO users (id, google_id, email, full_name, avatar_url, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, google_id, email, full_name, avatar_url, now, now))
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
            cursor = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    # --- Document Management (Strict User Isolation) ---
    def add_document(self, user_id: str, doc_id: str, title: str, original_filename: str,
                     file_type: str, file_size_bytes: int, chunk_count: int) -> Dict[str, Any]:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (id, user_id, title, original_filename, file_type, file_size_bytes, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, user_id, title, original_filename, file_type, file_size_bytes, chunk_count, now))
            conn.commit()
        return self.get_document(user_id, doc_id)

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC
            """, (user_id,))
            return [dict(r) for r in cursor.fetchall()]

    def get_document(self, user_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM documents WHERE user_id = ? AND id = ?
            """, (user_id, doc_id))
            row = cursor.fetchone()
            return dict(row) if row else None

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
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (conv_id, user_id, title, now, now))
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
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE conversations SET title = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
            """, (title, time.time(), user_id, conversation_id))
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
        cit_json = json.dumps(citations or [])
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO messages (id, conversation_id, sender, content, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, conversation_id, sender, content, cit_json, now))
            # Bump conversation updated_at
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            conn.commit()
        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "citations": citations or [],
            "created_at": now
        }

    def get_conversation_messages(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        # Verifies user ownership before returning messages
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
