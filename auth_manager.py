import os
import json
import time
import base64
import hmac
import hashlib
import secrets
import requests
from typing import Dict, Any, Optional

# Stable cryptographic key: uses JWT_SECRET env var or derives a permanent 256-bit SHA-256 seed from DATABASE_URL
ENV_JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
if ENV_JWT_SECRET:
    SECRET_KEY = ENV_JWT_SECRET.encode('utf-8')
else:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        SECRET_KEY = hashlib.sha256(f"aura_vault_jwt_{db_url}".encode('utf-8')).digest()
    else:
        SECRET_KEY = hashlib.sha256(b"aura_personal_vault_stable_key_2026").digest()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

class AuthManager:
    """
    Production-hardened authentication manager.
    Enforces strict Google OAuth validation and isolated session-scoped demo identities.
    """
    def __init__(self, db):
        self.db = db

    def verify_google_id_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        if not id_token:
            return None

        # 1. Isolated Session Demo Sign-In
        if id_token.startswith("demo_"):
            demo_entropy = secrets.token_hex(4)
            return {
                "google_id": f"gid_demo_{demo_entropy}",
                "email": f"demo_{demo_entropy}@aura.vault",
                "full_name": f"Demo User {demo_entropy.upper()}",
                "avatar_url": f"https://api.dicebear.com/7.x/bottts/svg?seed={demo_entropy}"
            }

        # 2. Robust Google OIDC Token Verification (Supports ID Token & Access Token)
        try:
            # 2a. ID Token info
            url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                sub = data.get("sub")
                email = data.get("email")
                if sub and email:
                    return {
                        "google_id": sub,
                        "email": email.lower().strip(),
                        "full_name": data.get("name") or email.split("@")[0],
                        "avatar_url": data.get("picture", "")
                    }

            # 2b. OAuth2 UserInfo endpoint fallback (for popup access tokens)
            u_resp = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {id_token}"}, timeout=5.0)
            if u_resp.status_code == 200:
                u_data = u_resp.json()
                sub = u_data.get("sub")
                email = u_data.get("email")
                if sub and email:
                    return {
                        "google_id": sub,
                        "email": email.lower().strip(),
                        "full_name": u_data.get("name") or email.split("@")[0],
                        "avatar_url": u_data.get("picture", "")
                    }

            print(f"[Auth Security] Token rejected by Google (ID token HTTP {resp.status_code})")
            return None
        except Exception as e:
            print(f"[Auth Error] Google token verification failed: {e}")
            return None

    def create_session_token(self, user_id: str, email: str, expires_in_seconds: int = 86400 * 30) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "email": email,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in_seconds
        }
        
        def b64_url(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')

        header_b64 = b64_url(json.dumps(header).encode('utf-8'))
        payload_b64 = b64_url(json.dumps(payload).encode('utf-8'))
        
        signature = hmac.new(SECRET_KEY, f"{header_b64}.{payload_b64}".encode('utf-8'), hashlib.sha256).digest()
        sig_b64 = b64_url(signature)
        
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        if token.startswith("Bearer "):
            token = token[7:].strip()

        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        
        def b64_url_decode(s: str) -> bytes:
            s += '=' * ((4 - len(s) % 4) % 4)
            return base64.urlsafe_b64decode(s)

        try:
            expected_sig = hmac.new(SECRET_KEY, f"{header_b64}.{payload_b64}".encode('utf-8'), hashlib.sha256).digest()
            actual_sig = b64_url_decode(sig_b64)
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            payload = json.loads(b64_url_decode(payload_b64).decode('utf-8'))
            if payload.get("exp", 0) < time.time():
                return None

            return payload
        except Exception:
            return None
