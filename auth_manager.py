import os
import json
import time
import base64
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional

SECRET_KEY = os.getenv("JWT_SECRET", "aura_personal_vault_secret_key_2026").encode('utf-8')
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

class AuthManager:
    """
    Manages Google OAuth verification and zero-dependency signed JWT session tokens.
    """
    def __init__(self, db):
        self.db = db

    def verify_google_id_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """
        Verifies Google ID Token via Google's official OAuth2 tokeninfo endpoint.
        """
        if not id_token:
            return None
        try:
            url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            resp = requests.get(url, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                # Verify audience if GOOGLE_CLIENT_ID is configured
                if GOOGLE_CLIENT_ID and data.get("aud") != GOOGLE_CLIENT_ID:
                    print(f"[Auth Warning] Audience mismatch: expected {GOOGLE_CLIENT_ID}, got {data.get('aud')}")
                
                return {
                    "google_id": data.get("sub"),
                    "email": data.get("email"),
                    "full_name": data.get("name") or data.get("email", "").split("@")[0],
                    "avatar_url": data.get("picture", "")
                }
            else:
                print(f"[Auth Warning] Google tokeninfo returned {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[Auth Error] Google verification failed: {e}")

        # Fallback for local testing / demo mode without Google API connection
        if id_token.startswith("demo_"):
            parts = id_token.split("_", 2)
            name = parts[1] if len(parts) > 1 else "Demo User"
            email = f"{name.lower().replace(' ', '')}@example.com"
            return {
                "google_id": f"gid_demo_{name.lower().replace(' ', '')}",
                "email": email,
                "full_name": name,
                "avatar_url": "https://api.dicebear.com/7.x/bottts/svg?seed=" + name
            }

        return None

    def create_session_token(self, user_id: str, email: str, expires_in_seconds: int = 86400 * 30) -> str:
        """
        Generates HMAC-SHA256 signed session token (30-day default expiry).
        """
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
        """
        Validates HMAC signature and expiration timestamp.
        """
        if not token:
            return None
        # Handle Bearer prefix
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
                return None # Expired

            return payload
        except Exception:
            return None
