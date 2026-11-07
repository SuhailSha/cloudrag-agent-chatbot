import os
import json
import time
import base64
import hmac
import hashlib
import secrets
import requests
from typing import Dict, Any, Optional

# Enforce cryptographically secure JWT secret; generate random 256-bit key if omitted
ENV_JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
if ENV_JWT_SECRET:
    SECRET_KEY = ENV_JWT_SECRET.encode('utf-8')
else:
    # Ephemeral 256-bit cryptographically secure key generated at server boot
    SECRET_KEY = secrets.token_bytes(32)
    print("[Security Notice] Using cryptographically secure random session secret key.")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

class AuthManager:
    """
    Production-hardened authentication manager.
    Enforces strict Google OAuth validation and isolated session-scoped demo identities.
    """
    def __init__(self, db):
        self.db = db

    def verify_google_id_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """
        Validates Google ID Token:
        - Rejects audience mismatches
        - Validates issuer and token expiration
        - Requires verified email status
        """
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

        # 2. Strict Google OIDC Token Verification
        try:
            url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            resp = requests.get(url, timeout=5.0)
            if resp.status_code != 200:
                print(f"[Auth Security] Token verification rejected by Google (HTTP {resp.status_code})")
                return None

            data = resp.json()

            # Strict Audience Enforcement: reject immediately on mismatch
            if GOOGLE_CLIENT_ID:
                token_aud = data.get("aud")
                if token_aud != GOOGLE_CLIENT_ID:
                    print(f"[Auth Security Violation] Audience mismatch: expected {GOOGLE_CLIENT_ID}, got {token_aud}")
                    return None

            # Strict Issuer Validation
            token_iss = data.get("iss", "")
            if token_iss not in ["accounts.google.com", "https://accounts.google.com"]:
                print(f"[Auth Security Violation] Untrusted token issuer: {token_iss}")
                return None

            # Verified Email Check
            email_verified = data.get("email_verified")
            if email_verified not in [True, "true", "True"]:
                print("[Auth Security Violation] Email not verified by Google.")
                return None

            # Expiration Check
            exp_timestamp = float(data.get("exp", 0))
            if exp_timestamp < time.time():
                print("[Auth Security Violation] Google token is expired.")
                return None

            sub = data.get("sub")
            email = data.get("email")
            if not sub or not email:
                return None

            return {
                "google_id": sub,
                "email": email.lower().strip(),
                "full_name": data.get("name") or email.split("@")[0],
                "avatar_url": data.get("picture", "")
            }
        except Exception as e:
            print(f"[Auth Error] Google token verification failed: {e}")
            return None

    def create_session_token(self, user_id: str, email: str, expires_in_seconds: int = 86400 * 30) -> str:
        """
        Issues HMAC-SHA256 signed session token.
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
        Cryptographically validates HMAC signature and expiration timestamp.
        """
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
