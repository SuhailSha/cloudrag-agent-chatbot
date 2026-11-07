import os
import time
import json
import base64
import hmac
import hashlib
import secrets
import requests
from typing import Optional, Dict, Any

DEFAULT_GOOGLE_CLIENT_ID = "277695292361-386p40ds1cptp6bf1f8v1sr8mr5tbv8v.apps.googleusercontent.com"

def get_configured_google_client_id() -> str:
    raw = (
        os.getenv("GOOGLE_CLIENT_ID") or 
        os.getenv("GOOGLE_OAUTH_CLIENT_ID") or 
        os.getenv("CLIENT_ID") or 
        DEFAULT_GOOGLE_CLIENT_ID
    ).strip()
    return raw.strip('"').strip("'").strip()

GOOGLE_CLIENT_ID = get_configured_google_client_id()

# Production JWT Key Management (Resolves H1 & P0-04)
ENV_JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
IS_PRODUCTION = bool(os.getenv("DATABASE_URL") or os.getenv("RENDER") or os.getenv("ENV", "").lower() == "production")

if ENV_JWT_SECRET:
    if len(ENV_JWT_SECRET) < 32 and IS_PRODUCTION:
        raise RuntimeError("Production Security Error: JWT_SECRET must be at least 32 characters long.")
    SECRET_KEY = ENV_JWT_SECRET.encode('utf-8')
    print("[Auth] Dedicated production JWT_SECRET initialized.")
else:
    if IS_PRODUCTION:
        raise RuntimeError(
            "Production Security Error: Missing required environment variable 'JWT_SECRET'. "
            "Application startup halted to prevent forged session vulnerability. "
            "Configure a secure 32+ character JWT_SECRET in your Render environment variables."
        )
    else:
        # In isolated local tests without production DB, generate an ephemeral random 256-bit key
        SECRET_KEY = secrets.token_bytes(32)
        print("[Auth Notice] Local test run: generated ephemeral 256-bit random key (no hardcoded fallback).")

class AuthManager:
    """
    Production-Hardened Authentication & Authorization Manager.
    Enforces:
    - Strict Google OIDC Audience, Issuer, and Email-Verification validation (Resolves P0-03).
    - Explicit Token-Type differentiation (ID Token vs Access Token).
    - Cryptographically signed JWT tokens with standard claims (iss, aud, exp, iat, sub) (Resolves P0-04).
    - Isolated demo sessions with strict zero-leakage partitions.
    """
    def __init__(self, db=None):
        self.db = db
        self.client_id = get_configured_google_client_id()

    def verify_google_id_token(self, token_str: str) -> Optional[Dict[str, Any]]:
        """
        Validates Google credentials with rigorous cryptographic and policy checks.
        Handles:
        1. Isolated Demo sessions (prefixed with 'demo_')
        2. Google OIDC ID Tokens (JWT verified by Google's tokeninfo API)
        3. Google OAuth 2.0 Access Tokens (strictly verified by Google's userinfo & tokeninfo APIs)
        """
        if not token_str or not isinstance(token_str, str):
            return None

        clean_token = token_str.strip()

        # 1. Isolated Session Demo Sign-In
        if clean_token.startswith("demo_"):
            demo_entropy = secrets.token_hex(4)
            return {
                "google_id": f"gid_demo_{demo_entropy}",
                "email": f"demo_{demo_entropy}@aura.vault",
                "full_name": f"Demo User {demo_entropy.upper()}",
                "avatar_url": f"https://api.dicebear.com/7.x/bottts/svg?seed={demo_entropy}"
            }

        # 2. Strict Google Token Verification
        try:
            # 2a. Attempt ID Token Verification
            id_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={clean_token}"
            id_resp = requests.get(id_url, timeout=5.0)

            if id_resp.status_code == 200:
                data = id_resp.json()
                
                # Strict Exact Audience Validation (Resolves P0-03)
                token_aud = data.get("aud", "").strip()
                if self.client_id and token_aud != self.client_id:
                    print(f"[Auth Security Violation] ID token audience mismatch: expected '{self.client_id}', received '{token_aud}'")
                    return None

                # Strict Issuer Validation
                token_iss = data.get("iss", "")
                if token_iss not in ["accounts.google.com", "https://accounts.google.com"]:
                    print(f"[Auth Security Violation] Untrusted token issuer: {token_iss}")
                    return None

                # Verified Email Enforcement
                email_verified = data.get("email_verified")
                if email_verified not in [True, "true", "True"]:
                    print("[Auth Security Violation] Google account email is not verified.")
                    return None

                # Expiry Check
                exp = int(data.get("exp", 0))
                if exp < time.time():
                    print("[Auth Security] Google ID token has expired.")
                    return None

                sub = data.get("sub")
                email = data.get("email")
                if sub and email:
                    return {
                        "google_id": sub,
                        "email": email.lower().strip(),
                        "full_name": data.get("name") or email.split("@")[0],
                        "avatar_url": data.get("picture", "")
                    }

            # 2b. Attempt Access Token Verification (User Info + Token Info Exact Audience Check)
            u_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            u_resp = requests.get(u_url, headers={"Authorization": f"Bearer {clean_token}"}, timeout=5.0)
            
            if u_resp.status_code == 200:
                u_data = u_resp.json()
                sub = u_data.get("sub")
                email = u_data.get("email")
                email_ver = u_data.get("email_verified")

                if email_ver not in [True, "true", "True"]:
                    print("[Auth Security Violation] Access token email is not verified.")
                    return None

                # Strict Tokeninfo Audience Verification - FAIL CLOSED (Resolves P0-03)
                chk_url = f"https://oauth2.googleapis.com/tokeninfo?access_token={clean_token}"
                chk_resp = requests.get(chk_url, timeout=5.0)
                if chk_resp.status_code != 200:
                    print(f"[Auth Security Violation] Tokeninfo rejected access token (HTTP {chk_resp.status_code}). Failing closed.")
                    return None

                chk_data = chk_resp.json()
                aud_candidates = [
                    str(chk_data.get("azp", "")).strip(),
                    str(chk_data.get("issued_to", "")).strip(),
                    str(chk_data.get("aud", "")).strip(),
                    str(chk_data.get("client_id", "")).strip()
                ]
                # Enforce exact client ID matching (Resolves P0-03)
                if self.client_id and self.client_id not in aud_candidates:
                    print(f"[Auth Security Violation] Access token audience mismatch: expected '{self.client_id}', got {aud_candidates}. Failing closed.")
                    return None

                if sub and email:
                    return {
                        "google_id": sub,
                        "email": email.lower().strip(),
                        "full_name": u_data.get("name") or email.split("@")[0],
                        "avatar_url": u_data.get("picture", "")
                    }

            print(f"[Auth Security] Token verification rejected by Google (HTTP {id_resp.status_code} / {u_resp.status_code})")
            return None

        except Exception as e:
            print(f"[Auth Error] Google token verification error: {e}")
            return None

    def create_session_token(self, user_id: str, email: str, expires_in_seconds: int = 86400 * 30) -> str:
        """
        Creates a cryptographically signed HMAC-SHA256 JWT with standard security claims.
        """
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": "aura-vault-auth",
            "aud": "aura-vault-users",
            "sub": str(user_id),
            "email": str(email),
            "iat": now,
            "exp": now + expires_in_seconds
        }
        
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode('utf-8')).decode('utf-8').rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode('utf-8')).decode('utf-8').rstrip('=')
        
        signature = hmac.new(SECRET_KEY, f"{header_b64}.{payload_b64}".encode('utf-8'), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
        
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Strictly verifies HMAC-SHA256 signature, header algorithm, and all required RFC 7519 claims:
        - alg == "HS256" and typ == "JWT"
        - iss == "aura-vault-auth"
        - aud == "aura-vault-users"
        - sub (non-empty user_id)
        - iat (present, numeric, within 30 days, not in future >60s)
        - exp (present, not expired)
        (Resolves P0-04 & P1-03)
        """
        if not token or not isinstance(token, str):
            return None
        parts = token.strip().split('.')
        if len(parts) != 3:
            return None
            
        header_b64, payload_b64, sig_b64 = parts
        try:
            # 1. Verify cryptographic signature first
            expected_sig = hmac.new(SECRET_KEY, f"{header_b64}.{payload_b64}".encode('utf-8'), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode('utf-8').rstrip('=')
            
            if not hmac.compare_digest(sig_b64, expected_sig_b64):
                return None
                
            # 2. Verify header structure & algorithm
            header_json = base64.urlsafe_b64decode((header_b64 + '=' * (4 - len(header_b64) % 4)).encode('utf-8')).decode('utf-8')
            header = json.loads(header_json)
            if header.get("alg") != "HS256" or header.get("typ") != "JWT":
                return None

            # 3. Verify payload claims
            payload_json = base64.urlsafe_b64decode((payload_b64 + '=' * (4 - len(payload_b64) % 4)).encode('utf-8')).decode('utf-8')
            payload = json.loads(payload_json)
            
            now = time.time()
            # Strict Claim Enforcement
            if payload.get("iss") != "aura-vault-auth":
                return None
            if payload.get("aud") != "aura-vault-users":
                return None
            if not payload.get("sub") or not isinstance(payload.get("sub"), str):
                return None
            if payload.get("exp") is None or not isinstance(payload.get("exp"), (int, float)) or payload.get("exp") < now:
                return None
            # iat is strictly REQUIRED (Resolves P1-03)
            iat = payload.get("iat")
            if iat is None or not isinstance(iat, (int, float)):
                return None
            if iat > now + 60 or iat < now - 86400 * 30:
                return None
                
            return payload
        except Exception:
            return None
