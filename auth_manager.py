import os
import time
import json
import base64
import hmac
import hashlib
import secrets
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger("aura.auth")

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


def get_production_jwt_secret() -> bytes:
    """
    Unconditional Fail-Closed JWT Secret Management (Fixes C5):
    - In production (RENDER, DATABASE_URL, or ENV=production), requires an explicit
      JWT_SECRET of at least 32 characters and raises RuntimeError if absent or short.
    - Removes all credential-derivation or static fallbacks.
    """
    env_secret = os.getenv("JWT_SECRET", "").strip()
    is_prod = bool(
        os.getenv("DATABASE_URL") or
        os.getenv("RENDER") or
        os.getenv("ENV", "").lower() == "production"
    )

    if env_secret:
        if len(env_secret) < 32:
            raise RuntimeError(
                "Production Security Error: JWT_SECRET must be at least 32 characters long."
            )
        logger.info("Dedicated production JWT_SECRET initialized.")
        return env_secret.encode("utf-8")

    if is_prod:
        raise RuntimeError(
            "Production Security Error: Missing required environment variable 'JWT_SECRET'. "
            "Application startup halted to prevent forged session vulnerability. "
            "Configure a secure 32+ character JWT_SECRET in your environment variables."
        )

    logger.info("Local test environment: generated ephemeral 256-bit random key.")
    return secrets.token_bytes(32)


ENV_JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
IS_PRODUCTION = bool(
    os.getenv("DATABASE_URL") or
    os.getenv("RENDER") or
    os.getenv("ENV", "").lower() == "production"
)
SECRET_KEY = get_production_jwt_secret()
REVOKED_TOKENS = set()


class AuthManager:
    """
    Production Authentication & Session Manager (Fixes C4 & C5):
    - Unconditional fail-closed JWT secret enforcement (Fixes C5).
    - Database-persisted token revocation surviving restarts and multi-worker instances (Fixes C5).
    - Strict Google OIDC Audience, Issuer, Email-Verification, and Nonce validation without URL string interpolation (Fixes C4).
    - Cryptographically signed HS256 JWT tokens with jti, iss, aud, exp, iat, and sub claims.
    """
    def __init__(self, db=None):
        self.db = db
        self.client_id = get_configured_google_client_id()

    def verify_google_id_token(self, token_str: str, expected_nonce: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Validates Google credentials with cryptographic, audience, issuer, email, and nonce checks (Fixes C4).
        """
        if not token_str or not isinstance(token_str, str):
            return None

        clean_token = token_str.strip()

        # 1. Isolated Session Demo Sign-In
        if clean_token.startswith("demo_"):
            demo_entropy = secrets.token_hex(8)
            return {
                "google_id": f"gid_demo_{demo_entropy}",
                "email": f"demo_{demo_entropy}@aura.vault",
                "full_name": f"Demo User {demo_entropy[:6].upper()}",
                "avatar_url": f"https://api.dicebear.com/7.x/bottts/svg?seed={demo_entropy}"
            }

        # 2. Strict Google Token Verification (using params/headers, never raw URL interpolation - Fixes C4)
        try:
            id_resp = requests.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": clean_token},
                timeout=5.0
            )

            if id_resp.status_code == 200:
                data = id_resp.json()

                # Strict Exact Audience Validation
                token_aud = str(data.get("aud", "")).strip()
                if self.client_id and token_aud != self.client_id:
                    logger.warning(f"ID token audience mismatch: expected '{self.client_id}', got '{token_aud}'.")
                    return None

                # Strict Issuer Validation
                token_iss = str(data.get("iss", "")).strip()
                if token_iss not in ["accounts.google.com", "https://accounts.google.com"]:
                    logger.warning(f"Untrusted ID token issuer: '{token_iss}'.")
                    return None

                # Nonce Verification (Fixes C4)
                if expected_nonce:
                    token_nonce = str(data.get("nonce", "")).strip()
                    if not token_nonce or not hmac.compare_digest(token_nonce, expected_nonce.strip()):
                        logger.warning("ID token nonce mismatch or missing nonce claim.")
                        return None

                # Verified Email Enforcement
                email_verified = data.get("email_verified")
                if email_verified not in [True, "true", "True"]:
                    logger.warning("Google account email is not verified.")
                    return None

                # Expiry Check
                exp = int(data.get("exp", 0))
                if exp < time.time():
                    logger.warning("Google ID token has expired.")
                    return None

                sub = data.get("sub")
                email = data.get("email")
                if sub and email:
                    return {
                        "google_id": str(sub),
                        "email": str(email).lower().strip(),
                        "full_name": data.get("name") or str(email).split("@")[0],
                        "avatar_url": data.get("picture", "")
                    }

            # 2b. Access Token Verification (User Info + Token Info Exact Audience Check)
            u_resp = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {clean_token}"},
                timeout=5.0
            )

            if u_resp.status_code == 200:
                u_data = u_resp.json()
                sub = u_data.get("sub")
                email = u_data.get("email")
                email_ver = u_data.get("email_verified")

                if email_ver not in [True, "true", "True"]:
                    logger.warning("Access token email is not verified.")
                    return None

                chk_resp = requests.get(
                    "https://oauth2.googleapis.com/tokeninfo",
                    params={"access_token": clean_token},
                    timeout=5.0
                )
                if chk_resp.status_code != 200:
                    logger.warning(f"Tokeninfo rejected access token (HTTP {chk_resp.status_code}). Failing closed.")
                    return None

                chk_data = chk_resp.json()
                aud_candidates = [
                    str(chk_data.get("azp", "")).strip(),
                    str(chk_data.get("issued_to", "")).strip(),
                    str(chk_data.get("aud", "")).strip(),
                    str(chk_data.get("client_id", "")).strip()
                ]
                if self.client_id and self.client_id not in aud_candidates:
                    logger.warning(f"Access token audience mismatch: expected '{self.client_id}', got {aud_candidates}.")
                    return None

                if sub and email:
                    return {
                        "google_id": str(sub),
                        "email": str(email).lower().strip(),
                        "full_name": u_data.get("name") or str(email).split("@")[0],
                        "avatar_url": u_data.get("picture", "")
                    }

            logger.warning(f"Google token rejected (HTTP {id_resp.status_code} / {u_resp.status_code}).")
            return None

        except Exception as e:
            logger.error(f"Google token verification error: {e}")
            return None

    def create_session_token(self, user_id: str, email: str, expires_in_seconds: int = 86400 * 7) -> str:
        """
        Creates a cryptographically signed HMAC-SHA256 JWT with standard security claims and 7-day lifetime (Fixes C4).
        """
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "jti": secrets.token_hex(16),
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

    def revoke_session_token(self, token: str) -> bool:
        """
        Revokes a session token both in the persistent database table (`revoked_tokens`) and in-memory cache (Fixes C5).
        """
        if not token or not isinstance(token, str):
            return False
        token_hash = hashlib.sha256(token.strip().encode('utf-8')).hexdigest()
        REVOKED_TOKENS.add(token_hash)

        expires_at = time.time() + 86400 * 7
        try:
            parts = token.strip().split('.')
            if len(parts) == 3:
                payload_json = base64.urlsafe_b64decode((parts[1] + '=' * (4 - len(parts[1]) % 4)).encode('utf-8')).decode('utf-8')
                payload = json.loads(payload_json)
                expires_at = float(payload.get("exp", expires_at))
        except Exception:
            pass

        if self.db and hasattr(self.db, "revoke_token_hash"):
            try:
                self.db.revoke_token_hash(token_hash, expires_at)
            except Exception as e:
                logger.warning(f"Failed persisting token revocation to DB: {e}")
        return True

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Strictly verifies HMAC-SHA256 signature, header algorithm, persistent revocation status, and RFC 7519 claims.
        """
        if not token or not isinstance(token, str):
            return None

        clean_tok = token.strip()
        token_hash = hashlib.sha256(clean_tok.encode('utf-8')).hexdigest()
        if token_hash in REVOKED_TOKENS:
            return None

        if self.db and hasattr(self.db, "is_token_hash_revoked"):
            if self.db.is_token_hash_revoked(token_hash):
                REVOKED_TOKENS.add(token_hash)
                return None

        parts = clean_tok.split('.')
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        try:
            expected_sig = hmac.new(SECRET_KEY, f"{header_b64}.{payload_b64}".encode('utf-8'), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode('utf-8').rstrip('=')

            if not hmac.compare_digest(sig_b64, expected_sig_b64):
                return None

            header_json = base64.urlsafe_b64decode((header_b64 + '=' * (4 - len(header_b64) % 4)).encode('utf-8')).decode('utf-8')
            header = json.loads(header_json)
            if header.get("alg") != "HS256" or header.get("typ") != "JWT":
                return None

            payload_json = base64.urlsafe_b64decode((payload_b64 + '=' * (4 - len(payload_b64) % 4)).encode('utf-8')).decode('utf-8')
            payload = json.loads(payload_json)

            now = time.time()
            if payload.get("iss") != "aura-vault-auth":
                return None
            if payload.get("aud") != "aura-vault-users":
                return None

            sub = payload.get("sub")
            if not sub or not isinstance(sub, str) or not sub.strip():
                return None

            exp = payload.get("exp")
            if not isinstance(exp, (int, float)) or exp < now:
                return None

            iat = payload.get("iat")
            if not isinstance(iat, (int, float)) or iat > now + 60 or (now - iat) > (86400 * 31):
                return None

            return payload
        except Exception:
            return None
