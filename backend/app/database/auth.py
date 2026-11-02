"""Account storage and revocable, hashed browser sessions."""
import hashlib
import hmac
import secrets
import uuid

from app.config import AUTH_SESSION_SECONDS
from app.database.connection import get_db_conn


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000)
    return f"pbkdf2_sha256$600000${salt}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False


# Avoid a fast, distinguishable path for unknown email addresses.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def init_auth_tables() -> None:
    with get_db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_attempts (
                client_key TEXT PRIMARY KEY,
                attempts INTEGER NOT NULL,
                window_start TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES users(id)")
        conn.execute("CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id)")


def allow_auth_attempt(client: str) -> bool:
    """Bound password work to 20 attempts per client per 15-minute window."""
    key = hashlib.sha256(client.encode()).hexdigest()
    with get_db_conn() as conn:
        conn.execute("DELETE FROM auth_attempts WHERE window_start < NOW() - INTERVAL '15 minutes'")
        row = conn.execute("""
            INSERT INTO auth_attempts (client_key, attempts) VALUES (%s, 1)
            ON CONFLICT (client_key) DO UPDATE SET attempts = auth_attempts.attempts + 1
            RETURNING attempts
        """, (key,)).fetchone()
    return row[0] <= 20


def create_user(name: str, email: str, password: str) -> dict | None:
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    with get_db_conn() as conn:
        row = conn.execute("""
            INSERT INTO users (id, name, email, password_hash) VALUES (%s, %s, %s, %s)
            ON CONFLICT (email) DO NOTHING RETURNING id, name, email
        """, (user_id, name, email, password_hash)).fetchone()
    return dict(zip(("id", "name", "email"), row)) if row else None


def authenticate(email: str, password: str) -> dict | None:
    with get_db_conn() as conn:
        row = conn.execute("SELECT id, name, email, password_hash FROM users WHERE email = %s", (email,)).fetchone()
    valid = verify_password(password, row[3] if row else DUMMY_PASSWORD_HASH)
    return dict(zip(("id", "name", "email"), row[:3])) if row and valid else None


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_auth_session(user_id: str, previous_token: str | None = None) -> str:
    token = secrets.token_urlsafe(32)
    with get_db_conn() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at <= NOW()")
        if previous_token:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = %s", (token_hash(previous_token),))
        conn.execute("""
            INSERT INTO auth_sessions (token_hash, user_id, expires_at)
            VALUES (%s, %s, NOW() + %s * INTERVAL '1 second')
        """, (token_hash(token), user_id, AUTH_SESSION_SECONDS))
    return token


def get_authenticated_user(token: str) -> dict | None:
    with get_db_conn() as conn:
        row = conn.execute("""
            SELECT u.id, u.name, u.email FROM users u
            JOIN auth_sessions s ON s.user_id = u.id
            WHERE s.token_hash = %s AND s.expires_at > NOW()
        """, (token_hash(token),)).fetchone()
    return dict(zip(("id", "name", "email"), row)) if row else None


def revoke_auth_session(token: str) -> None:
    with get_db_conn() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = %s", (token_hash(token),))
