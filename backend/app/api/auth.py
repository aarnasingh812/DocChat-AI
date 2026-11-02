"""Email/password endpoints and authentication dependency."""
import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from app.config import AUTH_COOKIE_SECURE, AUTH_SESSION_SECONDS, FRONTEND_ORIGINS
from app.database import auth as store

COOKIE_NAME = "docchat_session"


def require_trusted_origin(request: Request) -> None:
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        # A custom header forces preflight for browser requests, including uploads.
        if request.headers.get("x-docchat-request") != "1":
            raise HTTPException(403, "Missing request protection header.")
        origin = request.headers.get("origin")
        if origin and origin not in FRONTEND_ORIGINS:
            raise HTTPException(403, "Untrusted request origin.")


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    user = store.get_authenticated_user(token) if token else None
    if user is None:
        raise HTTPException(401, "Please sign in to continue.")
    return user


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address.")
        return value


class SignupRequest(Credentials):
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter your name.")
        return value.strip()


class UserResponse(BaseModel):
    id: str
    name: str
    email: str


router = APIRouter(prefix="/auth", tags=["Authentication"], dependencies=[Depends(require_trusted_origin)])


def check_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not store.allow_auth_attempt(client):
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.", headers={"Retry-After": "900"})


def start_session(response: Response, request: Request, user: dict) -> dict:
    token = store.create_auth_session(user["id"], request.cookies.get(COOKIE_NAME))
    response.set_cookie(COOKIE_NAME, token, max_age=AUTH_SESSION_SECONDS,
                        httponly=True, secure=AUTH_COOKIE_SECURE, samesite="lax", path="/")
    response.headers["Cache-Control"] = "no-store"
    return user


@router.post("/signup", response_model=UserResponse, status_code=201, dependencies=[Depends(check_rate_limit)])
def signup(body: SignupRequest, request: Request, response: Response):
    user = store.create_user(body.name, body.email, body.password)
    if user is None:
        raise HTTPException(409, "An account with this email already exists. Please sign in.")
    return start_session(response, request, user)


@router.post("/login", response_model=UserResponse, dependencies=[Depends(check_rate_limit)])
def login(body: Credentials, request: Request, response: Response):
    user = store.authenticate(body.email, body.password)
    if user is None:
        raise HTTPException(401, "Incorrect email or password.")
    return start_session(response, request, user)


@router.get("/me", response_model=UserResponse)
def me(response: Response, user: dict = Depends(current_user)):
    response.headers["Cache-Control"] = "no-store"
    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        store.revoke_auth_session(token)
    response.delete_cookie(COOKIE_NAME, path="/", secure=AUTH_COOKIE_SECURE, httponly=True, samesite="lax")
    response.headers["Cache-Control"] = "no-store"
