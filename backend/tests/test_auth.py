"""Authentication and authorization regressions; no AI calls or database needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api import auth, routes
from app.database import auth as store
from app.database import sessions

USER = {"id": "user-1", "name": "Alex", "email": "alex@example.com"}
HEADERS = {"X-DocChat-Request": "1", "Origin": "http://localhost:5173"}


class PasswordTests(unittest.TestCase):
    def test_passwords_are_salted_and_verified(self):
        password = "a long passphrase 🔒"
        hashed = store.hash_password(password)
        self.assertNotEqual(hashed, store.hash_password(password))
        self.assertNotIn(password, hashed)
        self.assertTrue(store.verify_password(password, hashed))
        self.assertFalse(store.verify_password("incorrect", hashed))
        self.assertFalse(store.verify_password(password, "malformed"))

    def test_unknown_email_still_verifies_dummy_hash(self):
        with patch.object(store, "get_db_conn") as connection, patch.object(store, "verify_password", return_value=False) as verify:
            connection.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = None
            self.assertIsNone(store.authenticate("unknown@example.com", "password"))
            verify.assert_called_once_with("password", store.DUMMY_PASSWORD_HASH)

    def test_document_lookup_scopes_query_to_user(self):
        with patch.object(sessions, "get_db_conn") as connection:
            execute = connection.return_value.__enter__.return_value.execute
            execute.return_value.fetchone.return_value = None
            self.assertIsNone(sessions.get_session("document-1", "user-1"))
            query, params = execute.call_args.args
            self.assertIn("AND user_id = %s", query)
            self.assertEqual(params, ("document-1", "user-1"))


class AuthAPITests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(auth.router)
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.rate = patch.object(store, "allow_auth_attempt", return_value=True)
        self.rate.start()
        self.addCleanup(self.rate.stop)

    def test_signup_normalizes_email_sets_cookie_and_hides_password(self):
        with patch.object(store, "create_user", return_value=USER) as create, patch.object(store, "create_auth_session", return_value="opaque-token"):
            response = self.client.post('/auth/signup', headers=HEADERS, json={
                "name": " Alex ", "email": " Alex@Example.com ", "password": "long password 123",
            })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), USER)
        create.assert_called_once_with("Alex", "alex@example.com", "long password 123")
        cookie = response.headers['set-cookie']
        self.assertIn('HttpOnly', cookie)
        self.assertIn('SameSite=lax', cookie)
        self.assertIn('Max-Age=', cookie)
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_signup_validation(self):
        for override in ({"password": "short"}, {"email": "invalid"}, {"name": "  "}):
            body = {"name": "Alex", "email": "alex@example.com", "password": "long password 123", **override}
            with self.subTest(override=override):
                self.assertEqual(self.client.post('/auth/signup', headers=HEADERS, json=body).status_code, 422)

    def test_duplicate_signup(self):
        with patch.object(store, 'create_user', return_value=None):
            response = self.client.post('/auth/signup', headers=HEADERS, json={"name": "Alex", "email": "alex@example.com", "password": "long password 123"})
        self.assertEqual(response.status_code, 409)

    def test_incorrect_password_is_rejected(self):
        with patch.object(store, 'authenticate', return_value=None):
            response = self.client.post('/auth/login', headers=HEADERS, json={"email": "alex@example.com", "password": "incorrect"})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn('set-cookie', response.headers)

    def test_login_rotates_cookie_and_me_restores_user(self):
        self.client.cookies.set(auth.COOKIE_NAME, 'previous-token')
        with patch.object(store, 'authenticate', return_value=USER), patch.object(store, 'create_auth_session', return_value='new-token') as create:
            response = self.client.post('/auth/login', headers=HEADERS, json={"email": "alex@example.com", "password": "long password 123"})
        self.assertEqual(response.status_code, 200)
        create.assert_called_once_with(USER['id'], 'previous-token')
        with patch.object(store, 'get_authenticated_user', return_value=USER):
            self.assertEqual(self.client.get('/auth/me').json(), USER)

    def test_missing_or_expired_session(self):
        self.assertEqual(self.client.get('/auth/me').status_code, 401)
        self.client.cookies.set(auth.COOKIE_NAME, 'expired-token')
        with patch.object(store, 'get_authenticated_user', return_value=None):
            self.assertEqual(self.client.get('/auth/me').status_code, 401)

    def test_logout_revokes_session_and_clears_cookie(self):
        self.client.cookies.set(auth.COOKIE_NAME, 'old-token')
        with patch.object(store, 'revoke_auth_session') as revoke:
            response = self.client.post('/auth/logout', headers=HEADERS)
        self.assertEqual(response.status_code, 204)
        revoke.assert_called_once_with('old-token')
        self.assertIn('Max-Age=0', response.headers['set-cookie'])

    def test_csrf_checks(self):
        for headers in ({}, {**HEADERS, 'Origin': 'https://attacker.example'}):
            with self.subTest(headers=headers):
                self.assertEqual(self.client.post('/auth/logout', headers=headers).status_code, 403)

    def test_rate_limit(self):
        with patch.object(store, 'allow_auth_attempt', return_value=False):
            response = self.client.post('/auth/login', headers=HEADERS, json={"email": "alex@example.com", "password": "password"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['retry-after'], '900')

    def test_document_endpoints_require_authentication(self):
        responses = [
            self.client.get('/session/document-1'),
            self.client.delete('/session/document-1', headers=HEADERS),
            self.client.post('/chat', headers=HEADERS, json={"session_id": "document-1", "question": "Hello?"}),
            self.client.post('/upload', headers=HEADERS, files={"file": ("test.pdf", b"%PDF", "application/pdf")}),
        ]
        self.assertEqual([r.status_code for r in responses], [401] * 4)

    def test_other_users_document_is_inaccessible_and_never_processed(self):
        self.client.cookies.set(auth.COOKIE_NAME, 'valid-token')
        with patch.object(store, 'get_authenticated_user', return_value=USER), patch.object(routes, 'get_session', return_value=None) as lookup, patch.object(routes, 'answer_question') as answer, patch.object(routes, 'delete_session_chunks') as delete:
            self.assertEqual(self.client.get('/session/other-document').status_code, 404)
            self.assertEqual(self.client.delete('/session/other-document', headers=HEADERS).status_code, 404)
            self.assertEqual(self.client.post('/chat', headers=HEADERS, json={"session_id": "other-document", "question": "Hello?"}).status_code, 404)
            self.assertEqual(lookup.call_count, 3)
            lookup.assert_called_with('other-document', USER['id'])
            answer.assert_not_called()
            delete.assert_not_called()

    def test_upload_records_owner(self):
        self.client.cookies.set(auth.COOKIE_NAME, 'valid-token')
        with patch.object(store, 'get_authenticated_user', return_value=USER), patch.object(routes, 'build_vector_store', return_value=(1, 2, [])), patch.object(routes, 'save_session') as save:
            response = self.client.post('/upload', headers=HEADERS, files={"file": ("test.pdf", b"%PDF", "application/pdf")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(save.call_args.args[-1], USER['id'])


if __name__ == '__main__':
    unittest.main()
