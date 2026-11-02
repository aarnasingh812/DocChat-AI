"""Saved conversations, account isolation, and remembered follow-ups."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from app.api import routes
from app.api.auth import current_user
from app.database import sessions
from app.pipeline import generation

USER = {"id": "alice", "name": "Alice", "email": "alice@example.com"}
DOC = {"session_id": "doc-1", "doc_name": "report.pdf", "pages": 2, "chunks": 4}
NOW = datetime.now(timezone.utc)
RESULT = {"answer": "Revenue rose.", "sources": [{"page": 1, "excerpt": "Revenue"}], "elapsed": 0.5}
HEADERS = {"X-DocChat-Request": "1"}


class HistoryAPITests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.app.dependency_overrides[current_user] = lambda: USER
        self.client = TestClient(self.app)

    def test_history_requires_sign_in(self):
        self.app.dependency_overrides.clear()
        for path in ('/sessions', '/session/doc-1/messages'):
            self.assertEqual(self.client.get(path).status_code, 401)

    def test_history_uses_authenticated_owner(self):
        with patch.object(routes, 'list_sessions', return_value=[{**DOC, "created_at": NOW, "updated_at": NOW}]) as listing:
            response = self.client.get('/sessions?user_id=bob')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]['session_id'], 'doc-1')
        listing.assert_called_once_with('alice')

    def test_reopening_restores_message_content_citations_and_document(self):
        messages = [{"id": "m1", "role": "assistant", "content": RESULT['answer'],
                     "sources": RESULT['sources'], "elapsed": 0.5, "created_at": NOW}]
        with patch.object(routes, 'get_session', return_value=DOC), patch.object(routes, 'get_messages', return_value=messages) as load:
            # Opening again after a return visit reads the same server history.
            for _ in range(2):
                response = self.client.get('/session/doc-1/messages')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['doc_name'], 'report.pdf')
                self.assertEqual(response.json()['messages'][0]['sources'], RESULT['sources'])
            load.assert_called_with('doc-1', 'alice')

    def test_other_accounts_history_is_not_loaded(self):
        with patch.object(routes, 'get_session', return_value=None), patch.object(routes, 'get_messages') as load:
            self.assertEqual(self.client.get('/session/bobs-doc/messages').status_code, 404)
            load.assert_not_called()

    def test_chat_reads_past_turns_and_persists_successful_answer(self):
        history = [{"role": "user", "content": "What is revenue?"}]
        with patch.object(routes, 'get_session', return_value=DOC), patch.object(routes, 'get_messages', return_value=history) as load, patch.object(routes, 'answer_question', return_value=RESULT) as answer, patch.object(routes, 'save_turn') as save:
            response = self.client.post('/chat', headers=HEADERS, json={"session_id": "doc-1", "question": "How did it change?"})
        self.assertEqual(response.status_code, 200)
        load.assert_called_once_with('doc-1', 'alice', 12)
        answer.assert_called_once_with('doc-1', 'How did it change?', history)
        save.assert_called_once_with('doc-1', 'alice', 'How did it change?', RESULT)

    def test_failed_generation_does_not_write_partial_turn(self):
        with patch.object(routes, 'get_session', return_value=DOC), patch.object(routes, 'get_messages', return_value=[]), patch.object(routes, 'answer_question', side_effect=RuntimeError('unavailable')), patch.object(routes, 'save_turn') as save:
            response = self.client.post('/chat', headers=HEADERS, json={"session_id": "doc-1", "question": "Hi"})
        self.assertEqual(response.status_code, 500)
        save.assert_not_called()

    def test_persistence_failure_is_reported(self):
        with patch.object(routes, 'get_session', return_value=DOC), patch.object(routes, 'get_messages', return_value=[]), patch.object(routes, 'answer_question', return_value=RESULT), patch.object(routes, 'save_turn', side_effect=RuntimeError('DB unavailable')):
            response = self.client.post('/chat', headers=HEADERS, json={"session_id": "doc-1", "question": "Hi"})
        self.assertEqual(response.status_code, 500)
        self.assertIn('Unable to save', response.json()['detail'])


class HistoryStorageTests(unittest.TestCase):
    def test_message_query_scopes_owner_and_restores_chronological_order(self):
        with patch.object(sessions, 'get_db_conn') as factory:
            execute = factory.return_value.__enter__.return_value.execute
            execute.return_value.fetchall.return_value = [
                ('a', 'assistant', 'answer', [], 0.5, NOW),
                ('u', 'user', 'question', [], None, NOW),
            ]
            messages = sessions.get_messages('doc-1', 'alice', 12)
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        query, params = execute.call_args.args
        self.assertIn('s.user_id = %s', query)
        self.assertEqual(params, ('doc-1', 'alice', 12))

    def test_turn_writes_share_one_transaction_and_activity_update(self):
        with patch.object(sessions, 'get_db_conn') as factory:
            execute = factory.return_value.__enter__.return_value.execute
            execute.return_value.fetchone.return_value = ('doc-1',)
            messages = sessions.save_turn('doc-1', 'alice', 'Question', RESULT)
        factory.assert_called_once()
        self.assertEqual(execute.call_count, 3)
        self.assertIn('UPDATE sessions', execute.call_args_list[0].args[0])
        self.assertEqual(execute.call_args_list[0].args[1], ('doc-1', 'alice'))
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        self.assertEqual(messages[1]['sources'], RESULT['sources'])

    def test_missing_owner_cannot_write_messages(self):
        with patch.object(sessions, 'get_db_conn') as factory:
            execute = factory.return_value.__enter__.return_value.execute
            execute.return_value.fetchone.return_value = None
            with self.assertRaises(ValueError):
                sessions.save_turn('doc-1', 'bob', 'Question', RESULT)
            self.assertEqual(execute.call_count, 1)


class FollowUpTests(unittest.TestCase):
    def test_follow_up_uses_rewritten_query_for_same_document(self):
        llm = RunnableLambda(lambda value: AIMessage(content='How did revenue change?'))
        chain = MagicMock()
        chain.invoke.return_value = RESULT['answer']
        history = [{"role": "user", "content": "What is revenue?"}]
        with patch.object(generation, 'get_llm', return_value=llm), patch.object(generation, 'create_stuff_documents_chain', return_value=chain), patch.object(generation, 'hybrid_retrieve', return_value=[]) as retrieve, patch.object(generation, 'expand_context', return_value=[]):
            generation.answer_question('doc-1', 'How did it change?', history)
        retrieve.assert_called_once_with('doc-1', 'How did revenue change?', k_final=5, k_candidates=20)
        self.assertIn('What is revenue?', chain.invoke.call_args.args[0]['history'])
        self.assertEqual(chain.invoke.call_args.args[0]['input'], 'How did it change?')


if __name__ == '__main__':
    unittest.main()
