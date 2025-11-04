import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_session():
    r = client.post('/api/session')
    assert r.status_code == 200
    return r.json()['session_id']


def test_positive_no_exercise():
    sid = create_session()
    r = client.post('/api/chat', json={'session_id': sid, 'messages': [{'role': 'user', 'content': "I'm feeling good"}]})
    assert r.status_code == 200
    j = r.json()
    reply = j.get('reply','').lower()
    assert 'exercise' not in reply and 'breath' not in reply
    assert any(x in reply for x in ['what', 'tell me', 'really', '?'])


def test_anxious_offers_option():
    sid = create_session()
    r = client.post('/api/chat', json={'session_id': sid, 'messages': [{'role': 'user', 'content': "I'm feeling anxious and can't sleep"}]})
    assert r.status_code == 200
    j = r.json()
    reply = j.get('reply','').lower()
    # should allow exercise suggestion or option
    assert 'exercise' in reply or 'breath' in reply or 'ground' in reply or 'option' in reply


def test_gibberish_clarify():
    sid = create_session()
    r = client.post('/api/chat', json={'session_id': sid, 'messages': [{'role': 'user', 'content': "asdasd asdhj"}]})
    assert r.status_code == 200
    j = r.json()
    reply = j.get('reply','')
    assert 'couldn' in reply.lower() or 'rephrase' in reply.lower() or 'didn' in reply.lower()


def test_crisis_detection():
    sid = create_session()
    r = client.post('/api/chat', json={'session_id': sid, 'messages': [{'role': 'user', 'content': "I want to kill myself"}]})
    assert r.status_code == 200
    j = r.json()
    assert j.get('crisis', False) is True
    assert '000' in j.get('reply','') or 'Lifeline' in j.get('reply','')
