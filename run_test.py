import os, traceback, json, logging
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO)
print('Working dir:', os.getcwd())
print('OPENAI_API_KEY present in process:', bool(os.environ.get('OPENAI_API_KEY')))
try:
    from app.main import app
    client = TestClient(app)
    resp = client.post('/api/session', json={})
    print('session status:', resp.status_code)
    print('session json:', resp.json())
    sid = resp.json().get('session_id')
    payload = {'session_id': sid, 'messages': [{'role': 'user', 'content': 'Hello, can you introduce yourself?'}]}
    resp2 = client.post('/api/chat', json=payload)
    print('chat status:', resp2.status_code)
    print('chat json:', json.dumps(resp2.json(), indent=2))
except Exception:
    print('Exception during test call:')
    traceback.print_exc()
