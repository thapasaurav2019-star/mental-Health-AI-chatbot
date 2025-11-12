import re
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_signup_and_verify_dev_link():
    # Use a unique email each run
    email = "user_test_auth@example.com"
    r = client.post('/api/auth/signup', json={
        'name': 'Test User',
        'email': email,
        'password': 'secret123'
    })
    assert r.status_code == 200
    data = r.json()
    # In dev (no SMTP), we expect a dev_link in response
    assert 'dev_link' in data
    link = data['dev_link']
    assert '/api/auth/verify?token=' in link
    # Hit the verification link
    r2 = client.get(link.replace('http://127.0.0.1:8002', ''))  # remove host to use test client
    assert r2.status_code == 200
    assert 'Email verified' in r2.text
