# Mental Health Chatbot

A lightweight FastAPI + SQLite chatbot focused on supportive, safe mental-health conversations. The backend enforces JSON-only LLM responses and a consent-first flow for optional exercises (like box breathing). A simple static frontend connects to the API.

Repository: thapasaurav2019-star/mental-Health-AI-chatbot

## Tech stack
- Backend: FastAPI, SQLModel (SQLite), Uvicorn
- LLM: OpenAI API (gpt-4o-mini by default)
- Frontend: Vanilla HTML/JS (served statically)
- Authentication: Email verification with token-based system

## Quick start (Windows)

Prereqs:
- Python 3.11+
- An OpenAI API key in your environment (or set `.env`), e.g. `OPENAI_API_KEY=...`

Install deps:

```powershell
# From repo root
py -3 -m venv .venv
. .venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Run backend (port 8002):
```powershell
py -3 -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Run frontend (port 5500):
```powershell
# In a separate terminal
Set-Location -LiteralPath "frontend"
py -3 -m http.server 5500
```

Open http://127.0.0.1:5500/ in your browser. The UI will auto-detect the API. If needed, in DevTools Console:
```js
localStorage.removeItem('mh_api');
window.API = 'http://127.0.0.1:8002';
location.reload();
```

## Environment
- `OPENAI_API_KEY`: your OpenAI key
- `LLM_ONLY`: `1` to require LLM, `0` to allow fallback (dev only)

### Email Verification (Optional)
To enable email verification for new user accounts, configure these environment variables:
- `SMTP_HOST`: SMTP server hostname (default: `smtp.gmail.com`)
- `SMTP_PORT`: SMTP server port (default: `587`)
- `SMTP_USER`: SMTP username (your email address)
- `SMTP_PASSWORD`: SMTP password or app-specific password
- `FROM_EMAIL`: Sender email address (defaults to `SMTP_USER`)

**Note:** If SMTP is not configured, verification links will be logged to the console (useful for development/testing).

## Project layout
```
app/            # FastAPI app
frontend/       # Static UI
requirements.txt
```

## API Endpoints

### Authentication
- `POST /api/register` - Register a new user account
  - Request: `{"name": "string", "email": "string", "password": "string"}`
  - Sends verification email to the user
- `POST /api/verify-email` - Verify email with token
  - Request: `{"token": "string"}`
- `POST /api/resend-verification` - Resend verification email
  - Query: `?email=user@example.com`

### Chat & Utilities
- `POST /api/session` - Create a new chat session
- `POST /api/chat` - Send a chat message
- `GET /api/messages/{session_id}` - Get chat history
- `POST /api/emotion` - Detect emotion from text
- `POST /api/journal` - Add a journal entry
- `GET /api/journal/{session_id}` - List journal entries
- `GET /api/cbt_tips` - Get CBT tips
- `POST /api/escalate` - Escalate to human support/hotlines

## Git & contributions
- Local SQLite (`chat.db`) and `.env` are ignored by default
- Feel free to open issues/PRs

## License
No license selected yet.
