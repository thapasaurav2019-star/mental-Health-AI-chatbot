# Mental Health Chatbot

A lightweight FastAPI + SQLite chatbot focused on supportive, safe mental-health conversations. The backend enforces JSON-only LLM responses and a consent-first flow for optional exercises (like box breathing). A simple static frontend connects to the API.

## Tech stack
- Backend: FastAPI, SQLModel (SQLite), Uvicorn
- LLM: OpenAI API (gpt-4o-mini by default)
- Frontend: Vanilla HTML/JS (served statically)

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

## Project layout
```
app/            # FastAPI app
frontend/       # Static UI
requirements.txt
```

## Git & contributions
- Local SQLite (`chat.db`) and `.env` are ignored by default
- Feel free to open issues/PRs

## License
No license selected yet.
