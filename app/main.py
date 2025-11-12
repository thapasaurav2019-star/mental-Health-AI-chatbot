# app/main.py
import os
# Load environment variables from a local .env file if present (non-fatal if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    # Prefer values from a local .env during development to avoid stale system envs overriding.
    # This helps when a previously set OPENAI_API_KEY in Windows is incorrect.
    load_dotenv(override=True)
except Exception:
    pass
import uuid
import random
from typing import List, Dict, Any
import re
try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker()
except Exception:
    _SPELL = None

from fastapi import FastAPI, Depends, HTTPException, Request, Query
import logging
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse, HTMLResponse
from sqlmodel import select, Session

from app.db import init_db, get_session
from app.db import engine
from app.models import ChatSession, Message, SafetyEvent, JournalEntry, User
from app.safety import looks_like_crisis, CRISIS_RESPONSE_AU, check_moderation
from app import tools as wellbeing_tools


# Small helper to infer a simple mood label from text so the frontend can show an emoji/label.
def detect_mood_from_text(text: str) -> str:
    if not text:
        return "neutral"
    t = text.lower()
    # negation or negative qualifiers that should NOT be treated as positive
    # examples: "not good", "not okay", "not great", "could be better", "worse than", "awful", "terrible", "bad"
    if re.search(r"\b(not\s+(?:good|okay|ok|great|fine|happy|alright|well)|not\s+so\s+good|could\s+be\s+better|feeling\s+worse|getting\s+worse|worse\b|awful\b|terrible\b|really\s+bad|so\s+bad|bad\b|meh\b|so-?so\b)\b", t):
        return "sad"
    # sadness
    if re.search(r"\b(sad|depress|down|miserable|hopeless|tear)\b", t):
        return "sad"
    # anxiety
    if re.search(r"\b(anxious|anxiety|panic|nervous|scared|worried)\b", t):
        return "anxious"
    # happiness / positive
    if re.search(r"\b(happy|good|great|awesome|fantastic|yay|glad)\b", t):
        return "happy"
    # gratitude
    if re.search(r"\b(thank|grateful|appreciate)\b", t):
        return "grateful"
    # encouraging / calming
    if re.search(r"\b(you can|you've got this|you got this|breathe|it's okay|it's ok)\b", t):
        return "encouraging"
    return "neutral"


def _is_gibberish(text: str) -> bool:
    """Heuristic to detect random letters / gibberish.
    Returns True when the text likely contains no meaningful words.
    """
    if not text or len(text.strip()) == 0:
        return True
    s = text.strip()
    # if too many non-alpha characters
    alpha = sum(1 for c in s if c.isalpha())
    if alpha / max(1, len(s)) < 0.5:
        return True
    # average vowel ratio across words
    words = [w for w in re.findall(r"[A-Za-z]+", s)]
    if not words:
        return True
    # detect obvious repeated-substring words like 'asdasd'
    for w in words:
        lw = w.lower()
        n = len(lw)
        for size in range(1, min(5, n//2 + 1)):
            if size > 0 and n % size == 0:
                sub = lw[:size]
                if sub * (n // size) == lw and (n // size) > 1:
                    return True
    # If a spellchecker is available, and most words are unknown, treat as gibberish
    try:
        if _SPELL:
            unknown = 0
            for w in words:
                if len(w) <= 2:
                    continue
                if _SPELL.correction(w) == w and w.lower() not in _SPELL:
                    unknown += 1
            if unknown / max(1, len(words)) > 0.5:
                return True
    except Exception:
        pass
    def vowel_frac(w):
        v = sum(1 for ch in w.lower() if ch in 'aeiou')
        return v / max(1, len(w))
    vf = sum(vowel_frac(w) for w in words) / len(words)
    if vf < 0.22 and sum(1 for w in words if len(w) <= 2) / len(words) > 0.5:
        return True
    return False


def _simple_normalize_text(text: str) -> str:
    """Lightweight normalization: collapse repeated characters and fix very common typos."""
    # collapse 3+ repeated letters to 2 (so loooove -> loove)
    s = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    # common typo fixes
    fixes = {"teh": "the", "recieve": "receive", "cant": "can't", "im": "I'm", "dont": "don't", "id": "I'd"}
    def fix_word(w):
        lw = w.lower()
        if lw in fixes:
            return fixes[lw]
        # use spellchecker if available (only simple corrections)
        if _SPELL and w.isalpha() and len(w) > 2:
            cand = _SPELL.correction(w)
            return cand or w
        return w
    s = ' '.join(fix_word(w) for w in s.split())
    return s


EXERCISE_KEYWORDS = [
    "exercise", "breath", "breathing", "box breathing", "ground", "grounding",
    "grounding cue", "tiny step", "small step", "short exercise", "quick exercise"
]

# --- Consent and tool helpers ---
def _is_affirmative(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.search(r"\b(yes|yep|yeah|yup|ok|okay|sure|pls|please|let's|lets|do it|go ahead|sounds good|absolutely|alright)\b", t))


def _is_negative(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.search(r"\b(no|nah|nope|not now|later|maybe later|another time|don't|do not|no thanks|pass|skip|not really)\b", t))


def _pretty_tool(name: str) -> str:
    m = {
        "breathing_box": "box breathing exercise",
        "grounding_54321": "5-4-3-2-1 grounding",
        "thought_record_prompt": "mini thought record",
        "cbt_tips": "CBT tips",
        "au_hotlines": "Australian support options",
    }
    return m.get(name, name.replace('_', ' '))


def _infer_tool_from_text(text: str | None) -> str | None:
    if not text:
        return None
    low = text.lower()
    # Prefer breathing when breath/box is present
    if ("breath" in low or "box" in low) and "ground" not in low:
        return "breathing_box"
    # Grounding patterns
    if "ground" in low or "5-4-3-2-1" in low or "54321" in low or "grounding cue" in low:
        return "grounding_54321"
    if "thought record" in low or ("cbt" in low and "tip" not in low):
        return "thought_record_prompt"
    if "tip" in low and "cbt" in low:
        return "cbt_tips"
    if "hotline" in low or "support" in low:
        return "au_hotlines"
    # Generic invitations that don't name a tool — pick a sensible default
    if any(p in low for p in ["tiny step", "small step", "short exercise", "quick exercise", "try something", "try a little", "exercise together"]):
        # If it also mentions grounding, prefer the grounding technique
        if "ground" in low or "grounding" in low:
            return "grounding_54321"
        return "breathing_box"
    return None


def _format_tool_output(name: str, data: dict) -> str:
    title = data.get("title") or _pretty_tool(name).title()
    out = [title]
    steps = data.get("steps")
    fields = data.get("fields")
    duration = data.get("duration_min")
    if duration:
        out.append(f"(~{duration} min)")
    if isinstance(steps, list) and steps:
        out.append("")
        for i, s in enumerate(steps, 1):
            out.append(f"{i}. {s}")
    if isinstance(fields, list) and fields:
        out.append("")
        out.append("You can jot down:")
        for i, f in enumerate(fields, 1):
            out.append(f"{i}. {f}")
    resources = data.get("resources")
    if isinstance(resources, list) and resources:
        out.append("")
        for r in resources:
            n = r.get("name", "Resource")
            p = r.get("phone", "")
            d = r.get("desc", "")
            extra = f" — {d}" if d else ""
            phone = f" ({p})" if p else ""
            out.append(f"• {n}{phone}{extra}")
    out.append("")
    out.append("How are you feeling now — any change, even slightly?")
    return "\n".join(out).strip()


def _perform_tool_message(name: str) -> str:
    try:
        if name == "breathing_box":
            data = wellbeing_tools.breathing_box()
        elif name == "grounding_54321":
            data = wellbeing_tools.grounding_54321()
        elif name == "thought_record_prompt":
            data = wellbeing_tools.thought_record_prompt()
        elif name == "cbt_tips":
            data = wellbeing_tools.cbt_tips()
        elif name == "au_hotlines":
            data = wellbeing_tools.au_hotlines()
        else:
            return "Let's try a short, gentle step together. How are you feeling now — any change, even slightly?"
        return _format_tool_output(name, data)
    except Exception:
        return "Let's try a short, gentle step together. How are you feeling now — any change, even slightly?"


def _tool_alternatives_line() -> str:
    return random.choice([
        "No worries — we can skip it. Would you like to keep talking, or try a different option like grounding or a thought record?",
        "That's okay — we can do something else. Want tips, a grounding exercise, or just to chat a bit more?",
        "Got it. We can choose another small step or simply talk it through — what would help most?",
    ])

def _postprocess_reply_for_positive_user(user_text: str, reply: str) -> str:
    """If the user message is positive, ensure the reply does not suggest exercises.
    If it does, replace with a friendly follow-up template.
    """
    mood = detect_mood_from_text(user_text)
    if mood in ("happy", "grateful", "encouraging"):
        low = (reply or "").lower()
        if any(k in low for k in EXERCISE_KEYWORDS):
            # choose a friendly follow-up
            return "Oh really — what made you feel that way?"
    return reply


# -------- Optional OpenAI chat with tool-calling --------
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
FORCE_LLM_ONLY = str(os.getenv("LLM_ONLY", "")).lower() in ("1", "true", "yes", "on")
client = None
try:
    if USE_LLM:
        from openai import OpenAI
        client = OpenAI()
except Exception:
    USE_LLM = False
    client = None

app = FastAPI(title="Advanced Mental Health Chatbot", version="1.0.0")

# Basic logging so startup clearly reports whether OpenAI is enabled (will not print keys)
logging.basicConfig(level=logging.INFO)

# Ensure database tables exist at import time as well (useful for tests without lifespan)
try:
    init_db()
except Exception:
    pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# expose a simple status endpoint so the frontend can show whether the LLM path is enabled
@app.get("/api/status")
def status():
    """Return whether the app will attempt to use the OpenAI LLM and whether a client was initialized.
    This endpoint intentionally does not return secrets or API keys.
    """
    using_llm = bool(USE_LLM and client)
    return {"using_llm": using_llm, "llm_configured": bool(USE_LLM), "llm_only": FORCE_LLM_ONLY, "message": "LLM enabled" if using_llm else "Using local fallback"}


# -------- Utility endpoints for requirements --------


# -------- Pydantic request models --------
class Turn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatIn(BaseModel):
    session_id: str
    messages: List[Turn]

class EmotionIn(BaseModel):
    text: str

class JournalIn(BaseModel):
    session_id: str
    note: str
    mood: str | None = None

# -------- Auth models --------
class SignUpIn(BaseModel):
    name: str
    email: str
    password: str

class VerifyStatus(BaseModel):
    email: str
    is_verified: bool

# -------- Utility endpoints for requirements --------
@app.post("/api/emotion")
def detect_emotion(payload: EmotionIn):
    mood = detect_mood_from_text(payload.text)
    return {"mood": mood}


@app.get("/api/cbt_tips")
def get_cbt_tips(mood: str | None = None):
    data = wellbeing_tools.cbt_tips()
    if mood:
        m = mood.lower()
        tips = (data.get("by_mood", {}) or {}).get(m)
        if tips:
            return {"title": data.get("title"), "mood": m, "tips": tips}
    return {"title": data.get("title"), "general": data.get("general", []), "by_mood": data.get("by_mood", {})}


# -------- Auth endpoints: signup and email verification --------
@app.post("/api/auth/signup")
def auth_signup(payload: SignUpIn, request: Request, db: Session = Depends(get_session)):
    email = (payload.email or "").strip().lower()
    name = (payload.name or "").strip()
    password = payload.password or ""
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    if not name:
        raise HTTPException(status_code=400, detail="Name required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Existing user check
    existing = db.exec(select(User).where(User.email == email)).first()
    if existing and existing.is_verified:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create or update unverified user
    token = uuid.uuid4().hex
    expires = datetime.utcnow() + timedelta(hours=24)
    pw_hash = _hash_password(password)
    if not existing:
        user = User(email=email, full_name=name, password_hash=pw_hash, is_verified=False,
                    verification_token=token, verification_expires=expires)
        db.add(user)
    else:
        existing.full_name = name or existing.full_name
        existing.password_hash = pw_hash
        existing.is_verified = False
        existing.verification_token = token
        existing.verification_expires = expires
        db.add(existing)
    db.commit()

    base = os.getenv("BACKEND_BASE_URL", f"http://{request.client.host}:{request.url.port or 8000}")
    verify_url = f"{base}/api/auth/verify?token={token}"

    subject = "Verify your email"
    html = (
        f"<p>Hi {name or 'there'},</p>"
        f"<p>Please verify your email for Mental Health Chat by clicking the link below:</p>"
        f"<p><a href='{verify_url}'>Verify my email</a></p>"
        f"<p>This link expires in 24 hours.</p>"
    )
    sent = _send_email(subject, email, html, text_body=f"Open this link to verify: {verify_url}")
    # For development convenience, include the link when SMTP is not configured
    if not sent:
        logging.info("Dev verify link for %s: %s", email, verify_url)
        return {"status": "ok", "message": "Sign-up created. Check your email for verification.", "dev_link": verify_url}
    return {"status": "ok", "message": "Sign-up created. Check your email for verification."}


@app.get("/api/auth/verify")
def auth_verify(token: str = Query(...), db: Session = Depends(get_session)):
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    user = db.exec(select(User).where(User.verification_token == token)).first()
    if not user:
        return HTMLResponse("<h2>Invalid or already used verification link.</h2>")
    if user.verification_expires and datetime.utcnow() > user.verification_expires:
        return HTMLResponse("<h2>Verification link has expired. Please sign up again.</h2>")
    user.is_verified = True
    user.verification_token = None
    user.verification_expires = None
    db.add(user)
    db.commit()

    front = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5500")
    return HTMLResponse(
        f"""
        <html><body style='font-family:system-ui'>
        <h2>Email verified ✅</h2>
        <p>You can now <a href='{front}/login.html'>log in</a>.</p>
        </body></html>
        """
    )


# -------- Email sending helper (SMTP) --------
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def _hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def _verify_password(pw: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(pw, hashed)
    except Exception:
        return False


def _send_email(subject: str, to_email: str, html_body: str, text_body: str | None = None) -> bool:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    use_tls = str(os.getenv("SMTP_TLS", "1")).lower() in ("1", "true", "yes", "on")
    from_email = os.getenv("FROM_EMAIL", user or "no-reply@example.com")

    if not host or not user or not pwd:
        logging.warning("SMTP not configured; would send to %s with subject '%s'", to_email, subject)
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        if text_body:
            msg.set_content(text_body)
            msg.add_alternative(html_body, subtype="html")
        else:
            # minimal plain text fallback as well
            msg.set_content(html_body)

        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            server.login(user, pwd)
            server.send_message(msg)
        return True
    except Exception:
        logging.exception("Failed to send email to %s", to_email)
        return False


@app.post("/api/journal")
def add_journal(entry: JournalIn, db: Session = Depends(get_session)):
    # Ensure session exists
    if not db.exec(select(ChatSession).where(ChatSession.session_id == entry.session_id)).first():
        db.add(ChatSession(session_id=entry.session_id))
        db.commit()
    row = JournalEntry(session_id=entry.session_id, mood=entry.mood, note=entry.note)
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "session_id": row.session_id, "mood": row.mood, "note": row.note, "created_at": row.created_at.isoformat()}


@app.get("/api/journal/{session_id}")
def list_journal(session_id: str, db: Session = Depends(get_session)):
    rows = db.exec(select(JournalEntry).where(JournalEntry.session_id == session_id).order_by(JournalEntry.created_at.desc())).all()
    return [{"id": r.id, "session_id": r.session_id, "mood": r.mood, "note": r.note, "created_at": r.created_at.isoformat()} for r in rows]


@app.post("/api/escalate")
def escalate_to_human(session_id: str, context: str | None = None, db: Session = Depends(get_session)):
    # Record an explicit escalation request and return hotlines/resources.
    db.add(SafetyEvent(session_id=session_id, kind="escalation", payload=context or ""))
    db.commit()
    resources = wellbeing_tools.au_hotlines()
    msg = (
        "If you're in immediate danger, please call emergency services right now. "
        "Here are support options you can contact to speak with a trained counselor."
    )
    return {"message": msg, **resources}


# -------- Startup: init DB --------
@app.on_event("startup")
def _startup():
    init_db()
    # Report whether we will attempt to use the OpenAI client or fall back to local responder.
    if USE_LLM and client:
        logging.info("OpenAI client configured: using LLM for responses.")
    else:
        logging.info("OpenAI key not found or client init failed: using local fallback responder.")


# -------- OpenAI tool specs --------
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "breathing_box",
            "description": "Guide the user through a 2-minute box breathing exercise.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grounding_54321",
            "description": "Guide the 5-4-3-2-1 grounding technique.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thought_record_prompt",
            "description": "Provide a mini CBT thought record template.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "au_hotlines",
            "description": "List Australian crisis hotlines.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cbt_tips",
            "description": "Provide basic CBT tips and strategies.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


# -------- LLM wrapper with tool-calling --------
async def llm_chat(history: List[Dict[str, str]]) -> Dict[str, str]:
    """Call the LLM with safety-focused system prompt and tool-calling.
    Returns a dict: {"content": str, "mood": str}.
    Falls back to a supportive text if no API key is set.
    """
    # Local fallback (no API key): use a small rule-based responder so replies
    # vary and are context-appropriate for simple inputs (greetings, sadness, etc.). Make responses friendlier and more
    # conversational (ask follow-ups, offer options, include light humor when asked) rather than always offering exercises.
    import random

    def local_rule_response(hist: List[Dict[str, str]]) -> Dict[str, str]:
        last_user = None
        for m in reversed(hist):
            if m.get("role") == "user" and m.get("content"):
                last_user = m.get("content").strip()
                break

        if not last_user:
            content = (
                "I hear that this is heavy. Would you like a quick exercise together: "
                "box breathing, a 5-4-3-2-1 grounding, or a mini thought record?"
            )
            return {"content": content, "mood": detect_mood_from_text(content)}

        # Normalize common typos/shorthand for better intent detection
        u = (_simple_normalize_text(last_user) or last_user).lower()
        # Handle common negations and negative qualifiers early (e.g., "not good", "not okay")
        if re.search(r"\b(not\s+(?:good|great|okay|ok|fine|happy|alright|well)|not\s+so\s+good|could\s+be\s+better|worse\b|awful\b|terrible\b|really\s+bad|so\s+bad|bad\b|meh\b|so-?so\b)\b", u):
            content = random.choice([
                "I'm really sorry you're not feeling well. Do you want to share a bit about what's been hardest, or should we try something gentle like a short grounding cue?",
                "That sounds rough — I'm here with you. Would it help to talk it through, or would you rather try a small calming exercise together?",
            ])
            return {"content": content, "mood": "sad"}
        # Positive / feeling-good replies: prefer curiosity and follow-up, not an exercise.
        if re.search(r"\b(feel(?:ing|s)? good|i(?:'|’)m feeling good|i feel good|feeling great|i'm feeling great|i feel great)\b", u):
            content = random.choice([
                "Oh really — what made you feel that way?",
                "That's lovely to hear — what was the highlight of your day?",
                "Nice! Tell me more — what made you feel good?",
            ])
            return {"content": content, "mood": "happy"}

        # greetings (include common abbreviations like hru, sup)
        if re.search(r"\b(hi|hello|hey|yo|howdy)\b", u) or re.search(r"\b(hru|sup|whats? up)\b", u):
            content = random.choice([
                "Hey — nice to meet you. How are you feeling today?",
                "Hi there! What would you like to talk about today?",
                "Hello — I'm here to listen. What's on your mind?",
            ])
            return {"content": content, "mood": detect_mood_from_text(content)}

        # informational: what is anxiety / depression
        if re.search(r"\b(what\s+is|define|meaning\s+of|explain)\s+(anxiety|depression)\b", u):
            topic = 'anxiety' if 'anxiety' in u else 'depression'
            if topic == 'anxiety':
                content = (
                    "Anxiety is a natural stress response — a mix of worry, tension, and body alarm.\n\n"
                    "Quick overview:\n"
                    "• Mind: racing or ‘what if?’ thoughts\n"
                    "• Body: fast heartbeat, tight chest, shaky or sweaty\n"
                    "• Life: hard to sleep, concentrate, or relax\n\n"
                    "Would you like a tiny grounding idea, or to talk about what’s been triggering it lately?"
                )
                return {"content": content, "mood": "anxious"}
            else:
                content = (
                    "Depression is more than sadness — it’s a lasting dip in mood, energy, and interest.\n\n"
                    "Common signs:\n"
                    "• Low mood, emptiness, or tearfulness\n"
                    "• Loss of interest or pleasure\n"
                    "• Sleep/appetite changes, low energy, foggy focus\n\n"
                    "I’m here with you — do you want to share what it’s been like, or try one small step that could help today?"
                )
                return {"content": content, "mood": "sad"}

        # light joke on request (safe, kind humor)
        if re.search(r"\b(joke|make\s+me\s+laugh|tell\s+me\s+a\s+joke|funny)\b", u):
            joke = random.choice([
                "Why did the scarecrow win an award? Because he was outstanding in his field.",
                "I told my anxiety we needed some space. It said, ‘Great, I’ll pack everything we’ve ever worried about.’",
                "Why don’t scientists trust atoms? Because they make up everything.",
                "I asked my calendar if it had time for me — it said, ‘I’m booked.’",
            ])
            return {"content": joke + "  Want another one?", "mood": "encouraging"}

        # thanks
        if re.search(r"\b(thank|thanks|thx)\b", u):
            content = random.choice([
                "You're welcome — I'm glad to help.",
                "No problem — I'm here whenever you need.",
            ])
            return {"content": content, "mood": detect_mood_from_text(content)}

        # feeling sad/depressed — prioritize empathy and a follow-up question; offer exercises only if user seems open.
        if re.search(r"\b(i\s+am\s+|i'm\s+|im\s+)?(sad|depressed|down|unhappy|miserable|awful|terrible|bad)\b", u) or "sad" in u:
            content = random.choice([
                "I’m really sorry you’re going through that. Do you want to tell me what happened, or would you prefer something to help ground you for a minute?",
                "That sounds really heavy — I’m here. If you’d like, you can tell me more, or I can guide a short breathing or grounding exercise. What would be most helpful?",
                "I’m so sorry — that must feel awful. Do you want to talk about it, or should I try to cheer you up with a small distraction or a light joke?",
            ])
            return {"content": content, "mood": "sad"}

        # anxiety/panic — validate and offer a short option set rather than a single prescriptive action
        if re.search(r"\b(anxious|anxiety|panic|nervous)\b", u):
            content = random.choice([
                "I hear that — anxiety can be overwhelming. Would you like a quick breathing cue, a grounding exercise, or some techniques to slow your thoughts?",
                "That sounds scary. I can stay here and listen, or we can try a short grounding exercise together — what would you prefer?",
            ])
            return {"content": content, "mood": "anxious"}

        # help / overwhelmed
        if re.search(r"\b(help|overwhelmed|too much|need help)\b", u):
            content = random.choice([
                "I’m here with you. Would talking through what’s overwhelming help, or would you prefer a short practical step we can try together?",
                "That sounds like a lot to hold. We can break it down together — want to tell me one thing that’s hardest right now?",
            ])
            return {"content": content, "mood": "encouraging"}

        # short conversational replies
        if u in ("hi", "hello", "hey"):
            content = random.choice(["Hey — how's your day going?", "Hi! What's on your mind?"])
            return {"content": content, "mood": detect_mood_from_text(content)}
        if u in ("bye", "goodbye", "see ya"):
            content = random.choice(["Take care — I'm here if you want to talk again.", "Goodbye for now — reach out anytime."])
            return {"content": content, "mood": detect_mood_from_text(content)}

        # fallback: empathic reflection without quoting the user's text; offer choices
        content = random.choice([
            "Thanks for sharing that — I can listen, offer a short exercise, or try to cheer you up. Which would you prefer?",
            "I’m here with you. Would it help to talk it through a bit, or should we try a tiny grounding cue together?",
        ])
        return {"content": content, "mood": detect_mood_from_text(last_user or content)}

    if not client:
        if FORCE_LLM_ONLY:
            # Enforce LLM-only: signal failure so caller can return an error
            raise RuntimeError("LLM_ONLY enabled but OpenAI client is not available")
        return local_rule_response(history)

    # First turn: allow tool suggestions
    try:
        # Log that we're about to call OpenAI (no sensitive values)
        try:
            # derive a short preview of the last user turn from history (safe: no secrets)
            last_user = None
            for m in reversed(history):
                if m.get("role") == "user" and m.get("content"):
                    last_user = m.get("content").strip()
                    break
            last_preview = (last_user[:120] + '...') if last_user and len(last_user) > 120 else (last_user or '<no preview>')
        except Exception:
            last_preview = '<no preview>'
        logging.info("Calling OpenAI for user preview=%s", last_preview)

        # Ask the model to return a strict JSON envelope so we can rely on model-derived mood labels.
        # The model should return ONLY a JSON object like: {"content": "...", "mood": "sad"}
        # Allowed moods: sad, anxious, happy, grateful, encouraging, neutral
        # Conversation style: fuller, friendlier replies similar to ChatGPT, while keeping safety constraints.
        system_msg = (
            "You are a warm, curious, and patient mental-health companion.\n"
            "Style: conversational and natural (like ChatGPT) — write 1–3 short paragraphs and, when helpful, 2–5 clear bullet points.\n"
            "Do NOT quote the user's text back verbatim and do not surround it with quotes. Interpret shorthand like 'hru' as 'how are you'.\n"
            "Use empathy and curiosity first; ask one light follow-up. Provide small, practical help when asked (breathing, grounding, CBT tips), but first ask for consent.\n"
            "If the user asks an informational question (e.g., 'what is anxiety?'), give a concise, accurate explanation with helpful bullets, then one invitational question.\n"
            "Crisis: if the user expresses self-harm or suicide intent, respond supportively and encourage immediate help, referencing local crisis options (do not provide instructions for harm).\n\n"
            "Output policy: OUTPUT ONLY a JSON object with keys: {\"content\": <string>, \"mood\": <one-word-label>} and nothing else.\n"
            "Mood must be one of: sad, anxious, happy, grateful, encouraging, neutral.\n\n"
            "Examples (exact JSON only):\n"
            "{\"content\": \"That makes a lot of sense — anxiety can feel like a fast, worried mind. Would you like a quick grounding idea, or to talk about what's been triggering it lately?\", \"mood\": \"anxious\"}\n"
            "{\"content\": \"I’m really sorry — that sounds painful. Which part feels heaviest right now, or would a short grounding option help?\", \"mood\": \"sad\"}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=650,
            tools=OPENAI_TOOLS,
            messages=[
                {"role": "system", "content": system_msg},
                *history,
            ],
        )
        msg = resp.choices[0].message
        # msg.content should contain the JSON; attempt to parse robustly below
    except Exception as e:
        # If the OpenAI call fails (quota, network, etc.), log and fall back to local responder.
        logging.exception("OpenAI API call failed: %s", e)
        if FORCE_LLM_ONLY:
            raise
        return local_rule_response(history)

    # helper: try to extract JSON object from model output robustly
    def _parse_model_json(s: str) -> Dict[str, str] | None:
        if not s:
            return None
        s = s.strip()
        # attempt direct parse
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # try to find the first {...} in the string
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return None

    # If the model attempted a tool call, DO NOT execute it automatically.
    # Instead, treat it as a proposal and ask the user for consent before running the tool.
    if getattr(msg, "tool_calls", None):
        tc = msg.tool_calls[0]
        fn_name = tc.function.name
        # Compose a proposal asking the user for permission to run the exercise/tool.
        # Use the tool's name in a friendly phrase.
        pretty_name = fn_name.replace('_', ' ')
        proposal = f"I could guide you through a {pretty_name}. Would you like to try that?"
        mood = detect_mood_from_text(proposal)
        # Return proposed_tool so the caller can surface consent UI if desired.
        return {"content": proposal, "mood": mood, "proposed_tool": fn_name}

    content_raw = (msg.content or "").strip()
    parsed = _parse_model_json(content_raw)
    if parsed:
        mood = parsed.get("mood") or detect_mood_from_text(parsed.get("content", ""))
        return {"content": parsed.get("content", ""), "mood": mood}
    # If parsing fails, log and return the raw text + heuristic mood
    logging.warning("LLM output was not valid JSON; attempting repair round to extract JSON")
    # Attempt a repair round: ask the model to re-emit only the JSON object extracted from its previous output.
    try:
        repair_prompt = (
            "The assistant produced an output that did not follow the required JSON-only format. "
            "Please EXTRACT and RETURN ONLY the JSON object matching this schema: {\"content\": <string>, \"mood\": <one-word-label>} "
            "from the text below. If you cannot, return an empty JSON object {}.\n\n" + content_raw
        )
        resp_retry = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": "You are a strict JSON extractor. Output ONLY valid JSON."},
                {"role": "user", "content": repair_prompt},
            ],
        )
        out2 = (resp_retry.choices[0].message.content or "").strip()
        parsed2 = _parse_model_json(out2)
        if parsed2:
            mood = parsed2.get("mood") or detect_mood_from_text(parsed2.get("content", ""))
            return {"content": parsed2.get("content", ""), "mood": mood}
    except Exception:
        logging.exception("Repair round failed")

    logging.warning("Repair failed or returned no JSON; falling back to raw content")
    return {"content": content_raw, "mood": detect_mood_from_text(content_raw)}


# -------- Session & history endpoints --------
@app.post("/api/session")
def create_session(session: str | None = None, db: Session = Depends(get_session)):
    sid = session or str(uuid.uuid4())
    if not db.exec(select(ChatSession).where(ChatSession.session_id == sid)).first():
        db.add(ChatSession(session_id=sid))
        db.commit()
    return {"session_id": sid}


@app.get("/api/messages/{session_id}")
def list_messages(session_id: str, db: Session = Depends(get_session)):
    rows = db.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    ).all()
    return [
        {"role": r.role, "content": r.content, "mood": getattr(r, 'mood', None), "created_at": r.created_at.isoformat()}
        for r in rows
    ]


@app.get("/api/debug/session/{session_id}")
def debug_session(session_id: str, db: Session = Depends(get_session)):
    """Lightweight debug endpoint to verify server-side consent state.
    Returns whether the ChatSession exists plus pending_tool and state snapshot.
    """
    sess = db.exec(select(ChatSession).where(ChatSession.session_id == session_id)).first()
    if not sess:
        return {"exists": False}
    return {
        "exists": True,
        "pending_tool": getattr(sess, "pending_tool", None),
        "state": getattr(sess, "state", None),
    }


# -------- Chat endpoint --------
@app.post("/api/chat")
async def chat(payload: ChatIn, request: Request, db: Session = Depends(get_session)):
    # Basic validation
    if not payload.messages or payload.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    sid = payload.session_id

    # Ensure session exists and load it
    sess = db.exec(select(ChatSession).where(ChatSession.session_id == sid)).first()
    if not sess:
        sess = ChatSession(session_id=sid)
        db.add(sess)
        db.commit()
        db.refresh(sess)

    user_text = payload.messages[-1].content.strip()

    # Ensure DB schema includes mood column (safety for older DBs)
    try:
        # Use exec_driver_sql for raw, dialect-specific SQL (PRAGMA/ALTER) to avoid
        # SQLAlchemy ObjectNotExecutableError when passing plain strings to execute().
        with engine.connect() as conn:
            res = conn.exec_driver_sql("PRAGMA table_info('message')")
            cols = [r[1] for r in res.fetchall()]
            if 'mood' not in cols:
                conn.exec_driver_sql("ALTER TABLE message ADD COLUMN mood TEXT")
    except Exception:
        logging.exception("Failed to ensure mood column exists; continuing")

    # Save user turn
    db.add(Message(session_id=sid, role="user", content=user_text))

    # Moderation + crisis
    mod = check_moderation(user_text)
    if mod.get("flagged"):
        db.add(SafetyEvent(session_id=sid, kind="moderation_flag", payload=mod.get("reason", "")))

    if looks_like_crisis(user_text):
        db.add(SafetyEvent(session_id=sid, kind="crisis", payload=user_text))
        db.add(Message(session_id=sid, role="assistant", content=CRISIS_RESPONSE_AU))
        db.commit()
        return JSONResponse({"reply": CRISIS_RESPONSE_AU, "crisis": True})

    # Normalize user input and handle gibberish / misspellings conservatively
    original_user_text = payload.messages[-1].content.strip()
    # quick heuristics
    if _is_gibberish(original_user_text):
        # Save the raw user turn and ask for clarification
        db.add(Message(session_id=sid, role="user", content=original_user_text))
        clar = "Sorry — I couldn't understand that. Could you rephrase or type it again more clearly?"
        db.add(Message(session_id=sid, role="assistant", content=clar, mood="neutral"))
        db.commit()
        return JSONResponse({"reply": clar, "mood": "neutral", "crisis": False})

    normalized_user_text = _simple_normalize_text(original_user_text)
    # If normalization changed text, use the normalized version for downstream logic
    user_text = normalized_user_text

    # Inspect previous assistant turn (if any)
    prev_assistant_text = None
    # Look back to find the most recent assistant message (more robust than just [-2])
    for m in reversed(payload.messages[:-1]):
        if getattr(m, 'role', '') == 'assistant' and getattr(m, 'content', None):
            prev_assistant_text = m.content
            break

    # If a tool is pending, handle consent now
    if getattr(sess, 'pending_tool', None):
        pending = sess.pending_tool
        if _is_affirmative(user_text):
            reply_text = _perform_tool_message(pending)
            reply_mood = detect_mood_from_text(reply_text)
            # update session state
            try:
                st = json.loads(sess.state) if getattr(sess, 'state', None) else {}
            except Exception:
                st = {}
            st['last_completed_tool'] = pending
            sess.pending_tool = None
            sess.state = json.dumps(st)
            db.add(sess)
            db.add(Message(session_id=sid, role="assistant", content=reply_text, mood=reply_mood))
            db.commit()
            return JSONResponse({"reply": reply_text, "mood": reply_mood, "crisis": False})
        if _is_negative(user_text):
            sess.pending_tool = None
            db.add(sess)
            alt = _tool_alternatives_line()
            db.add(Message(session_id=sid, role="assistant", content=alt, mood="neutral"))
            db.commit()
            return JSONResponse({"reply": alt, "mood": "neutral", "crisis": False})

    # If no pending tool but previous assistant suggested one and user said yes, infer and run
    if prev_assistant_text and _is_affirmative(user_text):
        inferred = _infer_tool_from_text(prev_assistant_text) or "breathing_box"
        reply_text = _perform_tool_message(inferred)
        reply_mood = detect_mood_from_text(reply_text)
        try:
            st = json.loads(sess.state) if getattr(sess, 'state', None) else {}
        except Exception:
            st = {}
        st['last_completed_tool'] = inferred
        sess.pending_tool = None
        sess.state = json.dumps(st)
        db.add(sess)
        db.add(Message(session_id=sid, role="assistant", content=reply_text, mood=reply_mood))
        db.commit()
        return JSONResponse({"reply": reply_text, "mood": reply_mood, "crisis": False})
    if prev_assistant_text and _is_negative(user_text) and _infer_tool_from_text(prev_assistant_text):
        alt = _tool_alternatives_line()
        db.add(Message(session_id=sid, role="assistant", content=alt, mood="neutral"))
        db.commit()
        return JSONResponse({"reply": alt, "mood": "neutral", "crisis": False})

    # Build history for the model (compatible with Pydantic v1/v2)
    # replace the last user message with normalized content for history
    history_in = []
    for m in payload.messages:
        if m.role == 'user' and m is payload.messages[-1]:
            history_in.append({"role": "user", "content": user_text})
        elif m.role in ("user", "assistant"):
            history_in.append({"role": m.role, "content": m.content})
    history = history_in

    # Log incoming request (short preview only, no secrets)
    try:
        preview = (user_text[:120] + '...') if len(user_text) > 120 else user_text
    except Exception:
        preview = '<unable to preview>'
    logging.info("/api/chat called session=%s preview=%s", sid, preview)

    # Generate reply (LLM + tools, or fallback). Expect a dict {content, mood} or with {proposed_tool}.
    try:
        reply_obj = await llm_chat(history)
    except Exception as e:
        logging.exception("llm_chat failed: %s", e)
        # If LLM-only is enforced, return a 503 so the frontend can surface a connection issue (no hardcoded content).
        if FORCE_LLM_ONLY:
            raise HTTPException(status_code=503, detail="LLM unavailable; please check server/API key")
        # Otherwise continue with a generic fallback
        reply_obj = {"content": "Sorry — I ran into a temporary issue. Could we try again?", "mood": "neutral"}
    # If model proposed a tool, persist and ask for consent with varied phrasing
    if isinstance(reply_obj, dict) and reply_obj.get("proposed_tool"):
        proposed = reply_obj.get("proposed_tool")
        sess.pending_tool = proposed
        db.add(sess)
        consent_line = random.choice([
            f"I can guide you through a {_pretty_tool(proposed)} now. Want to try it together?",
            f"Would you like to do a {_pretty_tool(proposed)} with me?",
            f"We could try {_pretty_tool(proposed)} — shall we start?",
        ])
        reply_text = consent_line
        reply_mood = detect_mood_from_text(reply_text)
        db.add(Message(session_id=sid, role="assistant", content=reply_text, mood=reply_mood))
        db.commit()
        return JSONResponse({"reply": reply_text, "mood": reply_mood, "crisis": False})

    # If no explicit tool_call, but the content clearly proposes an exercise, set pending_tool too
    if isinstance(reply_obj, dict):
        maybe = reply_obj.get("content", "")
        inferred_tool = _infer_tool_from_text(maybe)
        # Only treat as a consent proposal if it's phrased as an invitation/question
        if inferred_tool and re.search(r"\b(would you like|shall we|want to|we could try|i can guide you)\b", (maybe or '').lower()):
            sess.pending_tool = inferred_tool
            db.add(sess)
            ask = random.choice([
                f"I can guide you through a {_pretty_tool(inferred_tool)} now. Want to try it together?",
                f"Would you like to do a {_pretty_tool(inferred_tool)} with me?",
                f"We could try {_pretty_tool(inferred_tool)} — shall we start?",
            ])
            reply_text = ask
            reply_mood = detect_mood_from_text(reply_text)
            db.add(Message(session_id=sid, role="assistant", content=reply_text, mood=reply_mood))
            db.commit()
            return JSONResponse({"reply": reply_text, "mood": reply_mood, "crisis": False})

    # Normalize standard reply
    if isinstance(reply_obj, dict):
        reply_text = reply_obj.get("content", "")
        reply_mood = reply_obj.get("mood", "neutral")
    else:
        reply_text = str(reply_obj)
        reply_mood = detect_mood_from_text(reply_text)

    # Post-process: if user was positive, ensure assistant does not suggest exercises
    reply_text = _postprocess_reply_for_positive_user(user_text, reply_text)

    # Save assistant turn (store textual content and mood)
    db.add(Message(session_id=sid, role="assistant", content=reply_text, mood=reply_mood))
    db.commit()

    return JSONResponse({"reply": reply_text, "mood": reply_mood, "crisis": False})


@app.get("/api/logs")
def get_logs(lines: int = 200):
    """Return the last `lines` lines from server.log for quick debugging in the browser.

    This endpoint is intentionally read-only and returns plain text lines. It does not expose
    environment variables or secrets. If the log file is missing, a helpful message is returned.
    """
    try:
        log_path = os.path.join(os.getcwd(), "server.log")
        if not os.path.exists(log_path):
            return {"lines": [], "message": f"Log file not found at {log_path}."}

        # Read the file and return the last N lines. For simplicity we read the whole file; if
        # logs grow large in production, this should be replaced with a streaming/tail implementation.
        with open(log_path, "rb") as f:
            raw = f.read()
        # Try multiple common encodings and pick the one that looks most like text
        encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]
        best = None
        best_score = None
        for enc in encodings:
            try:
                decoded = raw.decode(enc, errors="replace")
            except Exception:
                continue
            # score by number of printable chars minus nulls (higher is better)
            nulls = decoded.count("\x00")
            printable = sum(1 for c in decoded if c.isprintable())
            score = printable - (nulls * 5)
            if best is None or score > best_score:
                best = decoded
                best_score = score

        text = best or raw.decode("latin-1", errors="replace")
        # If decoded text still contains NULs or looks garbled, fall back to extracting printable ASCII
        if "\x00" in text or sum(1 for c in text if ord(c) >= 128) < 5:
            import re
            parts = re.findall(rb"[\x20-\x7E]+", raw)
            try:
                text = b"\n".join(parts).decode("ascii", errors="replace")
            except Exception:
                text = b"\n".join(parts).decode("latin-1", errors="replace")
        all_lines = text.splitlines()
        tail = all_lines[-lines:]
        # Sanitize each line by removing control characters (including NUL) for safe JSON transport
        def sanitize(s: str) -> str:
            return ''.join(ch for ch in s if ord(ch) >= 32)

        tail = [sanitize(l) for l in tail]
        return {"lines": tail}
    except Exception as e:
        logging.exception("Failed to read server.log: %s", e)
        return {"lines": [], "error": str(e)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )