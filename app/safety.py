# app/safety.py
import os
import re
from typing import Dict, Any

# --- Crisis heuristics (always available) ---
CRISIS_PATTERNS = [
    r"\b(suicide|kill myself|end my life|want to die|self[- ]?harm)\b",
    r"\b(harm|hurt) (myself|my self)\b",
    r"\bI (don['’]t|do not) feel safe\b",
    r"\b(plan|method) (to|for) (die|harm)\b",
]

CRISIS_RESPONSE_AU = (
    "I’m really glad you reached out. If you are in immediate danger, call **000** right now.\n\n"
    "24/7 support in Australia: **Lifeline (13 11 14)**, **Beyond Blue (1300 22 4636)**, "
    "**Suicide Call Back Service (1300 659 467)**. If you’d like, I can stay with you and "
    "we can take a small step together."
)

def looks_like_crisis(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in CRISIS_PATTERNS)

# --- Optional OpenAI moderation (used if OPENAI_API_KEY is set) ---
_USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
_client = None
if _USE_OPENAI:
    try:
        from openai import OpenAI  # pip install openai>=1
        _client = OpenAI()
    except Exception:
        _client = None
        _USE_OPENAI = False

def check_moderation(text: str) -> Dict[str, Any]:
    """
    Returns: {"flagged": bool, "reason": str}
    Tries OpenAI moderation when available; otherwise falls back to heuristic.
    """
    # Try OpenAI moderation
    if _client:
        try:
            resp = _client.moderations.create(
                model="omni-moderation-latest",
                input=text
            )
            flagged = bool(resp.results[0].flagged)  # type: ignore[attr-defined]
            return {"flagged": flagged, "reason": "openai-moderation" if flagged else "none"}
        except Exception:
            # fall through to heuristic
            pass

    # Heuristic fallback (very conservative)
    if looks_like_crisis(text):
        return {"flagged": True, "reason": "crisis-heuristic"}

    return {"flagged": False, "reason": "none"}