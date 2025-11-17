# app/safety.py
import os
import re
from typing import Dict, Any
import random

# --- Crisis heuristics (always available) ---
CRISIS_PATTERNS = [
    r"\b(suicide|kill myself|end my life|want to die|self[- ]?harm)\b",
    r"\b(harm|hurt) (myself|my self)\b",
    r"\bI (don['’]t|do not) feel safe\b",
    r"\b(plan|method) (to|for) (die|harm)\b",
]

CRISIS_RESPONSE_AU = (
    "I’m really sorry you’re feeling this much pain — I’m here with you. "
    "If you feel you might be in immediate danger or can’t stay safe, please call emergency services now.\n\n"
    "Are you in a place where you feel physically safe right now? "
    "Is there someone you trust who could be with you or check in on you?\n\n"
    "If you’re in Australia, you can contact:\n"
    "• Lifeline: 13 11 14 (24/7)\n"
    "• Suicide Call Back Service: 1300 659 467 (24/7)\n"
    "• Beyond Blue: 1300 22 4636\n\n"
    "If you’re in the United States, you can call or text 988 (Suicide & Crisis Lifeline).\n"
    "If you’re elsewhere, please contact your local emergency number or the nearest crisis hotline.\n\n"
    "You’re not alone. I can stay here with you. If it helps, we can take a slow breath together — in for four, hold for four, out for six."
)

def looks_like_crisis(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in CRISIS_PATTERNS)

# --- Panic/anxiety-with-physical-symptoms heuristics ---
PANIC_PATTERNS = [
    r"\b(panic|panic attack|anxiety attack)\b",
    r"\b(can'?t\s*(breathe|breath)|short\s*of\s*breath|struggling\s*to\s*breathe|hard\s*to\s*breathe)\b",
    r"\b(dizzy|dizziness|lightheaded|faint|fainting)\b",
    r"\b(shak(?:e|ing)|trembl(?:e|ing)|sweat(?:y)?)\b",
    r"\b(heart|heartbeat|pulse)\s*(is\s*)?(racing|fast|pounding|beating\s*so\s*fast)\b",
    r"\b(chest\s*(tight|tightness|pressure))\b",
    r"\b(feel like (i'?m|i am) (going to|gonna) (die|pass out|faint))\b",
    r"\b(scared|terrified|fearful|afraid)\b",
]

def looks_like_panic(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in PANIC_PATTERNS)


def grounding_mode_reply() -> str:
    """Return a short, gentle grounding reply that follows constraints:
    - Validate first, soft tone, short sentences
    - One small grounding cue
    - Safety check
    - No diagnosis or numbered steps
    """
    openings = [
        "That sounds really scary.",
        "I can hear the fear in this.",
        "This feels intense — I'm here.",
        "I'm with you. That’s a lot.",
    ]
    cues = [
        "If you can, feel your feet on the ground.",
        "Place a hand on your chest and notice the movement.",
        "Look around and name one thing you can see.",
        "Let’s take one slow breath together.",
    ]
    safety = "Are you in a safe place right now?"
    closing = [
        "I'm here with you.",
        "We can go one small step at a time.",
        "You don't have to do this alone.",
    ]
    return " ".join([
        random.choice(openings),
        "Let’s slow down together.",
        safety,
        random.choice(cues),
        random.choice(closing),
    ])

# --- Trauma / dissociation heuristics ---
TRAUMA_PATTERNS = [
    r"\b(flashback|flash back)\b",
    r"\b(trigger(?:ed)?|triggering)\b",
    r"\b(dissociat(?:e|ing|ion)|numb|floating|unreal|detached|spaced? out)\b",
    r"\b(memories?\s+coming\s+back)\b",
]

def looks_like_trauma(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in TRAUMA_PATTERNS)


def trauma_mode_reply() -> str:
    openings = [
        "I’m so sorry this is coming up.",
        "That sounds really hard — I’m here.",
        "This is a lot. You’re not alone.",
    ]
    cues = [
        "If it helps, place a hand on your chest and notice the rise and fall.",
        "See if you can feel your feet on the ground — just notice the contact.",
        "Look around and name one thing you can see, quietly to yourself.",
    ]
    safety = "Are you in a safe place right now?"
    note = "We don’t need to go into details."
    return " ".join([random.choice(openings), note, safety, random.choice(cues), "I’ll stay with you."])

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