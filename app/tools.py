from typing import Dict, Any

AU_RESOURCES = [
    {"name": "Lifeline", "phone": "13 11 14", "desc": "24/7 crisis support"},
    {"name": "Beyond Blue", "phone": "1300 22 4636", "desc": "Anxiety & depression support"},
    {"name": "Suicide Call Back Service", "phone": "1300 659 467", "desc": "24/7 telephone and online"},
]

def breathing_box() -> Dict[str, Any]:
    return {"title": "Box Breathing", "steps": ["Inhale 4s", "Hold 4s", "Exhale 4s", "Hold 4s"], "duration_min": 2}

def grounding_54321() -> Dict[str, Any]:
    return {"title": "5-4-3-2-1 Grounding", "steps": [
        "Name 5 things you can see",
        "Name 4 things you can touch",
        "Name 3 things you can hear",
        "Name 2 things you can smell",
        "Name 1 thing you can taste",
    ]}

def thought_record_prompt() -> Dict[str, Any]:
    return {"title": "Thought record (mini)", "fields": [
        "Situation", "Emotion (0-100%)", "Automatic thought",
        "Evidence for", "Evidence against", "Balanced thought"
    ]}

def au_hotlines() -> Dict[str, Any]:
    return {"title": "Support in Australia", "resources": AU_RESOURCES}

def cbt_tips() -> Dict[str, Any]:
    """Provide basic CBT tips: general strategies and mood-specific suggestions.
    This is intentionally simple and non-clinical.
    """
    return {
        "title": "CBT tips (basic)",
        "general": [
            "Name the thought and rate how much you believe it (0–100%).",
            "Look for evidence for and against the thought.",
            "Try a balanced alternative thought.",
            "Do a tiny behavioral experiment to test the belief.",
            "Use present-focused grounding before thinking work if very activated.",
        ],
        "by_mood": {
            "anxious": [
                "Differentiate possibility vs probability: what’s the realistic likelihood?",
                "Shrink the problem: one small step you can do in 5 minutes.",
                "Postpone worry: schedule a 10‑minute worry slot later.",
            ],
            "sad": [
                "Behavioral activation: plan one rewarding or meaningful activity today.",
                "Challenge all‑or‑nothing thoughts: look for grey areas.",
                "Connect with someone for 5–10 minutes, even by text.",
            ],
            "neutral": [
                "Keep a brief thought record when strong emotions show up.",
                "Practice labeling feelings + needs in 1–2 sentences.",
            ],
        },
    }
