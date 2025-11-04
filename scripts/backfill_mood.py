#!/usr/bin/env python3
"""
Backfill script: scan assistant messages in chat.db and populate the `mood` column where empty.
If `pyspellchecker` is available, try to correct obvious typos before mood detection.
"""
import os
import sys
from sqlmodel import Session, select
from app.db import engine
from app.models import Message

SPELL = None
try:
    from spellchecker import SpellChecker
    SPELL = SpellChecker()
except Exception:
    SPELL = None


def correct_text(t: str) -> str:
    if not SPELL:
        return t
    words = t.split()
    corrected = []
    for w in words:
        if not w.isalpha() or len(w) <= 2:
            corrected.append(w)
            continue
        cand = SPELL.correction(w)
        corrected.append(cand or w)
    return ' '.join(corrected)


def main():
    db = Session(engine)
    stmt = select(Message).where(Message.role == 'assistant')
    rows = db.exec(stmt).all()
    updated = 0
    for r in rows:
        if getattr(r, 'mood', None):
            continue
        content = (r.content or '').strip()
        if not content:
            continue
        corrected = correct_text(content)
        # import local detect_mood function
        from app.main import detect_mood_from_text
        mood = detect_mood_from_text(corrected)
        r.mood = mood
        db.add(r)
        updated += 1
    db.commit()
    print(f"Updated {updated} assistant messages with mood.")


if __name__ == '__main__':
    main()
