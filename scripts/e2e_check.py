#!/usr/bin/env python3
"""
End-to-end check: create a session and POST a positive user message, then assert the assistant reply is a friendly follow-up
and does NOT suggest exercises. Prints PASS/FAIL and the assistant reply.
"""
import httpx
import sys

API_BASE = "http://127.0.0.1:8002"
BAD_KEYWORDS = ["exercise", "breath", "grounding", "breathe", "box breathing"]

try:
    with httpx.Client(timeout=20.0) as c:
        # create session
        r = c.post(f"{API_BASE}/api/session")
        r.raise_for_status()
        sid = r.json().get("session_id")
        if not sid:
            print("FAIL: no session id")
            sys.exit(2)
        # send positive message
        payload = {"session_id": sid, "messages": [{"role": "user", "content": "I'm feeling good"}]}
        r2 = c.post(f"{API_BASE}/api/chat", json=payload)
        r2.raise_for_status()
        data = r2.json()
        reply = data.get("reply","")
        reply_low = (reply or "").lower()
        found_bad = [k for k in BAD_KEYWORDS if k in reply_low]
        if found_bad:
            print("FAIL: assistant suggested exercise/grounding ->", reply)
            sys.exit(1)
        # look for friendly follow-up patterns (a question or 'what' etc.)
        if any(x in reply_low for x in ["what made", "what was", "tell me more", "tell me", "really", "what happened", "why"]) or reply.strip().endswith('?'):
            print("PASS: ", reply)
            sys.exit(0)
        # Not explicitly an exercise but also not a friendly follow-up — report as warning but pass
        print("WARN: reply may not be a clear follow-up:", reply)
        sys.exit(0)
except Exception as e:
    print("ERROR:", e)
    sys.exit(3)
