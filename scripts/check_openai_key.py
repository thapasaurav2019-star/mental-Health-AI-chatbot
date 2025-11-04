#!/usr/bin/env python3
"""
Check whether OPENAI_API_KEY is present in the environment and if it appears valid by calling OpenAI's models endpoint.
This script prints only non-sensitive status lines:
 - MISSING (no env var)
 - VALID (200 OK)
 - INVALID (<status code>)
 - ERROR (<message>)

Run with the project's venv python.
"""
import os
import sys

try:
    import httpx
except Exception:
    print("ERROR: httpx not installed in the venv.")
    sys.exit(2)

KEY = os.getenv("OPENAI_API_KEY")
if not KEY:
    print("MISSING")
    sys.exit(0)

headers = {"Authorization": f"Bearer {KEY}"}
url = "https://api.openai.com/v1/models"

try:
    with httpx.Client(timeout=10.0) as client:
        r = client.get(url, headers=headers)
        if r.status_code == 200:
            print("VALID")
            sys.exit(0)
        elif r.status_code == 401:
            print("INVALID: 401 Unauthorized")
            sys.exit(1)
        else:
            print(f"INVALID: {r.status_code}")
            sys.exit(1)
except Exception as e:
    print("ERROR:", str(e))
    sys.exit(3)
