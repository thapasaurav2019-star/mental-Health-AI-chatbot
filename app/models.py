from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ChatSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Track a proposed tool awaiting user consent and lightweight session state
    pending_tool: Optional[str] = None
    state: Optional[str] = None

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str
    content: str
    # Optional mood label: sad, anxious, happy, grateful, encouraging, neutral
    mood: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SafetyEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    kind: str
    payload: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class JournalEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    mood: Optional[str] = None
    note: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
