from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ChatSession(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    pending_tool: Optional[str] = None
    state: Optional[str] = None
    # Short emotional summary title (2–4 words) generated after first user message
    title: Optional[str] = Field(default=None, index=True)

class Message(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str
    content: str
    mood: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SafetyEvent(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    kind: str
    payload: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class JournalEntry(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    mood: Optional[str] = None
    note: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class User(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: Optional[str] = None
    password_hash: str
    is_verified: bool = Field(default=False, index=True)
    verification_token: Optional[str] = Field(default=None, index=True)
    verification_expires: Optional[datetime] = None
    # New hashed token field (preferred over plain token); if set, use this for verification lookup
    verification_token_hash: Optional[str] = Field(default=None, index=True)
    last_verification_sent: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- V2: Account-bound Chats & Messages ---
class Chat(SQLModel, table=True):
    __tablename__ = "chats"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="users.id")
    title: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "messages"
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: int = Field(index=True, foreign_key="chats.id")
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
