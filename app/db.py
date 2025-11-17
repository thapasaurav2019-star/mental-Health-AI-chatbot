from __future__ import annotations
from typing import Generator
from sqlmodel import SQLModel, create_engine, Session
import os

DB_PATH = os.getenv("CHATBOT_DB", "chat.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})

def init_db():
    # Ensure FK enforcement on SQLite
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    SQLModel.metadata.create_all(engine)
    # Ensure schema has the `mood` column on Message for older databases.
    # SQLite doesn't automatically add columns via SQLModel.create_all for existing tables,
    # so we run a safe ALTER TABLE if the column is missing.
    try:
        # Use exec_driver_sql for raw dialect-specific SQL (PRAGMA/ALTER)
        with engine.connect() as conn:
            res = conn.exec_driver_sql("PRAGMA table_info('message')")
            cols = [r[1] for r in res.fetchall()]
            if 'mood' not in cols:
                conn.exec_driver_sql("ALTER TABLE message ADD COLUMN mood TEXT")
            # Ensure ChatSession has pending_tool and state columns for older DBs
            res2 = conn.exec_driver_sql("PRAGMA table_info('chatsession')")
            c2 = [r[1] for r in res2.fetchall()]
            if 'pending_tool' not in c2:
                conn.exec_driver_sql("ALTER TABLE chatsession ADD COLUMN pending_tool TEXT")
            if 'state' not in c2:
                conn.exec_driver_sql("ALTER TABLE chatsession ADD COLUMN state TEXT")
            if 'title' not in c2:
                conn.exec_driver_sql("ALTER TABLE chatsession ADD COLUMN title TEXT")
            # Ensure User table has new columns if added later (best-effort idempotent additions)
            try:
                res3 = conn.exec_driver_sql("PRAGMA table_info('users')")
                ucols = [r[1] for r in res3.fetchall()]
                # If table exists but missing newer columns, add them
                if ucols and 'is_verified' in ucols and 'verification_token' not in ucols:
                    conn.exec_driver_sql("ALTER TABLE users ADD COLUMN verification_token TEXT")
                if ucols and 'is_verified' in ucols and 'verification_expires' not in ucols:
                    conn.exec_driver_sql("ALTER TABLE users ADD COLUMN verification_expires DATETIME")
                if ucols and 'verification_token_hash' not in ucols:
                    conn.exec_driver_sql("ALTER TABLE users ADD COLUMN verification_token_hash TEXT")
                if ucols and 'last_verification_sent' not in ucols:
                    conn.exec_driver_sql("ALTER TABLE users ADD COLUMN last_verification_sent DATETIME")
            except Exception:
                pass
    except Exception:
        # If anything goes wrong here, ignore — the app will still run but moods won't persist until manual migration.
        pass

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
