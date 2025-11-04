#!/usr/bin/env python3
"""
Simple safe migration to add a nullable `mood` TEXT column to the `message` table in chat.db.
Usage: run from the project root where chat.db lives.
It will exit with code 0 on success or if the column already exists; non-zero on error.
"""
import sqlite3
import os
import sys

DB = os.path.abspath("chat.db")

def has_mood_column(conn):
    cur = conn.execute("PRAGMA table_info('message')")
    cols = [r[1] for r in cur.fetchall()]
    return 'mood' in cols


def add_mood_column(conn):
    # SQLite supports adding a single column with ALTER TABLE ... ADD COLUMN
    conn.execute("ALTER TABLE message ADD COLUMN mood TEXT")
    conn.commit()


def verify(conn):
    cur = conn.execute("PRAGMA table_info('message')")
    cols = [r[1] for r in cur.fetchall()]
    return cols


def main():
    if not os.path.exists(DB):
        print(f"ERROR: database not found at {DB}")
        sys.exit(2)

    try:
        conn = sqlite3.connect(DB)
    except Exception as e:
        print("ERROR: could not open database:", e)
        sys.exit(3)

    try:
        if has_mood_column(conn):
            print("SKIP: 'mood' column already exists on table 'message'.")
            cols = verify(conn)
            print("Columns:", cols)
            sys.exit(0)

        print("Adding 'mood' column to 'message' table...")
        add_mood_column(conn)

        cols = verify(conn)
        if 'mood' in cols:
            print("SUCCESS: 'mood' column added. Columns now:", cols)
            sys.exit(0)
        else:
            print("ERROR: Migration ran but 'mood' not found in PRAGMA table_info output.")
            print("Columns after attempt:", cols)
            sys.exit(4)

    except sqlite3.OperationalError as oe:
        print("SQLite OperationalError:", oe)
        print("This may mean the table doesn't exist or the ALTER failed. No changes were committed.")
        sys.exit(5)
    except Exception as e:
        print("ERROR during migration:", e)
        sys.exit(6)
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
