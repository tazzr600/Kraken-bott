import os
import sqlite3

DB_PATH = os.getenv("DB_PATH", "kraken_bot.db")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS risk_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_risk(key, default=None):
    init_db()

    conn = db()
    row = conn.execute(
        "SELECT value FROM risk_state WHERE key=?",
        (key,)
    ).fetchone()
    conn.close()

    if not row:
        return default

    return row["value"]


def set_risk(key, value):
    init_db()

    conn = db()

    conn.execute("""
        INSERT INTO risk_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key)
        DO UPDATE SET value=excluded.value
    """, (key, str(value)))

    conn.commit()
    conn.close()


# Make absolutely sure the database exists
# before the bot is imported/started.
init_db()
