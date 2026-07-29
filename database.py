"""
Database functions for the Counterfeit Medicine Risk Assessment System.
"""

import sqlite3

from config import DB_PATH


def get_connection():
    """Create a database connection."""

    return sqlite3.connect(DB_PATH)


def create_table():
    """Create the prediction history table."""

    query = """
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        medicine_name TEXT,
        manufacturer TEXT,
        risk TEXT,
        confidence REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """

    with get_connection() as connection:
        connection.execute(query)
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(prediction_history)")
        columns = [column[1] for column in cursor.fetchall()]
        if "created_at" not in columns:
            cursor.execute("ALTER TABLE prediction_history ADD COLUMN created_at DATETIME")


def save_prediction(medicine_name, manufacturer, risk, confidence):
    """Save prediction details."""

    query = """
    INSERT INTO prediction_history
    (medicine_name, manufacturer, risk, confidence)
    VALUES (?, ?, ?, ?)
    """

    with get_connection() as connection:
        connection.execute(
            query,
            (medicine_name, manufacturer, risk, confidence)
        )


def get_predictions(limit=50):
    """Return recent predictions."""

    query = """
    SELECT
        medicine_name,
        manufacturer,
        risk,
        confidence,
        created_at
    FROM prediction_history
    ORDER BY created_at DESC
    LIMIT ?
    """

    with get_connection() as connection:
        rows = connection.execute(query, (limit,)).fetchall()

    return rows


def search_prediction(medicine_name):
    """Search predictions by medicine name."""

    query = """
    SELECT
        medicine_name,
        manufacturer,
        risk,
        confidence,
        created_at
    FROM prediction_history
    WHERE medicine_name LIKE ?
    ORDER BY created_at DESC
    """

    with get_connection() as connection:
        rows = connection.execute(
            query,
            (f"%{medicine_name}%",)
        ).fetchall()

    return rows