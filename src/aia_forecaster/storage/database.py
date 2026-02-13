"""SQLite persistence for forecast runs and results."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from aia_forecaster.config import settings
from aia_forecaster.models import ForecastRun


class ForecastDatabase:
    """SQLite database for storing forecast pipeline outputs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    id TEXT PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    strike REAL,
                    tenor TEXT,
                    raw_probability REAL,
                    calibrated_probability REAL,
                    num_agents INTEGER,
                    mean_probability REAL,
                    supervisor_confidence TEXT,
                    supervisor_probability REAL,
                    data_json TEXT,
                    started_at TEXT,
                    completed_at TEXT
                )
            """)

    def save_run(self, run: ForecastRun) -> str:
        """Save a forecast run and return its ID."""
        if not run.id:
            run.id = uuid.uuid4().hex[:12]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_runs
                (id, question_text, pair, strike, tenor,
                 raw_probability, calibrated_probability,
                 num_agents, mean_probability,
                 supervisor_confidence, supervisor_probability,
                 data_json, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.question.text,
                    run.question.pair,
                    run.question.strike,
                    run.question.tenor.value if run.question.tenor else None,
                    run.calibrated.raw_probability if run.calibrated else None,
                    run.calibrated.calibrated_probability if run.calibrated else None,
                    len(run.ensemble.agent_forecasts) if run.ensemble else None,
                    run.ensemble.mean_probability if run.ensemble else None,
                    run.ensemble.supervisor.confidence.value if run.ensemble and run.ensemble.supervisor else None,
                    run.ensemble.supervisor.reconciled_probability if run.ensemble and run.ensemble.supervisor else None,
                    run.model_dump_json(),
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                ),
            )
        return run.id

    def get_run(self, run_id: str) -> ForecastRun | None:
        """Retrieve a forecast run by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data_json FROM forecast_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ForecastRun.model_validate_json(row[0])

    def list_runs(self, limit: int = 20) -> list[dict]:
        """List recent forecast runs (summary only)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, question_text, pair, strike, tenor,
                       raw_probability, calibrated_probability,
                       num_agents, started_at
                FROM forecast_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
