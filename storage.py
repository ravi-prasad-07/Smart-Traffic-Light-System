from __future__ import annotations

import sqlite3
from pathlib import Path


LANE_NAMES = {
    1: "North Lane",
    2: "East Lane",
    3: "South Lane",
    4: "West Lane",
}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def init_database(db_path: str | Path) -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                lane_count INTEGER NOT NULL,
                priority_lane INTEGER NOT NULL,
                cycle_total INTEGER NOT NULL,
                decision_text TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lane_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                lane_id INTEGER NOT NULL,
                lane_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                vehicle_count INTEGER NOT NULL,
                average_count REAL NOT NULL,
                peak_count INTEGER NOT NULL,
                occupancy_ratio REAL NOT NULL,
                density_level TEXT NOT NULL,
                emergency_detected INTEGER NOT NULL,
                emergency_reason TEXT,
                priority_score REAL NOT NULL,
                signal_order INTEGER NOT NULL,
                green_time INTEGER NOT NULL,
                yellow_time INTEGER NOT NULL,
                sampled_frames INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                video_path TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                pedestrian_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(cycle_id) REFERENCES cycles(id)
            );

            CREATE INDEX IF NOT EXISTS idx_lane_logs_cycle
            ON lane_logs(cycle_id);
            """
        )

        # Migrate existing databases: add pedestrian_count if missing
        try:
            cursor = connection.execute("PRAGMA table_info(lane_logs)")
            columns = {row["name"] for row in cursor.fetchall()}
            if "pedestrian_count" not in columns:
                connection.execute(
                    "ALTER TABLE lane_logs ADD COLUMN pedestrian_count INTEGER NOT NULL DEFAULT 0"
                )
        except Exception:
            pass  # table may not exist yet


def fetch_wait_history(db_path: str | Path) -> dict[int, int]:
    query = """
        SELECT lane_id, signal_order
        FROM lane_logs
        WHERE cycle_id = (
            SELECT id
            FROM cycles
            ORDER BY id DESC
            LIMIT 1
        )
    """
    with _connect(db_path) as connection:
        rows = connection.execute(query).fetchall()

    history = {}
    for row in rows:
        lane_id = int(row["lane_id"])
        order = int(row["signal_order"])
        history[lane_id] = max(0, order - 1)
    return history


def save_cycle(db_path: str | Path, plan: dict) -> int:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO cycles (lane_count, priority_lane, cycle_total, decision_text)
            VALUES (?, ?, ?, ?)
            """,
            (
                len(plan["lanes"]),
                int(plan["priority_lane"]),
                int(plan["cycle_total"]),
                plan["decision_text"],
            ),
        )
        cycle_id = int(cursor.lastrowid)

        rows = [
            (
                cycle_id,
                lane["lane_id"],
                lane["lane_name"],
                lane["filename"],
                lane["vehicle_count"],
                lane["average_count"],
                lane["peak_count"],
                lane["occupancy_ratio"],
                lane["density_level"],
                1 if lane["emergency_detected"] else 0,
                lane["emergency_reason"],
                lane["priority_score"],
                lane["signal_order"],
                lane["green_time"],
                lane["yellow_time"],
                lane["sampled_frames"],
                lane["duration_seconds"],
                lane["video_path"],
                lane["snapshot_path"],
                lane.get("pedestrian_count", 0),
            )
            for lane in plan["lanes"]
        ]

        connection.executemany(
            """
            INSERT INTO lane_logs (
                cycle_id,
                lane_id,
                lane_name,
                filename,
                vehicle_count,
                average_count,
                peak_count,
                occupancy_ratio,
                density_level,
                emergency_detected,
                emergency_reason,
                priority_score,
                signal_order,
                green_time,
                yellow_time,
                sampled_frames,
                duration_seconds,
                video_path,
                snapshot_path,
                pedestrian_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    return cycle_id


def fetch_recent_cycles(db_path: str | Path, limit: int = 6) -> list[dict]:
    query = """
        SELECT
            c.id,
            c.created_at,
            c.lane_count,
            c.priority_lane,
            c.cycle_total,
            c.decision_text,
            SUM(l.vehicle_count) AS total_vehicles,
            MAX(l.emergency_detected) AS has_emergency
        FROM cycles AS c
        JOIN lane_logs AS l
            ON l.cycle_id = c.id
        GROUP BY c.id
        ORDER BY c.id DESC
        LIMIT ?
    """
    with _connect(db_path) as connection:
        rows = connection.execute(query, (limit,)).fetchall()

    history = []
    for row in rows:
        priority_lane = int(row["priority_lane"])
        history.append(
            {
                "id": int(row["id"]),
                "created_at": row["created_at"],
                "lane_count": int(row["lane_count"]),
                "priority_lane": priority_lane,
                "priority_lane_name": LANE_NAMES.get(priority_lane, f"Lane {priority_lane}"),
                "cycle_total": int(row["cycle_total"]),
                "decision_text": row["decision_text"],
                "total_vehicles": int(row["total_vehicles"] or 0),
                "has_emergency": bool(row["has_emergency"]),
            }
        )
    return history
