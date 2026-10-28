# ============================================================
# Multi-Vehicle Speed Detection System — Database & Analytics
# ============================================================
# Persistent SQLite Audit Database, Multi-Format Event Logging,
# and Comprehensive Matplotlib Session Analytics Generator.
# ============================================================

import os
import sqlite3
import datetime
import csv
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ViolationDatabase:
    """
    Decoupled state persistence and audit storage engine.
    Stores immutable violation records and vehicle passage events
    into SQLite with WAL mode for high concurrency.
    """

    def __init__(self, db_path="logs/violations.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Violations Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_id INTEGER NOT NULL,
                vehicle_class TEXT NOT NULL,
                speed_kmh REAL NOT NULL,
                speed_limit REAL NOT NULL,
                excess_kmh REAL NOT NULL,
                plate_number TEXT NOT NULL,
                plate_confidence REAL DEFAULT 0.0,
                evidence_path TEXT NOT NULL,
                lane TEXT DEFAULT 'Lane 1',
                camera_id TEXT DEFAULT 'CAM-01'
            );
            """)

            # 2. Complete Passage Stream Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_id INTEGER NOT NULL,
                vehicle_class TEXT NOT NULL,
                speed_kmh REAL NOT NULL,
                is_violation INTEGER NOT NULL,
                lane TEXT DEFAULT 'Lane 1'
            );
            """)

            # 3. Session Metadata Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_vehicles INTEGER DEFAULT 0,
                total_violations INTEGER DEFAULT 0,
                avg_speed REAL DEFAULT 0.0,
                max_speed REAL DEFAULT 0.0,
                tracker_type TEXT DEFAULT 'bytetrack'
            );
            """)
            conn.commit()

    def log_violation(self, record):
        """
        Record an overspeeding violation event.
        
        Args:
            record (dict): Keys: timestamp, vehicle_id, vehicle_class, speed_kmh,
                                 speed_limit, plate_number, plate_confidence,
                                 evidence_path, lane, camera_id
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            excess = max(0.0, record["speed_kmh"] - record["speed_limit"])
            cursor.execute("""
            INSERT INTO violations (
                timestamp, vehicle_id, vehicle_class, speed_kmh, speed_limit,
                excess_kmh, plate_number, plate_confidence, evidence_path, lane, camera_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                record["vehicle_id"],
                record["vehicle_class"],
                float(record["speed_kmh"]),
                float(record["speed_limit"]),
                float(excess),
                record.get("plate_number", "N/A"),
                float(record.get("plate_confidence", 0.0)),
                record.get("evidence_path", ""),
                record.get("lane", "Lane 1"),
                record.get("camera_id", "CAM-01")
            ))
            conn.commit()

    def log_passage(self, timestamp, vehicle_id, vehicle_class, speed_kmh, speed_limit, lane="Lane 1"):
        """Record any vehicle passage event."""
        is_violation = 1 if speed_kmh > speed_limit else 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO vehicle_passages (timestamp, vehicle_id, vehicle_class, speed_kmh, is_violation, lane)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, vehicle_id, vehicle_class, float(speed_kmh), is_violation, lane))
            conn.commit()

    def export_csv(self, csv_filepath):
        """Export all vehicle passage logs to CSV."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT p.timestamp, p.vehicle_id, p.vehicle_class, p.speed_kmh,
                   CASE WHEN p.is_violation = 1 THEN 'Yes' ELSE 'No' END as over_limit,
                   COALESCE(v.plate_number, 'N/A') as plate_number,
                   p.lane
            FROM vehicle_passages p
            LEFT JOIN violations v ON p.vehicle_id = v.vehicle_id
            ORDER BY p.id ASC
            """)
            rows = cursor.fetchall()

        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        with open(csv_filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Vehicle_ID", "Vehicle_Type", "Speed_KMH", "Over_Limit", "Plate_Number", "Lane"])
            writer.writerows(rows)

    def export_json_summary(self, json_filepath, session_stats):
        """Export comprehensive structured JSON summary for external APIs/dashboards."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM violations ORDER BY id DESC LIMIT 50")
            violations = [dict(row) for row in cursor.fetchall()]

        payload = {
            "session_summary": session_stats,
            "recent_violations": violations,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
        with open(json_filepath, 'w') as f:
            json.dump(payload, f, indent=2)


# ============================================================
# COMPREHENSIVE MATPLOTLIB ANALYTICS GENERATOR
# ============================================================

def generate_session_analytics(speeds, class_counts, violations_data, speed_limit, output_path="output/session_analytics.png"):
    """
    Generate a 4-panel professional traffic analytics report image using Matplotlib.
    """
    if not HAS_MATPLOTLIB or not speeds:
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("CCTV TRAFFIC MONITORING & SPEED VIOLATION ANALYTICS", fontsize=16, fontweight='bold', y=0.98)

        # Panel 1: Speed Distribution Histogram
        ax1 = axes[0, 0]
        ax1.hist(speeds, bins=12, color='#3498db', edgecolor='black', alpha=0.75, label='Measured Speeds')
        ax1.axvline(speed_limit, color='#e74c3c', linestyle='dashed', linewidth=2.5, label=f'Speed Limit ({speed_limit:.0f} km/h)')
        mean_spd = np.mean(speeds)
        ax1.axvline(mean_spd, color='#2ecc71', linestyle='dotted', linewidth=2, label=f'Mean Speed ({mean_spd:.1f} km/h)')
        ax1.set_title("Speed Distribution & Limit Threshold", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Speed (km/h)", fontsize=10)
        ax1.set_ylabel("Vehicle Count", fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend(loc='upper right')

        # Panel 2: Vehicle Classification Breakdown
        ax2 = axes[0, 1]
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#2980b9'][:len(classes)]
        bars = ax2.bar(classes, counts, color=colors, edgecolor='black', alpha=0.85)
        ax2.set_title("Vehicle Classification Breakdown", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Vehicle Class", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.1, f'{int(h)}', ha='center', va='bottom', fontsize=9)

        # Panel 3: Speeding vs Compliance Donut Chart
        ax3 = axes[1, 0]
        overspeed_count = sum(1 for s in speeds if s > speed_limit)
        compliant_count = len(speeds) - overspeed_count
        pie_labels = [f'Compliant ({compliant_count})', f'Violations ({overspeed_count})']
        pie_colors = ['#2ecc71', '#e74c3c']
        ax3.pie([compliant_count, overspeed_count], labels=pie_labels, colors=pie_colors,
                autopct='%1.1f%%', startangle=140, explode=(0, 0.1),
                wedgeprops=dict(width=0.45, edgecolor='black'))
        ax3.set_title("Speed Compliance Rate", fontsize=12, fontweight='bold')

        # Panel 4: Speeds Timeline Scatter
        ax4 = axes[1, 1]
        x_indices = list(range(1, len(speeds) + 1))
        point_colors = ['#e74c3c' if s > speed_limit else '#27ae60' for s in speeds]
        ax4.scatter(x_indices, speeds, c=point_colors, s=50, edgecolors='black', alpha=0.8)
        ax4.axhline(speed_limit, color='#e74c3c', linestyle='dashed', linewidth=1.5)
        ax4.set_title("Sequential Passage Speeds (Over Time)", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Sequential Vehicle Index", fontsize=10)
        ax4.set_ylabel("Speed (km/h)", fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"📊 Multi-panel session analytics saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to generate session analytics: {e}")

