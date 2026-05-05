import json
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text
from core.interfaces import Repository, Logger


class SQLAlchemyRepository(Repository):
    def __init__(self, postgres_uri: str, tables: Any, columns: Any, logger: Logger) -> None:
        self.engine = create_engine(postgres_uri)
        self.tables = tables
        self.columns = columns
        self.logger = logger

    def get_threshold(self) -> int:
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT threshold_value FROM thresholds LIMIT 1")).fetchone()
            return int(row[0]) if row and row[0] is not None else 50

    def get_device_schedule(self, device_id: int, weekday: str) -> Tuple[Optional[int], Optional[int], Optional[Any], Optional[Any]]:
        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"""
                    SELECT {self.columns.device_data_est_opening}, {self.columns.device_data_est_closing}
                    FROM {self.tables.device_data_table}
                    WHERE {self.columns.device_data_device_id} = :id AND {self.columns.device_data_day} = :day
                """),
                {"id": str(device_id), "day": weekday}
            ).fetchone()
        if result:
            open_time, close_time = result
            if open_time and close_time:
                return open_time.hour, close_time.hour, open_time, close_time
        return None, None, None, None

    def get_recent_txns(self, device_id: int) -> List[float]:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(f'SELECT "{self.columns.z_score_recent_txns}" FROM "{self.tables.device_z_score_table}" WHERE "{self.columns.z_score_device_id}" = :id'),
                {"id": str(device_id)}
            ).fetchone()
        if row is None or row[0] is None:
            return []
        data = row[0]
        if isinstance(data, str):
            try:
                return list(json.loads(data))
            except Exception:
                return []
        if isinstance(data, list):
            return data
        return list(data)

    def upsert_device_stats(self, device_id: int, mean_val: float, std_val: float, z_score: float, recent_txns_json: str) -> None:
        # Cap z-score to prevent database overflow (max value for precision 8, scale 2 is ~999,999.99)
        capped_z_score = max(min(z_score, 999999.99), -999999.99)
        
        with self.engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO "{self.tables.device_z_score_table}" ("{self.columns.z_score_device_id}", "{self.columns.z_score_mean}", "{self.columns.z_score_std}", "{self.columns.z_score}", "{self.columns.z_score_recent_txns}")
                    VALUES (:id, :mean, :std, :z, :txns)
                    ON CONFLICT ("{self.columns.z_score_device_id}")
                    DO UPDATE SET
                        "{self.columns.z_score_mean}" = :mean,
                        "{self.columns.z_score_std}" = :std,
                        "{self.columns.z_score}" = :z,
                        "{self.columns.z_score_recent_txns}" = :txns
                """),
                {
                    "id": str(device_id),
                    "mean": round(mean_val, 2),
                    "std": round(std_val, 2),
                    "z": round(capped_z_score, 2),
                    "txns": recent_txns_json,
                }
            )

    def insert_anomaly_log(self, payload: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {self.tables.anomaly_results_table} (
                        {self.columns.anomaly_device_id}, {self.columns.anomaly_txn_id}, {self.columns.anomaly_ts},
                        {self.columns.anomaly_amt}, {self.columns.anomaly_z}, {self.columns.anomaly_conf},
                        {self.columns.anomaly_label}, {self.columns.anomaly_open}, {self.columns.anomaly_close}
                    )
                    VALUES (:id, :tid, :ts, :amt, :z, :conf, :label, :open, :close)
                """),
                payload,
            )
