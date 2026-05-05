import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    postgres_uri: str


@dataclass
class TableConfig:
    device_data_table: str
    device_z_score_table: str
    anomaly_results_table: str


@dataclass
class ColumnConfig:
    device_data_device_id: str
    device_data_day: str
    device_data_est_opening: str
    device_data_est_closing: str

    z_score_device_id: str
    z_score_mean: str
    z_score_std: str
    z_score: str
    z_score_recent_txns: str

    anomaly_device_id: str
    anomaly_txn_id: str
    anomaly_ts: str
    anomaly_amt: str
    anomaly_z: str
    anomaly_conf: str
    anomaly_label: str
    anomaly_open: str
    anomaly_close: str


@dataclass
class ProcessingConfig:
    rolling_window: int
    log_file_path: str


class ConfigManager:
    def __init__(self) -> None:
        pass

    def database(self) -> DatabaseConfig:
        return DatabaseConfig(postgres_uri=os.getenv("POSTGRES_URI"))

    def tables(self) -> TableConfig:
        return TableConfig(
            device_data_table=os.getenv("DEVICE_DATA_TABLE"),
            device_z_score_table=os.getenv("DEVICE_Z_SCORE_TABLE"),
            anomaly_results_table=os.getenv("ANOMALY_RESULTS_TABLE"),
        )

    def columns(self) -> ColumnConfig:
        return ColumnConfig(
            device_data_device_id=os.getenv("DEVICE_DATA_DEVICE_ID"),
            device_data_day=os.getenv("DEVICE_DATA_DAY"),
            device_data_est_opening=os.getenv("DEVICE_DATA_EST_OPENING"),
            device_data_est_closing=os.getenv("DEVICE_DATA_EST_CLOSING"),
            z_score_device_id=os.getenv("Z_SCORE_DEVICE_ID"),
            z_score_mean=os.getenv("Z_SCORE_MEAN"),
            z_score_std=os.getenv("Z_SCORE_STD"),
            z_score=os.getenv("Z_SCORE"),
            z_score_recent_txns=os.getenv("Z_SCORE_RECENT_TXNS"),
            anomaly_device_id=os.getenv("ANOMALY_DEVICE_ID"),
            anomaly_txn_id=os.getenv("ANOMALY_TXN_ID"),
            anomaly_ts=os.getenv("ANOMALY_TS"),
            anomaly_amt=os.getenv("ANOMALY_AMT"),
            anomaly_z=os.getenv("ANOMALY_Z"),
            anomaly_conf=os.getenv("ANOMALY_CONF"),
            anomaly_label=os.getenv("ANOMALY_LABEL"),
            anomaly_open=os.getenv("ANOMALY_OPEN"),
            anomaly_close=os.getenv("ANOMALY_CLOSE"),
        )

    def processing(self) -> ProcessingConfig:
        return ProcessingConfig(
            rolling_window=int(os.getenv("ROLLING_WINDOW", 100)),
            log_file_path=os.getenv("LOG_FILE_PATH", "etl_device.log"),
        )
