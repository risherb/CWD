# ======================== dynamic code =========================================
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import pytz
from sqlalchemy import create_engine
from sqlalchemy.types import BigInteger, Float, String, TIME, JSON
import os
from dotenv import load_dotenv
import json
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import insert
from pathlib import Path
import logging

# =================== LOGGING SETUP ===================
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

LOG_DIR = "Init_LOGS"
os.makedirs(LOG_DIR, exist_ok=True)
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

logger = logging.getLogger("baseline_builder")
logger.setLevel(logging.DEBUG)

dev_handler = logging.FileHandler(f"{LOG_DIR}/init_developer.log")
dev_handler.setLevel(logging.DEBUG)

client_handler = logging.FileHandler(f"{LOG_DIR}/init_client.log")
client_handler.setLevel(logging.INFO)

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

dev_handler.setFormatter(log_formatter)
client_handler.setFormatter(log_formatter)

logger.addHandler(dev_handler)
logger.addHandler(client_handler)
if not ENABLE_LOGGING:
    logger.disabled = True



# =================== LOAD ENV ===================
logger.debug(
    f"ENV loaded | MongoHost={os.getenv('MONGO_HOST')} DB={os.getenv('MONGO_DBNAME')}"
)

POSTGRES_URI = os.getenv("POSTGRES_URI_2")

# Tables
DEVICE_DATA_TABLE = os.getenv("DEVICE_DATA_TABLE")
DEVICE_Z_SCORE_TABLE = os.getenv("DEVICE_Z_SCORE_TABLE")

# Columns for device_data
DEVICE_DATA_DEVICE_ID = os.getenv("DEVICE_DATA_DEVICE_ID")
DEVICE_DATA_DAY = os.getenv("DEVICE_DATA_DAY")
DEVICE_DATA_EST_OPENING = os.getenv("DEVICE_DATA_EST_OPENING")
DEVICE_DATA_EST_CLOSING = os.getenv("DEVICE_DATA_EST_CLOSING")

# Columns for device_z_score
Z_SCORE_DEVICE_ID = os.getenv("Z_SCORE_DEVICE_ID")
Z_SCORE_MEAN = os.getenv("Z_SCORE_MEAN")
Z_SCORE_STD = os.getenv("Z_SCORE_STD")
Z_SCORE = os.getenv("Z_SCORE")
Z_SCORE_RECENT_TXNS = os.getenv("Z_SCORE_RECENT_TXNS")

# =================== CONFIG ===================
MONGO_CONFIG = {
    "host": os.getenv("MONGO_HOST"),
    "dbname": os.getenv("MONGO_DBNAME"),
    "user": os.getenv("MONGO_USER"),
    "password": os.getenv("MONGO_PASSWORD"),
    "port": int(os.getenv("MONGO_PORT", 27017)),
}
MONGO_AUTHSOURCE = os.getenv("MONGO_AUTHSOURCE", "admin")

ROLLING_AMT_WINDOW = int(os.getenv("ROLLING_AMT_WINDOW", 100))
TXNAMT_Z_STRONG = int(os.getenv("Z_THRESHOLD", 50))

# =================== MONGO CONNECTOR ===================
class MongoConnector:
    def __init__(self, config, authSource="admin"):
        self.client = None
        self.db = None
        self.collection = None
        self.config = config
        self.authSource = authSource

    def connect(self):
        try:
            self.client = MongoClient(
                host=self.config["host"],
                port=self.config["port"],
                username=self.config["user"],
                password=self.config["password"],
                authSource=self.authSource
            )
            self.db = self.client[self.config["dbname"]]
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.exception("MongoDB connection failed")
            raise e

    def set_collection(self, collection_name):
        self.collection = self.db[collection_name]
        logger.debug(f"Mongo collection set: {collection_name}")

    def fetch_last_n_txns_per_device(self, n=102, projection=None):
        logger.debug(f"Fetching last {n} transactions per device")
        pipeline = [
            {"$match": {"actionStatus": 1}},
            {"$sort": {"createdAt": -1}},
            {
                "$setWindowFields": {
                    "partitionBy": "$deviceId",
                    "sortBy": {"txnTimeStamp": -1},
                    "output": {"row_num": {"$documentNumber": {}}}
                }
            },
            {"$match": {"row_num": {"$lte": n}}},
            {"$project": projection}
        ]
        return list(self.collection.aggregate(pipeline))

    def close(self):
        if self.client:
            self.client.close()
        logger.info("MongoDB connection closed")

# =================== MAIN FLOW ===================
if __name__ == "__main__":
    logger.info("Baseline generation job started")
    print("  Baseline Builder Service Started")
    print(f"  Logging Enabled  : {ENABLE_LOGGING}")
    print(f"  Rolling Window   : {ROLLING_AMT_WINDOW}")
    print(f"  Z Threshold      : {TXNAMT_Z_STRONG}")
    print(f"  Mongo DB         : {MONGO_CONFIG['dbname']}")
    print(f"  Mongo Host       : {MONGO_CONFIG['host']}")

    db = MongoConnector(MONGO_CONFIG, authSource=MONGO_AUTHSOURCE)
    db.connect()
    db.set_collection("transactionActionHistoryInfo")

    projection = {
        "deviceId": 1,
        "createdAt": 1,
        "txnAmt": 1,
        "actionId": 1,
        "_id": 0
    }

    results = db.fetch_last_n_txns_per_device(n=101, projection=projection)
    db.close()

    df = pd.DataFrame(results)

    if df.empty:
        logger.info("No transaction data fetched. Exiting job.")
        exit()

    logger.info(
        f"Fetched {df.shape[0]} transactions across {df['deviceId'].nunique()} devices"
    )
    print(f"\n  Transactions Fetched : {df.shape[0]}")
    print(f"  Total Devices        : {df['deviceId'].nunique()}")

    # ---- STEP 2: Time preprocessing ----
    df["txnTimeStamp"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

    ist = pytz.timezone("Asia/Kolkata")
    df["txnTimeStamp_IST"] = df["txnTimeStamp"].dt.tz_convert(ist).dt.tz_localize(None)
    df["weekday"] = df["txnTimeStamp_IST"].dt.day_name()
    df["txn_hour"] = df["txnTimeStamp_IST"].dt.hour

    df = df.dropna(subset=["txnTimeStamp"])

    # ---- STEP 3: Weekly opening / closing ----
    daily_open_close = (
        df.groupby(["deviceId", df["txnTimeStamp_IST"].dt.date])
        .agg(opening_time=("txn_hour", "min"), closing_time=("txn_hour", "max"))
        .reset_index()
    )

    daily_open_close["weekday"] = pd.to_datetime(
        daily_open_close["txnTimeStamp_IST"]
    ).dt.day_name()

    weekday_summary = (
        daily_open_close.groupby(["deviceId", "weekday"])
        .agg(
            avg_opening_time=("opening_time", "mean"),
            avg_closing_time=("closing_time", "mean")
        )
        .reset_index()
    )

    weekday_summary["avg_opening_time"] = (
        pd.to_datetime(weekday_summary["avg_opening_time"], unit="h")
        - pd.Timedelta(hours=2)
    ).dt.strftime("%H:%M")

    weekday_summary["avg_closing_time"] = (
        pd.to_datetime(weekday_summary["avg_closing_time"], unit="h")
        + pd.Timedelta(hours=2)
    ).dt.strftime("%H:%M")

    # ---- STEP 4: Z-score baseline ----
    df["txnAmt"] = pd.to_numeric(df["txnAmt"], errors="coerce").fillna(0)
    df = df.sort_values(["deviceId", "txnTimeStamp_IST"]).reset_index(drop=True)

    df["rolling_mean_amt"] = (
        df.groupby("deviceId")["txnAmt"]
        .rolling(window=ROLLING_AMT_WINDOW, min_periods=ROLLING_AMT_WINDOW)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["rolling_std_amt"] = (
        df.groupby("deviceId")["txnAmt"]
        .rolling(window=ROLLING_AMT_WINDOW, min_periods=ROLLING_AMT_WINDOW)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["zscore_amt"] = (
        (df["txnAmt"] - df["rolling_mean_amt"]) / df["rolling_std_amt"]
    )

    df["is_strong_outlier"] = df["zscore_amt"].abs() > TXNAMT_Z_STRONG

    valid_devices = (
        df.groupby("deviceId")
        .size()
        .reset_index(name="txn_count")
        .query("txn_count >= @ROLLING_AMT_WINDOW")["deviceId"]
    )

    baseline_df = (
        df[
            (df["deviceId"].isin(valid_devices)) &
            (~df["is_strong_outlier"])
        ]
        .dropna(subset=["rolling_mean_amt", "rolling_std_amt"])
    )

    latest_stats = (
        baseline_df
        .sort_values(["deviceId", "txnTimeStamp_IST"])
        .groupby("deviceId")
        .tail(1)[["deviceId", "rolling_mean_amt", "rolling_std_amt", "zscore_amt"]]
    )

    recent_txns = (
        baseline_df
        .groupby("deviceId")
        .tail(ROLLING_AMT_WINDOW)
        .groupby("deviceId")["txnAmt"]
        .apply(list)
        .reset_index()
        .rename(columns={"txnAmt": Z_SCORE_RECENT_TXNS})
    )

    latest_stats = latest_stats.merge(recent_txns, on="deviceId", how="left")

    latest_stats.rename(columns={
        "deviceId": Z_SCORE_DEVICE_ID,
        "rolling_mean_amt": Z_SCORE_MEAN,
        "rolling_std_amt": Z_SCORE_STD,
        "zscore_amt": Z_SCORE
    }, inplace=True)

    latest_stats[Z_SCORE_RECENT_TXNS] = latest_stats[Z_SCORE_RECENT_TXNS].apply(json.dumps)

    # ---- STEP 5: Persist to Postgres ----
    engine = create_engine(POSTGRES_URI)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    with engine.begin() as conn:
        weekday_summary.rename(columns={
            "avg_opening_time": DEVICE_DATA_EST_OPENING,
            "avg_closing_time": DEVICE_DATA_EST_CLOSING,
            "weekday": DEVICE_DATA_DAY,
            "deviceId": DEVICE_DATA_DEVICE_ID
        }, inplace=True)

        device_data_table = metadata.tables[DEVICE_DATA_TABLE]

        insert_device_data = insert(device_data_table).values(
            weekday_summary.to_dict(orient="records")
        )

        update_device_data = insert_device_data.on_conflict_do_update(
            index_elements=[DEVICE_DATA_DEVICE_ID, DEVICE_DATA_DAY],
            set_={
                DEVICE_DATA_EST_OPENING: insert_device_data.excluded[DEVICE_DATA_EST_OPENING],
                DEVICE_DATA_EST_CLOSING: insert_device_data.excluded[DEVICE_DATA_EST_CLOSING],
            }
        )

        conn.execute(update_device_data)

        if not latest_stats.empty:
            device_z_score_table = metadata.tables[DEVICE_Z_SCORE_TABLE]

            insert_zscore = insert(device_z_score_table).values(
                latest_stats.to_dict(orient="records")
            )

            update_zscore = insert_zscore.on_conflict_do_update(
                index_elements=[Z_SCORE_DEVICE_ID],
                set_={
                    Z_SCORE_MEAN: insert_zscore.excluded[Z_SCORE_MEAN],
                    Z_SCORE_STD: insert_zscore.excluded[Z_SCORE_STD],
                    Z_SCORE: insert_zscore.excluded[Z_SCORE],
                    Z_SCORE_RECENT_TXNS: insert_zscore.excluded[Z_SCORE_RECENT_TXNS],
                }
            )

            conn.execute(update_zscore)

    logger.info("Baseline data successfully stored into Postgres")
    print(f"\n  Devices with valid baseline : {latest_stats.shape[0]}")
    print("  Baseline successfully pushed to Postgres")
