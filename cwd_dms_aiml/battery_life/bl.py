# ======================== FINAL PIPELINE =========================================
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
import os
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()

MONGO_CONFIG = {
    "host": os.getenv("MONGO_HOST"),
    "dbname": os.getenv("MONGO_DBNAME"),
    "user": os.getenv("MONGO_USER"),
    "password": os.getenv("MONGO_PASSWORD"),
    "port": int(os.getenv("MONGO_PORT", 27017)),
}
MONGO_AUTHSOURCE = os.getenv("MONGO_AUTHSOURCE", "admin")

PG_URI = os.getenv("POSTGRES_URI")

# ========== FEATURE CONFIG ==========
IST_TIMEZONE = "Asia/Kolkata"

NUMERIC_COUNTERS = [
    "totalTxTrafficConsumed",
    "totalRxTrafficConsumed",
    "totalUSBPluginCount",
    "totalUSBPluginCountSincePowerOn",
    "totalUSBPluginDuration",
    "totalDeviceRebootCount",
    "totalModemResetCount",
    "totalNWFailureCount",
    "totalNWDiscDueToBadRSSICount",
    "mqttConnectionFailCount",
    "httpPostFailCount",
    "httpDownloadFailCount",
    "totalTransactionsPlayed",
    "totalTransactionsFailedToPlay",
    "totalFilesDownloadedCount",
    "volumeUpPressCounts",
    "volumeDownPressCounts",
    "replayPressCounts",
    "totalSystemAudioChangeCount",
    "totalLanguageChangeCount",
    "lastUSBPluginTimestamp",
]

STRING_FEATURES = [
    "operatorName",
    "networkType",
    "firmwareVersion",
    "deviceModemFirmWareName",
]

SIGNAL_COLUMNS = ["rsrp", "rsrq", "snr", "rxlev", "rscp", "ecno", "signalStrength"]

ACTIVITY_COLUMNS = [
    "volumeUpPressCounts",
    "volumeDownPressCounts",
    "replayPressCounts",
    "totalSystemAudioChangeCount",
    "totalLanguageChangeCount",
]

FAILURE_COLUMNS = [
    "totalNWFailureCount",
    "totalNWDiscDueToBadRSSICount",
    "mqttConnectionFailCount",
    "httpPostFailCount",
    "httpDownloadFailCount",
]

# ========== MONGO CONNECTOR ==========
class MongoConnector:
    def __init__(self, config, authSource="admin"):
        self.client = None
        self.db = None
        self.collection = None
        self.config = config
        self.authSource = authSource

    def connect(self):
        self.client = MongoClient(
            host=self.config["host"],
            port=self.config["port"],
            username=self.config["user"],
            password=self.config["password"],
            authSource=self.authSource
        )
        self.db = self.client[self.config["dbname"]]
        print("✅ Connected to MongoDB")

    def set_collection(self, collection_name):
        self.collection = self.db[collection_name]

    def fetch_device_stats(self):
        projection = {
            "createdAt": 1,
            "metadata.deviceId": 1,
            "batteryLevel": 1,
            "chargingStatus": 1,
            "totalTxTrafficConsumed": 1,
            "totalRxTrafficConsumed": 1,
            "totalUSBPluginCount": 1,
            "totalUSBPluginCountSincePowerOn": 1,
            "totalUSBPluginDuration": 1,
            "totalDeviceRebootCount": 1,
            "totalModemResetCount": 1,
            "totalNWFailureCount": 1,
            "totalNWDiscDueToBadRSSICount": 1,
            "mqttConnectionFailCount": 1,
            "httpPostFailCount": 1,
            "httpDownloadFailCount": 1,
            "totalTransactionsPlayed": 1,
            "totalTransactionsFailedToPlay": 1,
            "totalFilesDownloadedCount": 1,
            "volumeUpPressCounts": 1,
            "volumeDownPressCounts": 1,
            "replayPressCounts": 1,
            "totalSystemAudioChangeCount": 1,
            "totalLanguageChangeCount": 1,
            "lastUSBPluginTimestamp": 1,
            "rsrp": 1,
            "rsrq": 1,
            "snr": 1,
            "rxlev": 1,
            "rscp": 1,
            "ecno": 1,
            "signalStrength": 1,
            "operatorName": 1,
            "networkType": 1,
            "firmwareVersion": 1,
            "deviceModemFirmWareName": 1,
            "_id": 0
        }
        return list(self.collection.find({}, projection))

    def close(self):
        if self.client:
            self.client.close()
        print("🔌 Mongo connection closed")


# ========== HELPERS ==========
def convert_to_ist(ts):
    """Convert ISO datetime string to IST without ms."""
    dt = pd.to_datetime(ts, utc=True)
    ist = dt.tz_convert(IST_TIMEZONE).tz_localize(None)  # drop tz info
    return ist.replace(microsecond=0)


def convert_epoch_to_ist(value):
    """Convert epoch seconds to naive IST datetime."""
    if pd.isna(value):
        return pd.NaT
    try:
        dt = pd.to_datetime(int(value), unit="s", utc=True)
        return dt.tz_convert(IST_TIMEZONE).tz_localize(None)
    except (ValueError, OverflowError, TypeError):
        return pd.NaT


def prepare_feature_columns(df):
    """Ensure required columns exist and derive helper counters."""
    for col in NUMERIC_COUNTERS + SIGNAL_COLUMNS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in STRING_FEATURES:
        if col not in df:
            df[col] = None
        df[col] = df[col].astype("object")

    if "lastUSBPluginTimestamp" not in df:
        df["lastUSBPluginTimestamp"] = np.nan

    if not ACTIVITY_COLUMNS:
        df["activity_counter"] = 0.0
    else:
        df[ACTIVITY_COLUMNS] = df[ACTIVITY_COLUMNS].fillna(0)
        df["activity_counter"] = df[ACTIVITY_COLUMNS].sum(axis=1)

    if not FAILURE_COLUMNS:
        df["network_failure_counter"] = 0.0
    else:
        df[FAILURE_COLUMNS] = df[FAILURE_COLUMNS].fillna(0)
        df["network_failure_counter"] = df[FAILURE_COLUMNS].sum(axis=1)

    df["reset_counter"] = (
        df["totalDeviceRebootCount"].fillna(0)
        + df["totalModemResetCount"].fillna(0)
    )

    df["radio_quality_score"] = df.apply(compute_radio_quality, axis=1)
    df["last_usb_dt"] = df["lastUSBPluginTimestamp"].apply(convert_epoch_to_ist)
    return df


def normalize_signal(value, low, high, invert=False):
    """Normalize a signal metric to 0-1."""
    if pd.isna(value):
        return np.nan
    value = max(min(value, high), low)
    norm = (value - low) / (high - low) if high != low else 0.0
    if invert:
        return 1 - norm
    return norm


def compute_radio_quality(row):
    """Blend multiple signal metrics into a single score."""
    scores = []
    scores.append(normalize_signal(row.get("rsrp"), -120, -70, invert=True))
    scores.append(normalize_signal(row.get("rsrq"), -20, -3, invert=True))
    scores.append(normalize_signal(row.get("snr"), 0, 30, invert=False))
    scores.append(normalize_signal(row.get("rxlev"), 0, 63, invert=False))
    scores.append(normalize_signal(row.get("signalStrength"), 0, 31, invert=False))
    valid_scores = [s for s in scores if not pd.isna(s)]
    if not valid_scores:
        return np.nan
    return float(np.clip(np.mean(valid_scores), 0, 1))


def safe_rate(delta, minutes):
    delta = np.asarray(delta, dtype=float)
    minutes = np.asarray(minutes, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = delta / minutes

    out[(minutes <= 0) | np.isnan(minutes) | np.isnan(delta)] = np.nan
    return out


def safe_per_hour(delta, minutes):
    delta = np.asarray(delta, dtype=float)
    minutes = np.asarray(minutes, dtype=float)
    hours = minutes / 60.0

    with np.errstate(divide='ignore', invalid='ignore'):
        out = delta / hours

    out[(hours <= 0) | np.isnan(hours) | np.isnan(delta)] = np.nan
    return out

def get_thresholds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def get_anomaly_reason(row, thresholds):
    status = row['cs']  # lowercase here

    if status == 1:  # Charging
        low, high = thresholds.get(1, (None, None))
        rate = row['avg_rate']
        if low is None or high is None:
            return "Normal"
        if rate < low:
            return f"Charging too slow: {rate:.2f}%/min (expected ≥{low:.2f})"
        elif rate > high:
            return f"Charging too fast: {rate:.2f}%/min (expected ≤{high:.2f})"
        return "Normal"

    elif status == 2:  # Not charging / discharging
        low, high = thresholds.get(2, (None, None))
        rate = row['avg_rate']
        if low is None or high is None:
            return "Normal"
        if rate > high:
            return f"Discharging too slow: {rate:.2f}%/min (expected ≤{high:.2f})"
        elif rate < low:
            return f"Draining too fast: {rate:.2f}%/min (expected ≥{low:.2f})"
        return "Normal"

    elif status == 0:  # Unknown
        return "Unknown status"

    return "Normal"


# ========== MAIN FLOW ==========
if __name__ == "__main__":
    # --- Mongo Fetch ---
    db = MongoConnector(MONGO_CONFIG, authSource=MONGO_AUTHSOURCE)
    db.connect()
    db.set_collection("deviceStatHistoryInfo")
    raw = db.fetch_device_stats()
    db.close()

    df = pd.DataFrame(raw)
    df.rename(columns={
        "metadata": "metadata",
        "batteryLevel": "BL",
        "chargingStatus": "CS"
    }, inplace=True)
    print(df.columns)
    # Extract deviceId
    df['deviceId'] = df['metadata'].apply(lambda x: x.get("deviceId") if isinstance(x, dict) else None)
    df.drop(columns=['metadata'], inplace=True)

    # Convert timestamps
    df['dateTime_IST_no_ms'] = df['createdAt'].apply(convert_to_ist)
    df.drop(columns=['createdAt'], inplace=True)

    # Prepare additional telemetry columns
    df = prepare_feature_columns(df)

    print(f"📊 Loaded {len(df)} records from Mongo")

    all_blocks = []

    # --- Process each device ---
    for device_id, group in df.groupby("deviceId"):
        print("Enter in loop")
        device_df = group.sort_values("dateTime_IST_no_ms").copy()
        device_df['CS'] = device_df['CS'].astype("int64")
        print("pass stage 1")
        # Session blocks
        device_df['block'] = (device_df['CS'] != device_df['CS'].shift()).cumsum()
        print("pass stage 2")
        blocks = (
            device_df.groupby('block', as_index=False)
            .agg(
                device_id=('deviceId', 'first'),
                cs=('CS', 'first'),
                start_time=('dateTime_IST_no_ms', 'first'),
                end_time=('dateTime_IST_no_ms', 'last'),
                start_bl=('BL', 'first'),
                end_bl=('BL', 'last'),
                n_rows=('BL', 'size'),
                start_tx=('totalTxTrafficConsumed', 'first'),
                end_tx=('totalTxTrafficConsumed', 'last'),
                start_rx=('totalRxTrafficConsumed', 'first'),
                end_rx=('totalRxTrafficConsumed', 'last'),
                start_activity=('activity_counter', 'first'),
                end_activity=('activity_counter', 'last'),
                start_fail=('network_failure_counter', 'first'),
                end_fail=('network_failure_counter', 'last'),
                start_usb_count=('totalUSBPluginCount', 'first'),
                end_usb_count=('totalUSBPluginCount', 'last'),
                start_usb_duration=('totalUSBPluginDuration', 'first'),
                end_usb_duration=('totalUSBPluginDuration', 'last'),
                start_reset=('reset_counter', 'first'),
                end_reset=('reset_counter', 'last'),
                start_last_usb=('last_usb_dt', 'first'),
                avg_radio_quality=('radio_quality_score', 'mean'),
                operator_name=('operatorName', 'first'),
                network_type=('networkType', 'first'),
                firmware_version=('firmwareVersion', 'first'),
                modem_firmware=('deviceModemFirmWareName', 'first'),
            )
        )
        print("pass stage 3")
        blocks['delta_bl'] = blocks['end_bl'] - blocks['start_bl']
        blocks['delta_minutes'] = (
            (blocks['end_time'] - blocks['start_time']).dt.total_seconds() / 60.0
        )
        print("pass stage 4")
        blocks['avg_rate'] = np.where(
            blocks['delta_minutes'] > 0,
            blocks['delta_bl'] / blocks['delta_minutes'],
            np.nan
        )
        print("pass stage 5")
        blocks['delta_tx'] = blocks['end_tx'] - blocks['start_tx']
        blocks['delta_rx'] = blocks['end_rx'] - blocks['start_rx']
        blocks['delta_activity'] = blocks['end_activity'] - blocks['start_activity']
        blocks['network_failure_events'] = blocks['end_fail'] - blocks['start_fail']
        blocks['usb_plugins'] = np.maximum(blocks['end_usb_count'] - blocks['start_usb_count'], 0)
        blocks['usb_duration_minutes'] = np.maximum(
            (blocks['end_usb_duration'] - blocks['start_usb_duration']) / 60.0,
            0
        )
        blocks['reset_events'] = np.maximum(blocks['end_reset'] - blocks['start_reset'], 0)
        blocks['tx_per_min'] = safe_rate(blocks['delta_tx'], blocks['delta_minutes'])
        blocks['rx_per_min'] = safe_rate(blocks['delta_rx'], blocks['delta_minutes'])
        blocks['activity_per_min'] = safe_rate(blocks['delta_activity'], blocks['delta_minutes'])
        blocks['network_failures_per_hour'] = safe_per_hour(
            blocks['network_failure_events'], blocks['delta_minutes']
        )
        blocks['minutes_since_last_charge'] = (
            (blocks['start_time'] - blocks['start_last_usb']).dt.total_seconds() / 60.0
        )
        blocks['avg_radio_quality'] = blocks['avg_radio_quality'].clip(0, 1)
        blocks['minutes_since_last_charge'] = blocks['minutes_since_last_charge'].where(
            blocks['minutes_since_last_charge'] >= 0
        )
        # Robust slope
        tmp = device_df[['block', 'dateTime_IST_no_ms', 'BL']].copy()
        tmp['delta_BL'] = tmp['BL'].diff()
        tmp['delta_min'] = tmp['dateTime_IST_no_ms'].diff().dt.total_seconds() / 60.0
        tmp['rate_per_min'] = tmp['delta_BL'] / tmp['delta_min']
        robust = (
            tmp.groupby('block')['rate_per_min']
            .median()
            .rename('median_rate_per_min')
            .reset_index()
        )
        print("pass stage 6")
        blocks = blocks.merge(robust, on='block', how='left')
        helper_cols = [
            'start_tx', 'end_tx', 'start_rx', 'end_rx', 'start_activity', 'end_activity',
            'start_fail', 'end_fail', 'start_usb_count', 'end_usb_count',
            'start_usb_duration', 'end_usb_duration', 'start_reset', 'end_reset',
            'start_last_usb'
        ]
        blocks.drop(columns=[c for c in helper_cols if c in blocks.columns], inplace=True)
        print("entering in stage 7")
        # Thresholds per device
        thresholds = {}
        for cs_val in [0, 1, 2]:
            sub = blocks[blocks['cs'] == cs_val]['avg_rate']
            if not sub.empty:
                thresholds[cs_val] = get_thresholds(sub)
            else:
                thresholds[cs_val] = (None, None)

        # Anomaly labeling
        blocks['anomaly_reason'] = blocks.apply(lambda r: get_anomaly_reason(r, thresholds), axis=1)
        blocks['is_anomaly'] = blocks['anomaly_reason'].apply(
            lambda x: 0 if x in ["Normal", "Unknown status"] else 1
        )
        print("exiting stage 7")
        all_blocks.append(blocks)

    final_df = pd.concat(all_blocks, ignore_index=True)
    print("✅ Processed session blocks:", final_df.shape)

    from sqlalchemy import Table, MetaData
    from sqlalchemy.dialects.postgresql import insert

    # --- Save to Postgres with UPSERT ---
    engine = create_engine(PG_URI)
    meta = MetaData()
    meta.reflect(bind=engine)
    table = meta.tables['battery_health']

    result_columns = [
        "block", "device_id", "cs", "start_time", "end_time",
        "start_bl", "end_bl", "n_rows", "delta_bl",
        "delta_minutes", "avg_rate", "median_rate_per_min",
        "delta_tx", "delta_rx", "tx_per_min", "rx_per_min",
        "activity_per_min", "network_failure_events",
        "network_failures_per_hour", "usb_plugins",
        "usb_duration_minutes", "reset_events", "avg_radio_quality",
        "minutes_since_last_charge", "operator_name", "network_type",
        "firmware_version", "modem_firmware", "is_anomaly",
        "anomaly_reason"
    ]

    db_ready = final_df.replace([np.inf, -np.inf], np.nan)
    db_ready = db_ready.where(pd.notnull(db_ready), None)
    
    rows = db_ready[result_columns].to_dict(orient="records")
    

    with engine.begin() as conn:
        for row in rows:
            stmt = insert(table).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=['block', 'device_id'],  # your PK
                set_={c: stmt.excluded[c] for c in row if c not in ['block', 'device_id']}
            )
            conn.execute(stmt)

    print("📥 Data upserted into Postgres (inserted or updated)")
    
    # =====================================================
    # --- NEW: Compute BL_score per device and save it ---
    # =====================================================
    device_scores = []
    for device_id, group in final_df.groupby("device_id"):
        delta_minutes = pd.to_numeric(group["delta_minutes"], errors="coerce")
        total_minutes = delta_minutes.sum(skipna=True)

        anomaly_minutes = pd.to_numeric(
            group.loc[group["is_anomaly"] == 1, "delta_minutes"], errors="coerce"
        ).sum(skipna=True)

        anomaly_component = 1.0
        if total_minutes > 0 and np.isfinite(anomaly_minutes):
            anomaly_component = 1 - (anomaly_minutes / total_minutes)

        discharge_blocks = group[group["cs"] == 2]
        discharge_rate = 0.0
        if not discharge_blocks.empty:
            discharge_rate = abs(
                pd.to_numeric(discharge_blocks["avg_rate"], errors="coerce")
                .clip(upper=0)
                .mean(skipna=True)
            )
            if np.isnan(discharge_rate):
                discharge_rate = 0.0

        discharge_drop = 0.0
        if not discharge_blocks.empty:
            drop_values = (
                pd.to_numeric(discharge_blocks["start_bl"], errors="coerce")
                - pd.to_numeric(discharge_blocks["end_bl"], errors="coerce")
            )
            discharge_drop = drop_values.clip(lower=0).mean(skipna=True)
            if np.isnan(discharge_drop):
                discharge_drop = 0.0

        radio_quality = pd.to_numeric(
            discharge_blocks["avg_radio_quality"], errors="coerce"
        ).dropna().mean()
        if pd.isna(radio_quality):
            radio_quality = 0.5

        expected_discharge = 0.8 + 0.4 * (1 - radio_quality)
        drain_penalty = 0.0
        if expected_discharge > 0:
            drain_penalty = min(discharge_rate / expected_discharge, 1.0)

        drop_penalty = np.clip(discharge_drop / 30.0, 0.0, 1.0)
        drain_penalty = min(0.5 * drain_penalty + 0.5 * drop_penalty, 1.0)

        total_hours = total_minutes / 60.0 if total_minutes > 0 else 0.0

        reset_sum = pd.to_numeric(group["reset_events"], errors="coerce").sum(skipna=True)
        usb_sum = pd.to_numeric(group["usb_plugins"], errors="coerce").sum(skipna=True)

        reset_rate = reset_sum / total_hours if total_hours > 0 else 0.0
        usb_rate = usb_sum / total_hours if total_hours > 0 else 0.0
        stability_penalty = min((reset_rate + 0.5 * usb_rate) / 4.0, 1.0)

        coverage_target_minutes = 12 * 60  # 12 hours reference window
        coverage_component = np.clip(total_minutes / coverage_target_minutes, 0.0, 1.0)

        components = np.array([
            anomaly_component,
            1 - drain_penalty,
            1 - stability_penalty,
            coverage_component
        ], dtype=float)
        components[~np.isfinite(components)] = 0.0

        score = 100 * (
            0.4 * components[0]
            + 0.3 * components[1]
            + 0.2 * components[2]
            + 0.1 * components[3]
        )
        score = float(np.clip(score, 0, 100))
        device_scores.append({"device_id": device_id, "device_bs": score})

    bl_score_df = pd.DataFrame(device_scores)

    # --- Reflect BL_score table from DB ---
    meta.reflect(bind=engine)
    bl_table = meta.tables['bl_score']

    rows = bl_score_df.to_dict(orient="records")

    with engine.begin() as conn:
        for row in rows:
            stmt = insert(bl_table).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["device_id"],
                set_={"device_bs": stmt.excluded.device_bs}
            )
            conn.execute(stmt)

    print("📥 Device scores upserted into BL_score")