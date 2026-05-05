import nt
import os
import json
import numpy as np
import pandas as pd
import pytz
import logging
import sys   # ADD THIS
from datetime import datetime
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8') 
load_dotenv(override=True)
# =================== LOGGING SETUP ===================
LOG_DIR = 'anomaly_real_time_LOGS'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
    print(f"Log directory created!")
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
logger = logging.getLogger("anomaly_detection")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Developer log formatter - Technical and detailed
dev_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)

# Client log formatter - Simple and business-friendly
client_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

# Developer handler - All levels (DEBUG and above)
dev_handler = logging.FileHandler(f"{LOG_DIR}/developer.log", encoding="utf-8")
dev_handler.setLevel(logging.DEBUG)
dev_handler.setFormatter(dev_formatter)
dev_handler.stream.reconfigure(line_buffering=True)
# Client handler - Only INFO and above (no DEBUG)
client_handler = logging.FileHandler(f"{LOG_DIR}/client.log", encoding="utf-8")
client_handler.setLevel(logging.INFO)
client_handler.setFormatter(client_formatter)

logger.addHandler(dev_handler)
logger.addHandler(client_handler)

if not ENABLE_LOGGING:
    logger.disabled = True
logger.info("Anomaly Detection Service Starting Up")

# =================== LOAD ENV ===================
logger.info("Environment variables loaded from .env file")

REQUIRED_ENV_VARS = [
    "POSTGRES_URI_2",
    "DEVICE_DATA_TABLE",
    "DEVICE_Z_SCORE_TABLE",
    "ANOMALY_RESULTS_TABLE",
    "DEVICE_DATA_DEVICE_ID",
    "DEVICE_DATA_DAY",
    "DEVICE_DATA_EST_OPENING",
    "DEVICE_DATA_EST_CLOSING",
    "Z_SCORE_DEVICE_ID",
    "Z_SCORE_MEAN",
    "Z_SCORE_STD",
    "Z_SCORE",
    "Z_SCORE_RECENT_TXNS",
    "ANOMALY_DEVICE_ID",
    "ANOMALY_TXN_ID",
    "ANOMALY_TS",
    "ANOMALY_AMT",
    "ANOMALY_Z",
    "ANOMALY_CONF",
    "ANOMALY_LABEL",
    "ANOMALY_OPEN",
    "ANOMALY_CLOSE",
    "ANOMALY_CURRENT_THRESHOLD",

]

missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    logger.error(f"Application cannot start - Missing required configuration: {', '.join(missing)}")
    raise RuntimeError(f"Missing ENV variables: {missing}")

logger.info(f"All {len(REQUIRED_ENV_VARS)} required environment variables validated and loaded successfully")

POSTGRES_URI = os.getenv("POSTGRES_URI_2")
engine = create_engine(POSTGRES_URI) # type: ignore
logger.info("Database connection established successfully")

ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", 100))

# Tables
DEVICE_DATA_TABLE = os.getenv("DEVICE_DATA_TABLE")
DEVICE_Z_SCORE_TABLE = os.getenv("DEVICE_Z_SCORE_TABLE")
ANOMALY_RESULTS_TABLE = os.getenv("ANOMALY_RESULTS_TABLE")

# Columns
DEVICE_DATA_DEVICE_ID = os.getenv("DEVICE_DATA_DEVICE_ID")
DEVICE_DATA_DAY = os.getenv("DEVICE_DATA_DAY")
DEVICE_DATA_EST_OPENING = os.getenv("DEVICE_DATA_EST_OPENING")
DEVICE_DATA_EST_CLOSING = os.getenv("DEVICE_DATA_EST_CLOSING")

Z_SCORE_DEVICE_ID = os.getenv("Z_SCORE_DEVICE_ID")
Z_SCORE_MEAN = os.getenv("Z_SCORE_MEAN")
Z_SCORE_STD = os.getenv("Z_SCORE_STD")
Z_SCORE = os.getenv("Z_SCORE")
Z_SCORE_RECENT_TXNS = os.getenv("Z_SCORE_RECENT_TXNS")

ANOMALY_DEVICE_ID = os.getenv("ANOMALY_DEVICE_ID")
ANOMALY_TXN_ID = os.getenv("ANOMALY_TXN_ID")
ANOMALY_TS = os.getenv("ANOMALY_TS")
ANOMALY_AMT = os.getenv("ANOMALY_AMT")
ANOMALY_Z = os.getenv("ANOMALY_Z")
ANOMALY_CONF = os.getenv("ANOMALY_CONF")
ANOMALY_LABEL = os.getenv("ANOMALY_LABEL")
ANOMALY_OPEN = os.getenv("ANOMALY_OPEN")
ANOMALY_CLOSE = os.getenv("ANOMALY_CLOSE")
ANOMALY_CURRENT_THRESHOLD = os.getenv("ANOMALY_CURRENT_THRESHOLD")


app = Flask(__name__)
logger.info("Flask application initialized")

# Default business hours when schedule is not available
DEFAULT_OPEN_HOUR = 9   # 9 AM
DEFAULT_CLOSE_HOUR = 21  # 9 PM
# =================== THRESHOLD ===================
def get_threshold():
    logger.debug("Attempting to fetch threshold value from database")
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT threshold_value FROM thresholds LIMIT 1")
            ).fetchone()
        threshold = int(row[0]) if row else 100
        logger.info(f"Anomaly detection threshold set to: {threshold}")
        return threshold
    except Exception as e:
        logger.warning(f"Could not retrieve threshold from database, using default value of 50")
        logger.debug(f"Threshold fetch error details: {str(e)}")
        return 100

Z_THRESHOLD = get_threshold()
print(f"  Anomaly Detection Service Started")
print(f"  Port: 9198")
print(f"  Logging Enabled: {ENABLE_LOGGING}")
print(f"  Z-Score Threshold: {Z_THRESHOLD}")
print(f"  Rolling Window: {ROLLING_WINDOW}")
# =================== HELPERS ===================
def preprocess_timestamp(created_at: str):
    """Convert timestamp to UTC and IST, extract weekday"""
    logger.debug(f"Preprocessing timestamp: {created_at}")
    try:
        ts_utc = pd.to_datetime(created_at, utc=True)
        ist = pytz.timezone("Asia/Kolkata")
        ts_ist = ts_utc.tz_convert(ist)
        weekday = ts_ist.strftime("%A")
        logger.debug(f"Timestamp processed - UTC: {ts_utc}, IST: {ts_ist}, Weekday: {weekday}")
        return ts_utc, ts_ist, weekday
    except Exception as e:
        logger.error(f"Failed to process transaction timestamp")
        logger.debug(f"Timestamp error details - Input: '{created_at}', Error: {str(e)}")
        raise

def calculate_confidence(z_score, txn_amt, mean_val, txn_hour, open_hour, close_hour, std_val, Z_THRESHOLD):
    """Calculate anomaly confidence based on amount and time"""
    logger.debug(f"Calculating confidence - z_score: {z_score}, txn_amt: {txn_amt}, mean: {mean_val}, hour: {txn_hour}, open: {open_hour}, close: {close_hour}, std: {std_val}")
    
    threshold_amt = mean_val + Z_THRESHOLD * std_val if std_val else mean_val * 2
    logger.debug(f"Threshold amount calculated: {threshold_amt}")

    # Extreme cases
    if abs(z_score) > 1000 * Z_THRESHOLD or txn_amt > 100 * threshold_amt:
        logger.warning(f"Extreme anomaly pattern detected - Transaction amount significantly exceeds normal range")
        logger.debug(f"Extreme anomaly details - z_score: {z_score}, txn_amt: {txn_amt}")
        return {"anomaly_confidence": 100.0, "normal_confidence": 0.0}

    if np.isnan(z_score):
        logger.warning("Unable to calculate statistical confidence, returning neutral assessment")
        logger.debug("Z-score is NaN")
        return {"anomaly_confidence": 50.0, "normal_confidence": 50.0}

    # Calculate amount-based anomaly score
    amount_anomaly = min(abs(z_score) / (Z_THRESHOLD if Z_THRESHOLD > 0 else 1) * 100, 100)
    print("This is logic of CS and score for amount --->amount_anomaly: ",amount_anomaly)
    logger.debug(f"Amount-based anomaly score: {amount_anomaly}")

    # Use schedule if available, otherwise use default business hours
    effective_open = open_hour if open_hour is not None else DEFAULT_OPEN_HOUR
    print("Effective opening time :  ",effective_open)

    effective_close = close_hour if close_hour is not None else DEFAULT_CLOSE_HOUR
    print("Effective_close time : ",effective_close)
    
    if open_hour is None or close_hour is None:
        logger.debug(f"Using default business hours - Open: {effective_open}, Close: {effective_close}")
    else:
        logger.debug(f"Using device schedule - Open: {effective_open}, Close: {effective_close}")
    
    # Calculate buffer zones
    day_hrs = 24 - (effective_close - effective_open)

    buffer = max(round(day_hrs / 8), 1)
    print("Buffer zone hours -->: ",buffer)
    logger.debug(f"Buffer zone calculated: {buffer} hours")
    
    # Determine time-based anomaly
    time_anomaly = 0.0
    is_outside_hours = False
    
    if effective_open <= txn_hour <= effective_close:
        # Within operating hours
        time_anomaly = 0
        is_outside_hours = False
        print("Transaction within operating hours")
        logger.debug(f"Transaction within operating hours (hour {txn_hour})")
    elif effective_open - buffer <= txn_hour <= effective_close + buffer:
        # Just outside hours (buffer zone 1)
        time_anomaly = 50
        is_outside_hours = False
        print("Transaction in buffer zone 1 (just outside hours)")
        logger.debug(f"Transaction in buffer zone 1 (hour {txn_hour})")
    elif effective_open - 2 * buffer <= txn_hour <= effective_close + 2 * buffer:
        # Further outside (buffer zone 2)
        time_anomaly = 70
        is_outside_hours = True
        print("Transaction in buffer zone 2 (further outside hours)")
        logger.debug(f"Transaction in buffer zone 2 (hour {txn_hour})")
    else:
        # Significantly outside store hours (e.g., late night)
        time_anomaly = 100
        is_outside_hours = True
        print("Transaction significantly outside normal business hours")
        logger.warning(f"Transaction occurred outside normal business hours (hour: {txn_hour})")
        logger.debug(f"Transaction significantly outside hours (hour {txn_hour})")
    
    # Calculate final anomaly confidence
    if is_outside_hours:
        # For out-of-hours transactions, heavily weight time factor
        # Ensure minimum 70% confidence for significantly out-of-hours transactions
        anomaly_conf = max(70.0, min(0.7 * time_anomaly + 0.3 * amount_anomaly, 100))
        print("Out-of-hours transaction - anomaly confidence heavily influenced by time factor-->", anomaly_conf)
        logger.debug(f"Out-of-hours weighting applied - time: 70%, amount: 30%")
    else:
        # During hours or close buffer - use balanced weighting
        weight_time = 0.3
        weight_amt = 0.7
        anomaly_conf = min(weight_amt * amount_anomaly + weight_time * time_anomaly, 100)
        logger.debug(f"Standard weighting applied - time: 30%, amount: 70%")
    
    result = {
        "anomaly_confidence": round(anomaly_conf, 2),
        "normal_confidence": round(100 - anomaly_conf, 2)
    }
    logger.debug(f"Confidence calculated - Anomaly: {result['anomaly_confidence']}%, Normal: {result['normal_confidence']}%")
    return result

def get_historical_txn_count(device_id):
    """Get the count of historical transactions BEFORE processing current one"""
    logger.debug(f"Fetching historical transaction count for device {device_id}")
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(f'''SELECT "{Z_SCORE_RECENT_TXNS}" FROM "{DEVICE_Z_SCORE_TABLE}" WHERE "{Z_SCORE_DEVICE_ID}" = :id'''),
                {"id": device_id}
            ).fetchone()

            if not row or row[0] in (None, ""):
                logger.debug(f"No historical transactions found for device {device_id}")
                return 0
            elif isinstance(row[0], list):
                count = len(row[0])
                logger.debug(f"Historical transaction count for device {device_id}: {count} (list type)")
                return count
            elif isinstance(row[0], str):
                try: 
                    txns = json.loads(row[0])
                    count = len(txns)
                    logger.debug(f"Historical transaction count for device {device_id}: {count} (JSON string type)")
                    return count
                except Exception as e:
                    logger.error(f"Data format issue detected for device {device_id} transaction history")
                    logger.debug(f"Failed to parse transaction JSON for device {device_id}: {str(e)}")
                    return 0
            else:
                logger.warning(f"Unexpected data format found for device {device_id} transaction history")
                logger.debug(f"Unexpected data type for transactions in device {device_id}: {type(row[0])}")
                return 0
    except Exception as e:
        logger.error(f"Could not retrieve transaction history for device {device_id}")
        logger.debug(f"Error getting historical count for device {device_id}: {str(e)}")
        return 0

def calculate_device_stats(device_id, txnAmt):
    """
    STEP 1: Calculate statistics using ONLY historical data (before adding new transaction)
    This gives us a clean baseline to detect anomalies against.
    Does NOT update the database yet.
    """
    logger.debug(f"Calculating device statistics for device {device_id}, transaction amount: {txnAmt}")
    
    with engine.begin() as conn:
        row = conn.execute(
            text(f'''SELECT "{Z_SCORE_RECENT_TXNS}" FROM "{DEVICE_Z_SCORE_TABLE}" WHERE "{Z_SCORE_DEVICE_ID}" = :id'''),
            {"id": device_id}
        ).fetchone()

        # Parse existing transaction history
        if not row or row[0] in (None, ""):
            recent_txns = []
            logger.debug(f"No existing transaction history for device {device_id}")
        elif isinstance(row[0], list):
            recent_txns = row[0]
            logger.debug(f"Loaded {len(recent_txns)} historical transactions (list type)")
        elif isinstance(row[0], str):
            try: 
                recent_txns = json.loads(row[0])
                logger.debug(f"Loaded {len(recent_txns)} historical transactions (JSON string)")
            except Exception as e:
                logger.error(f"Could not load historical data for device {device_id}")
                logger.debug(f"Failed to parse transaction history: {str(e)}")
                recent_txns = []
        else:
            logger.warning(f"Unexpected data format in transaction history for device {device_id}")
            logger.debug(f"Unexpected transaction data type: {type(row[0])}")
            recent_txns = []

        # Calculate statistics using ONLY historical transactions
        if len(recent_txns) >= 2:
            mean_val = float(np.mean(recent_txns))
            std_val = float(np.std(recent_txns, ddof=1))
            z_score = (txnAmt - mean_val) / std_val if std_val > 0 else 0.0
            logger.debug(f"Stats calculated (≥2 txns) - Mean: {mean_val:.2f}, Std: {std_val:.2f}, Z-score: {z_score:.2f}")
        elif len(recent_txns) == 1:
            mean_val, std_val, z_score = recent_txns[0], 0.0, 0.0
            logger.debug(f"Stats calculated (1 txn) - Mean: {mean_val}, Std: 0.0, Z-score: 0.0")
        else:
            mean_val, std_val, z_score = txnAmt, 0.0, 0.0
            logger.debug(f"Stats calculated (0 txns) - Using current amount as baseline: {txnAmt}")

    # Return clean statistics WITHOUT updating database
    logger.debug(f"Returning statistics - mean: {mean_val}, std: {std_val}, z_score: {z_score}, history_count: {len(recent_txns)}")
    return mean_val, std_val, z_score, recent_txns

def finalize_device_stats(device_id, txnAmt, recent_txns, is_anomaly):
    """
    STEP 2: Update the database AFTER we've determined if transaction is anomaly
    Only add transaction to baseline if it's NOT an anomaly
    This keeps our baseline clean for future detection
    """
    logger.debug(f"Finalizing device statistics for device {device_id} - is_anomaly: {is_anomaly}")
    
    # Decision: Should we add this transaction to our baseline?
    if not is_anomaly:
        # Normal transaction - add to baseline
        recent_txns.append(txnAmt)
        logger.debug(f"Normal transaction - adding {txnAmt} to baseline (new count: {len(recent_txns)})")
        if len(recent_txns) > ROLLING_WINDOW:
            removed = recent_txns.pop(0)
            logger.debug(f"Rolling window exceeded - removed oldest transaction: {removed}")
    else:
        logger.debug(f"Anomaly detected - NOT adding transaction to baseline (maintaining count: {len(recent_txns)})")
    
    # Recalculate final statistics with updated transaction list
    if len(recent_txns) >= 2:
        mean_val = float(np.mean(recent_txns))
        std_val = float(np.std(recent_txns, ddof=1))
        # Note: z_score here is for storage, actual detection used earlier z_score
        z_score = (txnAmt - mean_val) / std_val if std_val > 0 else 0.0
        logger.debug(f"Final stats (≥2 txns) - Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    elif len(recent_txns) == 1:
        mean_val, std_val, z_score = recent_txns[0], 0.0, 0.0
        logger.debug(f"Final stats (1 txn) - Mean: {mean_val}")
    else:
        mean_val, std_val, z_score = txnAmt, 0.0, 0.0
        logger.debug(f"Final stats (0 txns) - Mean: {mean_val}")

    # Update database with final values
    logger.debug(f"Updating database for device {device_id}")
    try:
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO "{DEVICE_Z_SCORE_TABLE}" ("{Z_SCORE_DEVICE_ID}", "{Z_SCORE_MEAN}", "{Z_SCORE_STD}", "{Z_SCORE}", "{Z_SCORE_RECENT_TXNS}")
                    VALUES (:id, :mean, :std, :z, :txns) ON CONFLICT ("{Z_SCORE_DEVICE_ID}")
                    DO UPDATE SET "{Z_SCORE_MEAN}" = :mean, "{Z_SCORE_STD}" = :std, "{Z_SCORE}" = :z, "{Z_SCORE_RECENT_TXNS}" = :txns
                """),
                {"id": device_id, "mean": round(mean_val, 2), "std": round(std_val, 2), "z": round(z_score, 2), "txns": json.dumps(recent_txns)}
            )
        logger.debug(f"Database updated successfully for device {device_id}")
    except Exception as e:
        logger.error(f"Failed to update device statistics in database for device {device_id}")
        logger.debug(f"Database update error: {str(e)}")
        raise
    
    return mean_val, std_val

def get_device_schedule(device_id, weekday):
    """Get device operating schedule for the given weekday"""
    logger.debug(f"Fetching schedule for device {device_id} on {weekday}")
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text(f"""
                    SELECT {DEVICE_DATA_EST_OPENING}, {DEVICE_DATA_EST_CLOSING}
                    FROM {DEVICE_DATA_TABLE}
                    WHERE {DEVICE_DATA_DEVICE_ID} = :id
                    AND LOWER({DEVICE_DATA_DAY}) = LOWER(:day)
                """),
                {"id": device_id, "day": weekday}
            ).fetchone()

            if row and row[0] is not None and row[1] is not None:
                open_hour, close_hour = row[0].hour, row[1].hour
                logger.debug(f"Schedule found for device {device_id} on {weekday} - Open: {open_hour}:00, Close: {close_hour}:00")
                return open_hour, close_hour, row[0], row[1]
            else:
                logger.debug(f"No schedule found for device {device_id} on {weekday}, will use defaults")
                return None, None, None, None
    except Exception as e:
        logger.error(f"Could not retrieve business hours for device {device_id}")
        logger.debug(f"Error getting schedule for device {device_id}: {str(e)}")
        return None, None, None, None

# =================== API ===================
@app.route("/process_transaction", methods=["POST"])
def process_transaction():
    logger.info("New transaction received for processing")
    logger.debug("NEW TRANSACTION REQUEST RECEIVED")
    
    try:
        data = request.get_json()
        logger.debug(f"Request payload: {json.dumps(data, indent=2)}")
        
        device_id = int(data["deviceId"])
        txn_id = data["actionId"]
        txn_amt = float(data["amount"])
        print(f"\n--- New Transaction ---")
        print(f"  Transaction ID : {txn_id}")
        print(f"  Device ID      : {device_id}")
        print(f"  Amount         : ₹{txn_amt}")
        
        logger.info(f"Processing Transaction ID: {txn_id} | Device: {device_id} | Amount: ₹{txn_amt}")
        logger.debug(f"Processing transaction - Device: {device_id}, TxnID: {txn_id}, Amount: {txn_amt}")

        # ===== STEP 1: Timestamp Processing (Requirement 1) =====
        logger.debug("STEP 1: Processing timestamp")
        txn_ts_utc, txn_ts_ist, weekday = preprocess_timestamp(data["createdAt"])
        txn_hour = txn_ts_utc.hour
        print(f"  Time (UTC)     : {txn_ts_utc.strftime('%Y-%m-%d %I:%M %p')}")
        print(f"  Weekday        : {weekday}")
        logger.info(f"Transaction time: {txn_ts_ist.strftime('%Y-%m-%d %I:%M %p')} IST ({weekday})")
        logger.debug(f"Timestamp processed - Hour: {txn_hour}, Weekday: {weekday}")

        # ===== STEP 2: Get Historical Count (For is_supported flag - Requirement 2) =====
        logger.debug("STEP 2: Getting historical transaction count")
        historical_count = get_historical_txn_count(device_id)
        logger.debug(f"Historical count: {historical_count}")
        
        # ===== STEP 3: Get Device Schedule =====
        logger.debug("STEP 3: Fetching device schedule")
        open_hour, close_hour, open_time, close_time = get_device_schedule(device_id, weekday)
        
        # ===== STEP 4: Calculate Statistics (WITHOUT updating database) =====
        logger.debug("STEP 4: Calculating baseline statistics (historical data only)")
        mean_val, std_val, z_score, recent_txns = calculate_device_stats(device_id, txn_amt)

        # ===== STEP 5: Calculate Confidence =====
        logger.debug("STEP 5: Calculating anomaly confidence")
        conf_data = calculate_confidence(z_score, txn_amt, mean_val, txn_hour, open_hour, close_hour, std_val, Z_THRESHOLD)
        
        # ===== STEP 6: Determine Final Anomaly Status (Requirement 3 - FIX) =====
        logger.debug("STEP 6: Determining anomaly status")
        # Check BOTH z-score threshold AND confidence threshold
        is_anomaly_by_zscore = abs(z_score) > Z_THRESHOLD if len(recent_txns) >= 2 else False
        is_anomaly_by_confidence = conf_data["anomaly_confidence"] >= 70.0
        is_anomaly = is_anomaly_by_zscore or is_anomaly_by_confidence

        
        logger.debug(f"Anomaly checks - By Z-score: {is_anomaly_by_zscore} (|{z_score:.2f}| > {Z_THRESHOLD}), By Confidence: {is_anomaly_by_confidence} ({conf_data['anomaly_confidence']}% >= 70%)")
        logger.debug(f"Final anomaly decision: {is_anomaly}")
        
        label = "Yes" if is_anomaly else "No"
        confidence = conf_data["anomaly_confidence"] if label == "Yes" else conf_data["normal_confidence"]

        # ===== STEP 7: NOW Update Database Based on Anomaly Decision =====
        logger.debug("STEP 7: Updating database with anomaly decision")
        final_mean, final_std = finalize_device_stats(device_id, txn_amt, recent_txns, is_anomaly)
        # ===== STEP 8: Set is_supported Flag (Requirement 2) =====
        logger.debug("STEP 8: Setting is_supported flag")
        is_supported = historical_count >= 100
        if is_supported is False:
            label = "Unsupported"

            logger.warning(f"Device {device_id} does not have enough transaction history for reliable analysis (Count: {historical_count}/100 required)")
            logger.debug(f"Device {device_id} marked as Unsupported - historical count {historical_count} < 100")
        else:
            logger.debug(f"Device {device_id} is supported - historical count: {historical_count}")
        print(f"  Z-Score        : {round(z_score, 2)}")
        print(f"  Confidence     : {confidence}%")
        print(f"  Supported      : {is_supported}")
        print(f"  Result         : {label}")
        # ===== STEP 9: Store Results in Database =====
        logger.debug("STEP 9: Storing results in anomaly results table")
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(f"""
                        INSERT INTO {ANOMALY_RESULTS_TABLE}
                        ({ANOMALY_DEVICE_ID}, {ANOMALY_TXN_ID}, {ANOMALY_TS}, {ANOMALY_AMT},
                        {ANOMALY_Z}, {ANOMALY_CONF}, {ANOMALY_LABEL},
                        {ANOMALY_OPEN}, {ANOMALY_CLOSE}, {ANOMALY_CURRENT_THRESHOLD})
                        VALUES (:id, :tid, :ts, :amt, :z, :conf, :label,
                                :open, :close, :thresholdc)
                    """),
                    {
                        "id": device_id,
                        "tid": txn_id,
                        "ts": txn_ts_utc,
                        "amt": txn_amt,
                        "z": round(z_score, 2),
                        "conf": confidence,
                        "label": label,
                        "open": open_time,
                        "close": close_time,
                        "thresholdc": Z_THRESHOLD
                    }
                )

            logger.info(f"Analysis result saved to database successfully")
            logger.debug(f"Results stored successfully for transaction {txn_id}")
        except Exception as e:
            logger.error(f"Failed to save analysis results to database for transaction {txn_id}")
            logger.debug(f"Failed to store results for transaction {txn_id}: {str(e)}")
            raise

        # ===== STEP 10: Return Response (BOTH UTC AND IST) =====
        logger.debug("STEP 10: Preparing response")
        response_data = {
            "deviceId": device_id,
            "transaction_id": txn_id,
            "txn_dt_utc": txn_ts_utc.isoformat(),
            "txn_dt_ist": txn_ts_ist.isoformat(),
            "weekday": weekday,
            "txnAmt": txn_amt,
            "z_score": round(z_score, 2),
            "anomaly_label": label,
            "confidence": confidence,
            "new_mean": round(final_mean, 2),
            "new_std": round(final_std, 2),
            "recent_txns_count": len(recent_txns) if not is_anomaly else len(recent_txns) + 1,
            "is_supported": is_supported
        }
        
        # Client-facing summary
        if label == "Yes":
            logger.info(f" ANOMALY DETECTED | Transaction ID: {txn_id} | Confidence: {confidence}% | Amount: {txn_amt}")
        elif label == "Unsupported":
            logger.info(f"  Transaction processed but device needs more history for reliable analysis | Transaction ID: {txn_id}")
        else:
            logger.info(f" Normal transaction processed successfully | Transaction ID: {txn_id} | Confidence: {confidence}%")
        
        logger.debug(f"Transaction processed successfully - Label: {label}, Confidence: {confidence}%")
        logger.debug(f"Response data: {json.dumps(response_data, indent=2)}")

        logger.debug("TRANSACTION PROCESSING COMPLETED SUCCESSFULLY")
        
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f" Transaction processing failed - Please contact technical support")
        logger.debug("TRANSACTION PROCESSING FAILED")
        logger.error(f"Error occurred: {type(e).__name__} - {str(e)}")
        logger.debug(f"Error type: {type(e).__name__}")
        logger.debug(f"Error message: {str(e)}")
        
        import traceback
        error_traceback = traceback.format_exc()
        logger.debug(f"Full traceback:\n{error_traceback}")
        
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Anomaly Detection Service is starting on port 9198")
    logger.debug("Starting Flask application server on host=0.0.0.0, port=9198")
    app.run(host="0.0.0.0", port=9198, debug=True)