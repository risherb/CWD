import json
import numpy as np
import pandas as pd
import pytz
from typing import Any, Dict
from core.interfaces import AnomalyService as IAnomalyService, Repository, Logger


class AnomalyService(IAnomalyService):
    def __init__(self, repository: Repository, logger: Logger, rolling_window: int, z_threshold: int) -> None:
        self.repository = repository
        self.logger = logger
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold

    def _calculate_confidence(self, z_score, txn_amt, mean_val, txn_hour, open_hour, close_hour, std_val, z_threshold):
        threshold_amt = mean_val + z_threshold * std_val if std_val is not None else mean_val * 2
        if abs(z_score) > 1000 * z_threshold or txn_amt > 100 * threshold_amt:
            return {"anomaly_confidence": 100.0, "normal_confidence": 0.0}

        if open_hour is not None and close_hour is not None:
            day_hrs = 24 - (close_hour - open_hour)
            day_hrs_1 = day_hrs / 2
            day_hrs_2 = day_hrs_1 / 4
            dynamic_open_buffer = max(round(day_hrs_2), 1)
            dynamic_close_buffer = max(round(day_hrs_2), 1)
        else:
            dynamic_open_buffer = dynamic_close_buffer = 0

        if np.isnan(z_score):
            return {"anomaly_confidence": 50.0, "normal_confidence": 50.0}

        amount_anomaly = min(abs(z_score) / z_threshold * 100, 100)

        time_anomaly = 0.0
        if open_hour is not None and close_hour is not None:
            if open_hour <= txn_hour <= close_hour:
                time_anomaly = 0.0
            elif (open_hour - dynamic_open_buffer) <= txn_hour < open_hour or \
                 close_hour < txn_hour <= (close_hour + dynamic_close_buffer):
                time_anomaly = 30.0
            elif (open_hour - 2 * dynamic_open_buffer) <= txn_hour < (open_hour - dynamic_open_buffer) or \
                 (close_hour + dynamic_close_buffer) < txn_hour <= (close_hour + 2 * dynamic_close_buffer):
                time_anomaly = 50.0
            elif (open_hour - 3 * dynamic_open_buffer) <= txn_hour < (open_hour - 2 * dynamic_open_buffer) or \
                 (close_hour + 2 * dynamic_close_buffer) < txn_hour <= (close_hour + 3 * dynamic_close_buffer):
                time_anomaly = 70.0
            else:
                time_anomaly = 100.0

        anomaly_conf = 0.7 * amount_anomaly + 0.3 * time_anomaly
        anomaly_conf = min(max(anomaly_conf, 0), 100)
        normal_conf = 100 - anomaly_conf

        return {
            "anomaly_confidence": round(anomaly_conf, 2),
            "normal_confidence": round(normal_conf, 2)
        }

    def process_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        device_id = int(payload["deviceId"])
        txn_id = payload["actionId"]
        txn_amt = float(payload["amount"])
        created_at = payload["createdAt"]

        ts = pd.to_datetime(created_at, utc=True)
        ist = pytz.timezone("Asia/Kolkata")
        ts_ist = ts.tz_convert(ist).tz_localize(None)
        weekday = ts_ist.strftime("%A")
        txn_hour = ts_ist.hour

        open_hour, close_hour, open_time, close_time = self.repository.get_device_schedule(device_id, weekday)

        recent_txns = self.repository.get_recent_txns(device_id)

        if len(recent_txns) >= 2:
            mean_val = float(np.mean(recent_txns))
            std_val = float(np.std(recent_txns, ddof=1))
            z_score = (txn_amt - mean_val) / std_val if std_val > 0 else 0.0
        elif len(recent_txns) == 1:
            mean_val, std_val = float(recent_txns[0]), 0.0
            z_score = 0.0
        else:
            mean_val, std_val, z_score = txn_amt, 0.0, 0.0

        # Cap z-score to prevent database overflow and ensure reasonable values
        capped_z_score = max(min(z_score, 999999.99), -999999.99)
        
        is_anomaly = abs(capped_z_score) > self.z_threshold

        if not is_anomaly:
            recent_txns.append(txn_amt)
            if len(recent_txns) > self.rolling_window:
                recent_txns.pop(0)

        self.repository.upsert_device_stats(
            device_id=device_id,
            mean_val=mean_val,
            std_val=std_val,
            z_score=capped_z_score,
            recent_txns_json=json.dumps(recent_txns),
        )

        conf_dict = self._calculate_confidence(capped_z_score, txn_amt, mean_val, txn_hour, open_hour, close_hour, std_val, self.z_threshold)
        confidence = conf_dict["anomaly_confidence"] if is_anomaly else conf_dict["normal_confidence"]
        label = "Yes" if is_anomaly else "No"

        self.repository.insert_anomaly_log({
            "id": device_id,
            "tid": txn_id,
            "ts": ts_ist,
            "amt": txn_amt,
            "z": round(capped_z_score, 2),
            "conf": confidence,
            "label": label,
            "open": open_time,
            "close": close_time,
        })

        return {
            "deviceId": device_id,
            "transaction_id": txn_id,
            "txn_dt_ist": ts_ist.isoformat(),
            "weekday": weekday,
            "txnAmt": txn_amt,
            "z_score": round(capped_z_score, 2),
            "anomaly_label": label,
            "confidence": confidence,
            "new_mean": round(mean_val, 2),
            "new_std": round(std_val, 2),
            "recent_txns_count": len(recent_txns),
        }
