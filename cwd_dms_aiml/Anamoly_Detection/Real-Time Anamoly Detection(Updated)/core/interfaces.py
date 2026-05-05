from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class Logger(ABC):
    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str, exc_info: bool = False) -> None:
        pass

    @abstractmethod
    def debug(self, message: str) -> None:
        pass


class ConfigProvider(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        pass


class Repository(ABC):
    @abstractmethod
    def get_threshold(self) -> int:
        pass

    @abstractmethod
    def get_device_schedule(self, device_id: int, weekday: str) -> Tuple[Optional[int], Optional[int], Optional[Any], Optional[Any]]:
        pass

    @abstractmethod
    def get_recent_txns(self, device_id: int) -> List[float]:
        pass

    @abstractmethod
    def upsert_device_stats(self, device_id: int, mean_val: float, std_val: float, z_score: float, recent_txns_json: str) -> None:
        pass

    @abstractmethod
    def insert_anomaly_log(self, payload: Dict[str, Any]) -> None:
        pass


class AnomalyService(ABC):
    @abstractmethod
    def process_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pass
