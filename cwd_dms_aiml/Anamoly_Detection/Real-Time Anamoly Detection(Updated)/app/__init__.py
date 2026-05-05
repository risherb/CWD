from flask import Flask, request, jsonify
from config.settings import ConfigManager
from utils.logger import get_logger
from repositories.db_repository import SQLAlchemyRepository
from services.anomaly_service import AnomalyService


def create_app() -> Flask:
    cfg = ConfigManager()
    db = cfg.database()
    tables = cfg.tables()
    cols = cfg.columns()
    proc = cfg.processing()

    logger = get_logger("anomaly_api", proc.log_file_path)

    repo = SQLAlchemyRepository(db.postgres_uri, tables, cols, logger)
    z_threshold = repo.get_threshold()

    service = AnomalyService(
        repository=repo,
        logger=logger,
        rolling_window=proc.rolling_window,
        z_threshold=z_threshold,
    )

    app = Flask(__name__)

    @app.route("/process_transaction", methods=["POST"])
    def process_transaction():
        try:
            payload = request.get_json()
            result = service.process_transaction(payload)
            return jsonify(result), 200
        except Exception as e:
            logger.error("/process_transaction failed: %s", e, exc_info=True)
            return jsonify({"error": str(e)}), 500

    return app
