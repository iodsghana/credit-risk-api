"""
Real-time Home Credit scoring Kafka consumer.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

INPUT_TOPIC = "loan_applications"
OUTPUT_TOPIC = "loan_scores"
GROUP_ID = "credit-risk-scoring-group"
BOOTSTRAP = "localhost:9092"
HIGH_RISK_THRESHOLD = 0.50


def run():
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        logger.error("kafka-python not installed. Run: pip install kafka-python")
        return

    from predict import predict

    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks=1,
        compression_type="gzip",
    )

    logger.info("Consumer listening on: %s", INPUT_TOPIC)
    processed = 0

    try:
        for message in consumer:
            application = message.value
            application_id = application.get("application_id", "UNKNOWN")

            try:
                result = predict(application)
                probability = result["default_probability"]
                output = {
                    "application_id": application_id,
                    "default_probability": probability,
                    "risk_band": result["risk_band"],
                    "model_version": result["model_version"],
                    "decision": "DECLINE" if probability >= HIGH_RISK_THRESHOLD else "APPROVE",
                    "kafka_offset": message.offset,
                    "kafka_partition": message.partition,
                }

                producer.send(OUTPUT_TOPIC, value=output)
                processed += 1
                flag = "HIGH RISK" if probability >= HIGH_RISK_THRESHOLD else "LOW RISK"
                logger.info(
                    "[%s] %s | P(default)=%.3f | %s",
                    processed,
                    application_id,
                    probability,
                    flag,
                )
            except Exception as exc:
                logger.error("Scoring failed for %s: %s", application_id, exc)

    except KeyboardInterrupt:
        logger.info("Consumer stopped. Total processed: %s", processed)
    finally:
        consumer.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run()
