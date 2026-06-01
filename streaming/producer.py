"""
Kafka producer for sample Home Credit application payloads.
"""

from __future__ import annotations

import json
import logging
import random
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

TOPIC = "loan_applications"
BOOTSTRAP_SERVER = "localhost:9092"
PUBLISH_INTERVAL = 1.5


def _sample_application() -> dict:
    return {
        "application_id": f"APP-{random.randint(100000, 999999)}",
        "NAME_CONTRACT_TYPE": random.choice(["Cash loans", "Revolving loans"]),
        "CODE_GENDER": random.choice(["F", "M"]),
        "FLAG_OWN_CAR": random.choice(["Y", "N"]),
        "FLAG_OWN_REALTY": random.choice(["Y", "N"]),
        "CNT_CHILDREN": random.randint(0, 3),
        "AMT_INCOME_TOTAL": round(random.uniform(45_000, 300_000), 2),
        "AMT_CREDIT": round(random.uniform(100_000, 1_200_000), 2),
        "AMT_ANNUITY": round(random.uniform(8_000, 60_000), 2),
        "AMT_GOODS_PRICE": round(random.uniform(90_000, 1_100_000), 2),
        "NAME_INCOME_TYPE": random.choice(["Working", "Commercial associate", "State servant", "Pensioner"]),
        "NAME_EDUCATION_TYPE": random.choice(["Secondary / secondary special", "Higher education"]),
        "NAME_FAMILY_STATUS": random.choice(["Married", "Single / not married", "Civil marriage"]),
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -random.randint(21 * 365, 70 * 365),
        "DAYS_EMPLOYED": -random.randint(30, 30 * 365),
        "EXT_SOURCE_2": round(random.random(), 6),
        "EXT_SOURCE_3": round(random.random(), 6),
    }


def run():
    try:
        from kafka import KafkaProducer
    except ImportError:
        logger.error("kafka-python not installed. Run: pip install kafka-python")
        return

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVER,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5,
        compression_type="gzip",
    )

    logger.info("Producer started -> topic: %s", TOPIC)
    sent = 0
    try:
        while True:
            payload = _sample_application()
            producer.send(TOPIC, value=payload)
            sent += 1
            logger.info(
                "[%s] Published %s credit=%.0f income=%.0f",
                sent,
                payload["application_id"],
                payload["AMT_CREDIT"],
                payload["AMT_INCOME_TOTAL"],
            )
            time.sleep(PUBLISH_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Producer stopped. Total published: %s", sent)
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run()
