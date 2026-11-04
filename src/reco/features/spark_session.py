"""
PySpark Session Configuration and Factory.
Configures driver memory, local execution, and ensures the correct Python worker runtime.
"""

import os
import sys

from pyspark.sql import SparkSession

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def get_spark_session(
    app_name: str = "RecoEngineFeatures",
    master: str = "local[*]",
    driver_memory: str = "4g",
) -> SparkSession:
    """
    Create or retrieve an active SparkSession with optimized local settings.

    Args:
        app_name: Name of the Spark application.
        master: Spark master URL (default: 'local[*]').
        driver_memory: Maximum heap size for the driver (default: '4g').

    Returns:
        Configured SparkSession instance.
    """
    # Guarantee the PySpark worker uses the exact virtual environment Python
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession initialized: master=%s, memory=%s", master, driver_memory)
    return spark
