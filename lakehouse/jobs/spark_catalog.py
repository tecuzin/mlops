from __future__ import annotations

import logging
import os

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def catalog_name() -> str:
    return _env("LAKEHOUSE_CATALOG_NAME", "nessie")


def catalog_ref() -> str:
    return _env("LAKEHOUSE_CATALOG_REF", "main")


def catalog_namespace() -> str:
    return _env("LAKEHOUSE_CATALOG_NAMESPACE", "gold")


def warehouse_uri() -> str:
    bucket = _env("LAKEHOUSE_S3_BUCKET", "lakehouse")
    return _env("LAKEHOUSE_WAREHOUSE", f"s3a://{bucket}/warehouse")


def aws_region() -> str:
    return os.getenv("AWS_REGION", "").strip() or _env("AWS_DEFAULT_REGION", "us-east-1")


def build_spark_session(app_name: str) -> SparkSession:
    catalog = catalog_name()
    region = aws_region()
    s3_endpoint = _env("LAKEHOUSE_S3_ENDPOINT", "http://minio:9000")
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config(f"spark.sql.catalog.{catalog}.uri", _env("LAKEHOUSE_CATALOG_URI", "http://nessie:19120/api/v1"))
        .config(f"spark.sql.catalog.{catalog}.ref", catalog_ref())
        .config(f"spark.sql.catalog.{catalog}.warehouse", warehouse_uri())
        .config(f"spark.sql.catalog.{catalog}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"spark.sql.catalog.{catalog}.s3.endpoint", s3_endpoint)
        .config(f"spark.sql.catalog.{catalog}.s3.path-style-access", "true")
        .config(f"spark.sql.catalog.{catalog}.s3.region", region)
        .config(f"spark.sql.catalog.{catalog}.client.region", region)
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.access.key", _env("AWS_ACCESS_KEY_ID", "minio"))
        .config("spark.hadoop.fs.s3a.secret.key", _env("AWS_SECRET_ACCESS_KEY", "minio123"))
        .config("spark.hadoop.fs.s3a.endpoint.region", region)
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )


def publish_catalog_table(df: DataFrame, table: str) -> None:
    """
    Publish a DataFrame into the configured Nessie catalog table.

    Falls back gracefully if runtime jars are missing or catalog calls fail.
    """
    catalog = catalog_name()
    namespace = catalog_namespace()
    fq_namespace = f"{catalog}.{namespace}"
    fq_table = f"{fq_namespace}.{table}"
    try:
        # Nessie rejects table commits when the namespace does not exist yet.
        df.sparkSession.sql(f"CREATE NAMESPACE IF NOT EXISTS {fq_namespace}")
        df.writeTo(fq_table).using("iceberg").createOrReplace()
    except Exception as exc:
        # Keep medallion jobs operable even when catalog plugins are unavailable,
        # but log the root cause so it does not fail silently.
        logger.warning("Catalog publish skipped for %s: %s", fq_table, exc, exc_info=True)
