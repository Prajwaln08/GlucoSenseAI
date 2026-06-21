"""
Custom Prometheus metrics for GlucoSense AI.

HTTP metrics are provided by prometheus-fastapi-instrumentator (wired in api/main.py
and exposed at /metrics). These are the domain-specific counters/histograms that give
visibility into ingestion volume by source and Junction API health.
"""

from prometheus_client import Counter, Histogram

# CGM readings written, labelled by source (junction | xdrip) — shows the source mix
# and makes a failover (xDRIP taking over) observable.
cgm_readings_ingested = Counter(
    "glucosense_cgm_readings_ingested_total",
    "CGM readings written to the DB by the unified ingest layer",
    ["source"],
)

# Daily activity rows upserted, labelled by provider (google_fit | junction:<slug>).
activity_days_upserted = Counter(
    "glucosense_activity_days_upserted_total",
    "New daily activity rows written by the unified ingest layer",
    ["provider"],
)

# Junction API request latency — surfaces primary-source degradation.
junction_request_seconds = Histogram(
    "glucosense_junction_request_seconds",
    "Junction API request latency (seconds)",
    ["method"],
)
