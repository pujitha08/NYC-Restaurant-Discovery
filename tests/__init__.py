"""Tests run without real waiting: a rate-limited summary call is retried at once (the retry count is unchanged).

tests/test_summary_backoff.py patches the delays itself to check them.
"""

from app import pipeline

pipeline.BACKOFF_SECONDS = (0.0,) * len(pipeline.BACKOFF_SECONDS)
