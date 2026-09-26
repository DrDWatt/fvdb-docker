"""Polling helper for background jobs (model loads, extractions, exports)."""
import time


def wait_for(fn, timeout=10.0, interval=0.05, message="timed out waiting for background job"):
    """Poll fn() until it returns a truthy value and return it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(message)
