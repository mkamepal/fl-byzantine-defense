"""Small, immediate RSS samples; never retain client models or datasets."""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import uuid


_active_trace = ContextVar("worker_memory_trace", default=None)
_worker_id = uuid.uuid4().hex
_logger = logging.getLogger("flwr")


def current_rss_bytes():
    """Linux resident RAM now, not the process's lifetime high-water mark."""
    with open("/proc/self/statm", encoding="ascii") as source:
        resident_pages = int(source.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


class WorkerMemoryTrace:
    def __init__(self, phase, msg=None, context=None):
        config = msg.content.get("config", {}) if msg is not None else {}
        node_config = context.node_config if context is not None else {}
        partition = node_config.get("partition-id")
        self.metadata = {
            "request_id": uuid.uuid4().hex,
            "pid": os.getpid(), "hostname": socket.gethostname(),
            "client_id": f"client_{partition}" if partition is not None else None,
            "round": config.get("server-round"), "phase": phase,
        }
        directory = config.get("worker-memory-dir")
        self.path = (Path(directory) / f"worker_{os.getpid()}_{_worker_id}.jsonl"
                     if directory else None)
        self.start_rss = None
        self.previous_rss = None
        self.read_warning = False

    def sample(self, stage, outcome="running"):
        try:
            rss = current_rss_bytes()
        except (OSError, ValueError, IndexError) as error:
            rss = None
            if not self.read_warning:
                _logger.warning("[worker-memory] Current RSS unavailable: %s", error)
                self.read_warning = True
        if stage == "request_start":
            self.start_rss = rss
        delta = rss - self.previous_rss if rss is not None and self.previous_rss is not None else None
        total = rss - self.start_rss if rss is not None and self.start_rss is not None else None
        record = {**self.metadata, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "stage": stage, "outcome": outcome, "rss_bytes": rss,
                  "delta_previous_bytes": delta, "delta_request_bytes": total}
        self.previous_rss = rss

        def mib(value):
            return f"{value / 2**20:.2f}" if value is not None else "unavailable"

        _logger.info(
            "[worker-memory] pid=%s client=%s round=%s phase=%s stage=%s "
            "rss_mib=%s delta_previous_mib=%s delta_request_mib=%s outcome=%s request=%s",
            record["pid"], record["client_id"], record["round"], record["phase"], stage,
            mib(rss), mib(delta), mib(total), outcome, record["request_id"],
        )
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Close each sample so records survive a subsequent worker OOM.
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record) + "\n")
            except OSError as error:
                _logger.warning("[worker-memory] Cannot save %s: %s", self.path, error)
                self.path = None  # Keep terminal diagnostics; do not fail training.


def record_client_memory(stage):
    """Record a stage within the active request without passing model references."""
    trace = _active_trace.get()
    if trace is not None:
        trace.sample(stage)
