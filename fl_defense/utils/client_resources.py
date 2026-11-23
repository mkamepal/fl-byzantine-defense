"""Release per-request TensorFlow state in reused simulation workers."""

from functools import wraps

import keras

from .worker_memory import WorkerMemoryTrace, _active_trace


def release_client_resources(function):
    """Clean up after the handler frame (including model/history) has returned."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        msg = args[0] if args else kwargs.get("msg")
        context = args[1] if len(args) > 1 else kwargs.get("context")
        trace = WorkerMemoryTrace(function.__name__, msg, context)
        token = _active_trace.set(trace)
        outcome = "error"
        try:
            trace.sample("request_start")
            result = function(*args, **kwargs)
            outcome = "success"
            return result
        finally:
            try:
                trace.sample("before_cleanup", outcome)
                # At normal return, only the serialized Flower reply remains alive.
                # Collect model/History/callback cycles before the next client call.
                keras.backend.clear_session(free_memory=True)
            except BaseException:
                outcome = "cleanup_error"
                raise
            finally:
                try:
                    trace.sample("after_cleanup", outcome)
                finally:
                    _active_trace.reset(token)
    return wrapped
