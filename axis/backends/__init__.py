"""Inference backends for the Executor.

The Executor is backend-agnostic; it delegates the actual token production
to an InferenceBackend. Priority 1 ships with a deterministic stub so the
control flow can be tested without invoking a real model.
"""

from .base import BackendResult, InferenceBackend
from .stub import StubBackend

__all__ = ["BackendResult", "InferenceBackend", "StubBackend"]
