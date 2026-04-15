"""Memory Manager subsystem — spec §1.6.

Four persistent stores + a forgetting engine:

    WorkingMemory     (§1.6.1) — lives in axis.executor; per-task, volatile
    EpisodicMemory    (§1.6.2) — one record per completed task
    SemanticMemory    (§1.6.3) — distilled stable knowledge
    ProceduralMemory  (§1.6.4) — reusable execution patterns
    ForgettingEngine  (§1.6.5) — decay, pruning, contradiction resolution

The MemoryManager class is the single entry point that owns these stores
and enforces the §6 State Access Matrix rules (who can write what).
"""

from .episodic import Episode, EpisodicMemory
from .forgetting import ForgettingEngine
from .manager import MemoryManager
from .procedural import Procedure, ProceduralMemory
from .semantic import SemanticEntity, SemanticMemory

__all__ = [
    "Episode",
    "EpisodicMemory",
    "ForgettingEngine",
    "MemoryManager",
    "Procedure",
    "ProceduralMemory",
    "SemanticEntity",
    "SemanticMemory",
]
