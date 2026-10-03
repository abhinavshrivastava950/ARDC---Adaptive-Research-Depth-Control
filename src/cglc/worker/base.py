"""Document worker interface (Sec 3.3 ownership boundary).

The worker owns exact research execution: queries, tools, reading order,
drafting. The controller owns permission + coarse intent. If the
controller generated the full plan, the project would become a second
research agent -- that is explicitly out of scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Observation:
    source_id: str
    span_id: str
    text: str
    chunk_id: str
    cost: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkerResult:
    observations: List[Observation]
    draft: str = ""
    propose_final: bool = False
    blocker: str = ""
    contradiction: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


class DocumentWorker(ABC):
    @abstractmethod
    def act(self, intent: str, target_gaps: List[str],
            allowed_classes: List[str], draft: str) -> WorkerResult:
        """Execute ONE substantive action inside the current lease."""
