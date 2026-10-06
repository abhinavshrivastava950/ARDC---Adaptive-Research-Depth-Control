"""Document worker interface (Sec 3.3 ownership boundary).

The worker owns exact research execution: queries, tools, reading order,
drafting. The controller owns permission + coarse intent. If the
controller generated the full plan, the project would become a second
research agent -- that is explicitly out of scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Sequence


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
    # The controller's reason the last checkpoint did not pass, set by the
    # runner before each lease. Optional for workers to use.
    controller_note: str = ""
    # Action classes the current lease permits (empty = unrestricted). A worker
    # stores the ``allowed_classes`` argument here at the start of ``act()`` and
    # reports the class it actually used in ``WorkerResult.detail["action_class"]``
    # (SEARCH | READ | VERIFY | ANSWER). Workers enforce at their own tool
    # boundary; ``WorkerAdapter`` independently verifies and records (Sec 7.2).
    allowed_classes: Sequence[str] = ()

    @abstractmethod
    def act(self, intent: str, target_gaps: List[str],
            allowed_classes: List[str], draft: str) -> WorkerResult:
        """Execute ONE substantive action inside the current lease."""
