"""Query Router Module.

Classifies incoming student queries into:
- THEORY: Handled by RAG pipeline
- NUMERICAL: Handled by Deterministic Deadlock Solver
- GRAPH: Handled by Graph / cycle analysis and visualizer
- LAB: Handled by POSIX pthread code / synchronization examples
"""

import re
from enum import Enum
from pydantic import BaseModel


class QueryCategory(str, Enum):
    THEORY = "theory"
    NUMERICAL = "numerical"
    GRAPH = "graph"
    LAB = "lab"


class RoutingDecision(BaseModel):
    category: QueryCategory
    confidence: float
    reason: str


class QueryRouter:
    """Classifies queries into appropriate backend handling pipelines."""

    def __init__(self):
        self.numerical_patterns = [
            r"\bbanker('?s)?\b",
            r"\bsafe sequence\b",
            r"\bsafety algorithm\b",
            r"\bresource request\b",
            r"\ballocation matrix\b",
            r"\bmax matrix\b",
            r"\bavailable vector\b",
            r"\bneed matrix\b",
            r"\bwork vector\b",
            r"\bis the state safe\b",
            r"\bdetermine whether\b.*\bsafe\b",
        ]
        self.graph_patterns = [
            r"\bresource[- ]allocation graph\b",
            r"\bwait[- ]for graph\b",
            r"\bwfg\b",
            r"\bfind (the )?cycle\b",
            r"\bdetect cycle\b",
            r"\bcycle in (the )?graph\b",
        ]
        self.lab_patterns = [
            r"\bpthread\b",
            r"\bmutex\b",
            r"\bsemaphore\b",
            r"\bdining philosophers\b",
            r"\bc program\b",
            r"\bc\+\+\b",
            r"\bdeadlock code\b",
            r"\bdeadlock simulation\b",
        ]

    def route(self, query: str) -> RoutingDecision:
        """Determines the processing category for a query."""
        q_lower = query.lower().strip()

        # Check numerical patterns
        for pattern in self.numerical_patterns:
            if re.search(pattern, q_lower):
                return RoutingDecision(
                    category=QueryCategory.NUMERICAL,
                    confidence=0.9,
                    reason=f"Matched numerical pattern: {pattern}"
                )

        # Check graph patterns
        for pattern in self.graph_patterns:
            if re.search(pattern, q_lower):
                return RoutingDecision(
                    category=QueryCategory.GRAPH,
                    confidence=0.85,
                    reason=f"Matched graph pattern: {pattern}"
                )

        # Check lab patterns
        for pattern in self.lab_patterns:
            if re.search(pattern, q_lower):
                return RoutingDecision(
                    category=QueryCategory.LAB,
                    confidence=0.85,
                    reason=f"Matched lab/concurrency pattern: {pattern}"
                )

        # Default to theory (RAG)
        return RoutingDecision(
            category=QueryCategory.THEORY,
            confidence=0.8,
            reason="Query classified as theoretical/conceptual inquiry."
        )
