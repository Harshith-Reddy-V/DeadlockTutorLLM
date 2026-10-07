"""Query Router Module.

Classifies incoming student queries into:
  - THEORY    : Conceptual/definitional questions → RAG pipeline
  - NUMERICAL : Algorithm problems with matrices/numbers → Deterministic Solver
  - GRAPH     : Wait-for graph / resource-allocation graph questions → Graph Solver
  - LAB       : Code / pthread / concurrency implementation questions → RAG + code context

Classification is fully deterministic (pattern-matching), so the router is
fast, testable, and does NOT require an LLM call.

Patterns are intentionally broad to minimise false negatives for student
queries that may not use textbook-exact terminology.
"""

import re
from enum import Enum
from typing import List, Tuple

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


# ---------------------------------------------------------------------------
# Pattern tables — (compiled_regex, category, confidence_score)
# ---------------------------------------------------------------------------

# Each entry: (raw_pattern, category, confidence)
_PATTERNS: List[Tuple[str, QueryCategory, float]] = [
    # ---- NUMERICAL (highest priority — explicit algorithm application) ----
    (r"\bbanker'?s?\s+algorithm\b", QueryCategory.NUMERICAL, 0.95),
    (r"\bsafe\s+sequence\b", QueryCategory.NUMERICAL, 0.95),
    (r"\bsafety\s+algorithm\b", QueryCategory.NUMERICAL, 0.92),
    (r"\bresource[\s\-]request\s+algorithm\b", QueryCategory.NUMERICAL, 0.92),
    (r"\ballocation\s+matrix\b", QueryCategory.NUMERICAL, 0.90),
    (r"\bmax\s+matrix\b", QueryCategory.NUMERICAL, 0.90),
    (r"\bneed\s+matrix\b", QueryCategory.NUMERICAL, 0.90),
    (r"\bavailable\s+vector\b", QueryCategory.NUMERICAL, 0.90),
    (r"\bwork\s+vector\b", QueryCategory.NUMERICAL, 0.88),
    (r"\bis\s+(the\s+)?state\s+safe\b", QueryCategory.NUMERICAL, 0.90),
    (r"\bdetermine\s+(if|whether)\b.*\bsafe\b", QueryCategory.NUMERICAL, 0.88),
    (r"\bcalculate\b.*\bneed\b", QueryCategory.NUMERICAL, 0.88),
    (r"\bapply\s+(the\s+)?banker", QueryCategory.NUMERICAL, 0.95),
    (r"\brun\s+(the\s+)?banker", QueryCategory.NUMERICAL, 0.95),
    (r"\bmulti[\s\-]instance\s+detection\b", QueryCategory.NUMERICAL, 0.88),
    (r"\bmatrix[\s\-]based\s+detection\b", QueryCategory.NUMERICAL, 0.88),
    (r"\bgiven\b.*\ballocation\b.*\bmax\b", QueryCategory.NUMERICAL, 0.90),
    (r"\bfinish\s+vector\b", QueryCategory.NUMERICAL, 0.88),
    (r"\bpretend\s+allocat", QueryCategory.NUMERICAL, 0.88),

    # ---- GRAPH (second priority) ----
    (r"\bresource[\s\-]allocation\s+graph\b", QueryCategory.GRAPH, 0.92),
    (r"\bwait[\s\-]for\s+graph\b", QueryCategory.GRAPH, 0.92),
    (r"\bwfg\b", QueryCategory.GRAPH, 0.90),
    (r"\bfind\s+(the\s+)?cycle\b", QueryCategory.GRAPH, 0.90),
    (r"\bdetect\s+(a\s+)?cycle\b", QueryCategory.GRAPH, 0.90),
    (r"\bcycle\s+in\s+(the\s+)?graph\b", QueryCategory.GRAPH, 0.88),
    (r"\bdraw\s+(the\s+)?(wait|rag|resource|graph)\b", QueryCategory.GRAPH, 0.88),
    (r"\brag\s+graph\b", QueryCategory.GRAPH, 0.88),
    (r"\bgraph\s+(representation|analysis)\b", QueryCategory.GRAPH, 0.85),
    (r"\bcircular\s+wait\s+in\s+(the\s+)?graph\b", QueryCategory.GRAPH, 0.88),
    (r"\bedge\s+in\s+(the\s+)?(wait|resource|graph)\b", QueryCategory.GRAPH, 0.85),
    (r"\bsingle[\s\-]instance.*\bdetect\b", QueryCategory.GRAPH, 0.88),
    (r"\bdetect.*\bsingle[\s\-]instance\b", QueryCategory.GRAPH, 0.88),

    # ---- LAB (code / implementation) ----
    (r"\bpthread\b", QueryCategory.LAB, 0.92),
    (r"\bmutex\b", QueryCategory.LAB, 0.88),
    (r"\bsemaphore\b", QueryCategory.LAB, 0.88),
    (r"\bdining\s+philosophers\b", QueryCategory.LAB, 0.92),
    (r"\bc\s+program\b", QueryCategory.LAB, 0.85),
    (r"\bc\+\+\b", QueryCategory.LAB, 0.85),
    (r"\bimplement\b.*\bdeadlock\b", QueryCategory.LAB, 0.88),
    (r"\bdeadlock\s+code\b", QueryCategory.LAB, 0.90),
    (r"\bdeadlock\s+simulation\b", QueryCategory.LAB, 0.88),
    (r"\bmonitor\b.*\bdeadlock\b", QueryCategory.LAB, 0.85),
    (r"\blocksemaphore\b", QueryCategory.LAB, 0.88),
    (r"\bspin\s*lock\b", QueryCategory.LAB, 0.85),
    (r"\bfork\b.*\bphilosopher\b", QueryCategory.LAB, 0.90),
    (r"\bcritical\s+section\b.*\bcode\b", QueryCategory.LAB, 0.85),
    (r"\bwrite\s+(a\s+)?(program|code)\b", QueryCategory.LAB, 0.82),
]

# Pre-compile all patterns for performance
_COMPILED: List[Tuple[re.Pattern, QueryCategory, float]] = [
    (re.compile(pattern, re.IGNORECASE), category, confidence)
    for pattern, category, confidence in _PATTERNS
]


class QueryRouter:
    """Classifies queries into appropriate backend handling pipelines.

    Priority order: NUMERICAL > GRAPH > LAB > THEORY (default fallback).
    """

    def route(self, query: str) -> RoutingDecision:
        """Determines the processing category for a student query.

        Args:
            query: Raw student query string.

        Returns:
            RoutingDecision with category, confidence score, and matching reason.
        """
        q_stripped = (query or "").strip()

        if not q_stripped:
            return RoutingDecision(
                category=QueryCategory.THEORY,
                confidence=0.5,
                reason="Empty query — defaulting to THEORY.",
            )

        # Evaluate all patterns; collect category hits with max confidence per category
        category_scores: dict[QueryCategory, Tuple[float, str]] = {}
        for pattern, category, confidence in _COMPILED:
            if pattern.search(q_stripped):
                existing_conf, _ = category_scores.get(category, (0.0, ""))
                if confidence > existing_conf:
                    category_scores[category] = (confidence, f"Matched: /{pattern.pattern}/")

        # Choose winner by priority: NUMERICAL > GRAPH > LAB > THEORY
        for priority_cat in (QueryCategory.NUMERICAL, QueryCategory.GRAPH, QueryCategory.LAB):
            if priority_cat in category_scores:
                conf, reason = category_scores[priority_cat]
                return RoutingDecision(category=priority_cat, confidence=conf, reason=reason)

        # Default: THEORY
        return RoutingDecision(
            category=QueryCategory.THEORY,
            confidence=0.80,
            reason="No numerical, graph, or lab patterns matched — classified as theoretical/conceptual.",
        )
