from dataclasses import dataclass
from typing import Optional


@dataclass
class Program:
    id: Optional[int]
    island_id: int
    generation: int
    content: str
    parent_id: Optional[int] = None
    inspired_by_ids: Optional[list[int]] = None
    prompt: Optional[str] = None
    reasoning: Optional[str] = None
    diff: Optional[str] = None
    scores: Optional[dict[str, float]] = None
    artifacts: Optional[dict[str, str]] = None
