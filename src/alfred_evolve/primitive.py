from abc import ABC
from typing import Optional

from alfred_evolve.util import apply_diff


class Primitive(ABC):
    def __init__(self, content: str | dict[str, float]):
        self.content = content

    def __repr__(self):
        content = self.content
        return f"{self.__class__.__name__}({content=})"


class Program(Primitive):
    content: str

    def __init__(self, content: str, program_id: Optional[int], result: Optional["Result"] = None):
        super().__init__(content)
        self.program_id = program_id
        self.result = result


class Prompt(Primitive):
    content: str


class Diff(Primitive):
    content: str

    def apply(self, program: Program) -> Program:
        return Program(apply_diff(program.content, self.content), None)


class Result(Primitive):
    content: dict[str, float]
