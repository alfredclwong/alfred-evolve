from dataclasses import dataclass
from functools import reduce

from alfred_evolve.database.base import (
    SQLDiff,
    SQLProgram,
    SQLPrompt,
    SQLResult,
    SQLScore,
)
from alfred_evolve.database.sql import SQLDatabase
from alfred_evolve.primitive import Diff, Program, Prompt, Result
from alfred_evolve.util import apply_diff


@dataclass(frozen=True)
class ProgramDatabaseConfig:
    url: str
    n_islands: int
    initial_program_content: str
    n_inspirations: int


class ProgramDatabase(SQLDatabase):
    def __init__(self, cfg: ProgramDatabaseConfig):
        super().__init__(cfg.url)

        for i in range(cfg.n_islands):
            self.add(SQLProgram(island_id=i, parent=None, inspirations=[]))

        self.cfg = cfg
        self.current_island = 0

    def sample(self) -> tuple[Program, list[Program]]:
        parent = self._sample_parent()
        inspirations = self._sample_inspirations(parent)
        return parent, inspirations

    def _sample_parent(self) -> Program:
        filters = {"island_id": self.current_island}
        parent = self.get(SQLProgram, filter_by=filters, order_by="id desc")
        content = self._get_content(parent)
        return Program(content=content, program_id=parent.id)

    def _sample_inspirations(self, parent: Program) -> list[Program]:
        inspirations = self.get_n(
            SQLProgram,
            n=self.cfg.n_inspirations,
            filter_by={"island_id": self.current_island},
        )
        return [self.get_program(inspiration.id) for inspiration in inspirations]

    def get_program(self, program_id: int) -> Program:
        """Retrieve a program by its ID.

        Full program contents are not stored in the database. We need to
        reconstruct the contents by traversing the evolution tree up to the
        root program, then iteratively applying diffs down to the current
        program.

        Args:
            program_id (int): The ID of the program to retrieve.
        Returns:
            Program: The reconstructed program.
        """
        program = self.get(SQLProgram, filter_by={"id": program_id})
        content = self._get_content(program)
        try:
            result = self.get(SQLResult, filter_by={"program_id": program.id})
        except ValueError:
            result = None
        return Program(content=content, program_id=program.id, result=result)

    def _get_content(self, sql_program) -> str:
        child = sql_program
        diff_contents: list[str] = []
        while True:
            parent = (
                self.get(SQLProgram, filter_by={"id": child.parent_id})
                if child.parent_id is not None else None
            )
            if parent is None:
                break  # Reached the root program
            diff = self.get(SQLDiff, filter_by={"program_id": child.id})
            if diff is None:
                raise ValueError(
                    f"Child program with ID {child.id} has no diff."
                )  # Every child should have a diff
            diff_contents.append(diff.content)
            child = self.get(SQLProgram, filter_by={"id": parent.id})
        content = reduce(
            lambda acc, diff: apply_diff(acc, diff),
            reversed(diff_contents),
            self.cfg.initial_program_content,
        )
        return content

    def add_program(
        self,
        parent_id: int,
        inspiration_ids: list[int],
        prompt: Prompt,
        diff: Diff,
        result: Result,
    ):
        """Add a new program to the database.

        Args:
            parent (Program): The parent program.
            inspirations (list[Program]): List of inspiration programs.
            prompt (Prompt): The prompt used to generate the program.
            diff (Diff): The diff applied to the parent program.
            result (Result): The evaluation result of the child program.
        """
        sql_parent = self.get(SQLProgram, filter_by={"id": parent_id})
        sql_inspirations = [
            self.get(SQLProgram, filter_by={"id": inspiration_id})
            for inspiration_id in inspiration_ids
        ]

        sql_child = SQLProgram(
            island_id=sql_parent.island_id,
            parent_id=sql_parent.id,
            inspirations=sql_inspirations,
        )
        child_id = self.add(sql_child)

        sql_prompt = SQLPrompt(content=prompt.content, program_id=child_id)
        sql_diff = SQLDiff(content=diff.content, program_id=child_id)
        self.add(sql_prompt)
        self.add(sql_diff)
        self._add_result(child_id, result)

    def _add_result(self, program_id: int, result: Result):
        sql_result = SQLResult(program_id=program_id)
        result_id = self.add(sql_result)
        for name, value in result.content.items():
            sql_score = SQLScore(result_id=result_id, name=name, value=value)
            self.add(sql_score)

    def get_programs(self) -> list[Program]:
        sql_programs = self.get_n(SQLProgram)
        programs = []
        for sql_program in sql_programs:
            content = self._get_content(sql_program)
            try:
                result = self.get(SQLResult, filter_by={"program_id": sql_program.id})
            except ValueError:
                result = None
            programs.append(
                Program(content=content, program_id=sql_program.id, result=result)
            )
        return programs

    def get_results(self) -> list[Result]:
        sql_results = self.get_n(SQLResult)
        results = []
        for sql_result in sql_results:
            scores = self.get_n(SQLScore, filter_by={"result_id": sql_result.id})
            content = {score.name: score.value for score in scores}
            results.append(Result(content=content))
        return results
