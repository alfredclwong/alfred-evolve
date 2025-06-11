from typing import Optional


class Program:
    def __init__(self, island_id: int, parent: Optional["Program"] = None, diff: Optional["Diff"] = None):
        self.island_id = island_id
        self.parent = parent
        self.diff = diff

    def __repr__(self):
        return f"Program(island_id={self.island_id}, parent={self.parent}, diff={self.diff})"


class Prompt:
    pass


class Diff:
    pass


class Result:
    pass
