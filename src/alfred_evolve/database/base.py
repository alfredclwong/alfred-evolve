from typing import Optional

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship

Base = declarative_base()

inspirations = Table(
    "inspirations",
    Base.metadata,
    Column("program_id", ForeignKey("program.id"), primary_key=True),
    Column("inspiration_id", ForeignKey("program.id"), primary_key=True),
)


class Program(Base):
    __tablename__ = "program"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    island_id: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_id: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False, default="")
    reasoning: Mapped[str] = mapped_column(String, nullable=False, default="")
    prompt: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    diff: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    artifacts: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Foreign keys
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("program.id"), nullable=True
    )

    # Relationships
    parent: Mapped[Optional["Program"]] = relationship(
        "Program",
        remote_side=[id],
        backref="children",
    )
    inspirations: Mapped[list["Program"]] = relationship(
        "Program",
        secondary=inspirations,
        primaryjoin=id == inspirations.c.program_id,
        secondaryjoin=id == inspirations.c.inspiration_id,
        backref="inspired_by",
    )
    scores: Mapped[list["Score"]] = relationship(
        "Score",
        backref=backref("program"),
        lazy="joined",  # eager load the scores using joined loading
    )

    # def __repr__(self):
    #     # content_preview = self.content[:20] if self.content else "No content"
    #     inspirations_str = ",".join([str(insp.id) for insp in self.inspirations])
    #     return (
    #         f"<Program(id={self.id}, island_id={self.island_id}, "
    #         f"parent_id={self.parent_id}, inspirations={inspirations_str})>"
    # )


class Score(Base):
    __tablename__ = "score"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("program.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self):
        return f"<Score(id={self.id}, program_id={self.program_id}, name={self.name}, value={self.value})>"
