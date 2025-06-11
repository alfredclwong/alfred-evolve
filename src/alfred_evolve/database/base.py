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


class SQLProgram(Base):
    __tablename__ = "program"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    island_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Foreign keys
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("program.id"), nullable=True
    )
    prompt_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("prompt.id"), nullable=True
    )
    diff_id: Mapped[Optional[int]] = mapped_column(ForeignKey("diff.id"), nullable=True)
    result_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("result.id"), nullable=True
    )

    # Relationships
    parent: Mapped[Optional["SQLProgram"]] = relationship(
        "SQLProgram",
        remote_side=[id],
        backref="children",
    )
    inspirations: Mapped[list["SQLProgram"]] = relationship(
        "SQLProgram",
        secondary=inspirations,
        primaryjoin=id == inspirations.c.program_id,
        secondaryjoin=id == inspirations.c.inspiration_id,
        backref="inspired_programs",
    )
    prompt: Mapped[Optional["SQLPrompt"]] = relationship("SQLPrompt", backref="program", foreign_keys=[prompt_id])
    diff: Mapped[Optional["SQLDiff"]] = relationship("SQLDiff", backref="program", foreign_keys=[diff_id])
    result: Mapped[Optional["SQLResult"]] = relationship("SQLResult", backref="program", foreign_keys=[result_id])


class SQLPrompt(Base):
    __tablename__ = "prompt"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("program.id"), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)


class SQLDiff(Base):
    __tablename__ = "diff"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("program.id"), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)


class SQLResult(Base):
    __tablename__ = "result"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("program.id"), nullable=False)
    scores: Mapped[list["SQLScore"]] = relationship(
        "SQLScore", backref=backref("result")
    )


class SQLScore(Base):
    __tablename__ = "score"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("result.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
