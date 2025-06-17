from typing import Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Table, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

Base = declarative_base()


inspiration = Table(
    "inspiration",
    Base.metadata,
    Column("inspired_by_id", ForeignKey("program.id"), primary_key=True),
    Column("inspired_id", ForeignKey("program.id"), primary_key=True),
)


class ProgramModel(Base):
    __tablename__ = "program"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    island_id: Mapped[int] = mapped_column(Integer, nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)

    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("program.id"), nullable=True
    )
    parent: Mapped[Optional["ProgramModel"]] = relationship(
        "ProgramModel",
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[list["ProgramModel"]] = relationship(
        "ProgramModel",
        back_populates="parent",
    )

    inspired_by: Mapped[list["ProgramModel"]] = relationship(
        "ProgramModel",
        secondary=inspiration,
        primaryjoin=id == inspiration.c.inspired_by_id,
        secondaryjoin=id == inspiration.c.inspired_id,
        backref="inspired",
    )

    prompt: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    diff: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    scores: Mapped[Optional[dict[str, float]]] = mapped_column(JSON, nullable=True)
    artifacts: Mapped[Optional[dict[str, str]]] = mapped_column(JSON, nullable=True)
