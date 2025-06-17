import random
import threading
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

from alfred_evolve.models.data_models import Program
from alfred_evolve.models.database_models import Base, ProgramModel
from alfred_evolve.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(f"sqlite:///{db_url}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._lock = threading.Lock()

    @contextmanager
    def get_session(self):
        with self._lock:
            session = self.Session()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Database error: {e}")
                raise e
            finally:
                session.close()

    def add_program(self, program: Program, session = None) -> int:
        if session is None:
            with self.get_session() as session:
                return self.add_program(program, session)

        program_model = self._program_to_model(program)
        session.add(program_model)
        session.flush()

        program_id = program_model.id
        if program_id is None:
            raise ValueError("Failed to add a Program to the database.")
        program.id = program_id

        # Add inspiration relationships
        if program.inspired_by_ids:
            for insp_id in program.inspired_by_ids:
                insp_model = session.query(ProgramModel).get(insp_id)
                if insp_model:
                    program_model.inspired_by.append(insp_model)
            session.add(program_model)
        logger.info(f"Added program with ID {program.id} to the database.")

        best_programs = self.get_topk_programs(1, session=session)
        if best_programs:
            best_program = best_programs[0]
            if best_program.scores:
                best_score = best_program.scores.get("SCORE", 0)
                logger.info(f"Best program after addition: {best_program.id=}, {best_score=}")

        return program_id

    def get_program(self, program_id: int, session = None) -> Optional[Program]:
        if session is None:
            with self.get_session() as session:
                return self.get_program(program_id, session)

        program_model = session.query(ProgramModel).get(program_id)
        if not program_model:
            logger.warning(f"Program with ID {program_id} not found.")
            return None

        program = self._model_to_program(program_model)
        logger.info(f"Retrieved program with ID {program.id}.")
        return program

    def get_topk_programs(
        self,
        k: Optional[int] = None,
        score_key: str = "SCORE",
        island_id: Optional[int] = None,
        session = None,
    ) -> list[Program]:
        if session is None:
            with self.get_session() as session:
                return self.get_topk_programs(k, score_key, island_id, session)

        if self.get_program_count(island_id, session=session) == 0:
            return []

        query = session.query(ProgramModel)
        if island_id is not None:
            query = query.filter(ProgramModel.island_id == island_id)
        query = query.order_by(desc(ProgramModel.scores[score_key].as_float()))
        if k is not None:
            query = query.limit(k)
        program_models = query.all()
        programs = [self._model_to_program(pm) for pm in program_models]
        logger.info(
            f"Retrieved top {'all' if k is None else len(programs)} programs from island {island_id}."
            # f" Scores: {', '.join(str(p.scores.get(score_key, 0.0)) if p.scores else '0.0' for p in programs)}"
        )
        return programs

    def get_random_programs(
        self, n: int, island_id: Optional[int] = None, session = None
    ) -> list[Program]:
        if session is None:
            with self.get_session() as session:
                return self.get_random_programs(n, island_id, session)

        query = session.query(ProgramModel)
        if island_id is not None:
            query = query.filter(ProgramModel.island_id == island_id)
        program_models = query.all()
        if len(program_models) < n:
            logger.warning(
                f"Requested {n} random programs, but only found {len(program_models)}."
            )
            n = len(program_models)
        selected_programs = random.sample(program_models, n)
        programs = [self._model_to_program(pm) for pm in selected_programs]
        logger.info(f"Retrieved {n} random programs from island {island_id}.")
        return programs

    def get_previous_programs(
        self, n: int, island_id: Optional[int] = None, session = None
    ) -> list[Program]:
        if session is None:
            with self.get_session() as session:
                return self.get_previous_programs(n, island_id, session)

        query = session.query(ProgramModel)
        if island_id is not None:
            query = query.filter(ProgramModel.island_id == island_id)
        program_models = query.order_by(ProgramModel.id.desc()).all()
        if len(program_models) < n:
            logger.warning(
                f"Requested {n} previous programs, but only found {len(program_models)}."
            )
            n = len(program_models)
        selected_programs = program_models[:n]
        programs = [self._model_to_program(pm) for pm in selected_programs]
        logger.info(f"Retrieved {n} previous programs from island {island_id}.")
        return programs

    def _program_to_model(self, program: Program) -> ProgramModel:
        return ProgramModel(
            id=program.id,
            island_id=program.island_id,
            generation=program.generation,
            content=program.content,
            parent_id=program.parent_id,
            prompt=program.prompt,
            reasoning=program.reasoning,
            diff=program.diff,
            scores=program.scores,
            artifacts=program.artifacts,
        )

    def _model_to_program(self, program_model: ProgramModel) -> Program:
        return Program(
            id=program_model.id,
            island_id=program_model.island_id,
            generation=program_model.generation,
            content=program_model.content,
            parent_id=program_model.parent_id,
            inspired_by_ids=[p.id for p in program_model.inspired_by],
            prompt=program_model.prompt,
            reasoning=program_model.reasoning,
            diff=program_model.diff,
            scores=program_model.scores,
            artifacts=program_model.artifacts,
        )

    def get_program_count(self, island_id: Optional[int] = None, session = None) -> int:
        if session is None:
            with self.get_session() as session:
                return self.get_program_count(island_id, session)

        query = session.query(ProgramModel)
        if island_id is not None:
            query = query.filter(ProgramModel.island_id == island_id)
        count = query.count()
        # logger.info(f"Island {island_id} contains {count} programs.")
        return count
