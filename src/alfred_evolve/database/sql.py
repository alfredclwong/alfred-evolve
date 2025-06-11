from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import text

from alfred_evolve.database.base import Base
from typing import Type, TypeVar


T = TypeVar("T")


class SQLDatabase:
    def __init__(self, url: str):
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.Session()

    def get(self, model_class: Type[T], filter_by: dict = {}, order_by: str = "") -> T:
        with self.get_session() as session:
            x = (
                session.query(model_class)
                .filter_by(**filter_by)
                .order_by(text(order_by))
                .first()
            )
            if x is None:
                raise ValueError(
                    f"{model_class.__name__} not found with filters {filter_by}"
                )
            return x

    def get_n(
        self,
        model_class: Type[T],
        n: Optional[int] = None,
        filter_by: dict = {},
        order_by: str = "",
    ) -> list[T]:
        with self.get_session() as session:
            query = (
                session.query(model_class)
                .filter_by(**filter_by)
                .order_by(text(order_by))
            )
            if n is not None:
                query = query.limit(n)
            return query.all()

    def add(self, model_instance: T) -> int:
        with self.get_session() as session:
            session.add(model_instance)
            session.commit()
            return model_instance.id

    def close(self):
        self.engine.dispose()

    def __del__(self):
        self.close()
