"""Postgres + pgvector datastore.

One table, ``chunks``: each row is a citable Source (one FHIR resource in the
MVP) with its embedding. Retrieval is exact cosine distance scoped to a patient
— at one-patient/MVP scale an ANN index is unnecessary, so we keep it simple and
exact. (An HNSW index is the obvious add when the cohort grows.)
"""

from __future__ import annotations

from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    String,
    Text,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import get_settings
from app.schemas import Coding, Source


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunks"

    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    resource_type: Mapped[str] = mapped_column(String, index=True)
    resource_id: Mapped[str] = mapped_column(String)
    patient_id: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    date: Mapped[str | None] = mapped_column(String, nullable=True)
    codings: Mapped[list] = mapped_column(JSON, default=list)
    embedding: Mapped[list[float]] = mapped_column(Vector(get_settings().embedding_dim))

    def to_source(self) -> Source:
        return Source(
            source_id=self.source_id,
            resource_type=self.resource_type,
            resource_id=self.resource_id,
            patient_id=self.patient_id,
            title=self.title,
            text=self.body,
            date=self.date,
            codings=[Coding(**c) for c in (self.codings or [])],
        )


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def init_db() -> None:
    """Enable pgvector and create tables. Idempotent."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def upsert_chunks(rows: list[Chunk]) -> int:
    """Replace any existing rows with the same source_id, then insert."""
    engine = get_engine()
    with Session(engine) as session:
        ids = [r.source_id for r in rows]
        session.execute(delete(Chunk).where(Chunk.source_id.in_(ids)))
        session.add_all(rows)
        session.commit()
        return len(rows)


def delete_patient(patient_id: str) -> int:
    engine = get_engine()
    with Session(engine) as session:
        result = session.execute(delete(Chunk).where(Chunk.patient_id == patient_id))
        session.commit()
        return result.rowcount or 0


def list_patients() -> list[str]:
    engine = get_engine()
    with Session(engine) as session:
        rows = session.execute(select(Chunk.patient_id).distinct()).scalars().all()
        return sorted(rows)


def semantic_search(patient_id: str, query_embedding: list[float], k: int) -> list[Source]:
    """Top-k chunks for a patient by cosine distance."""
    engine = get_engine()
    with Session(engine) as session:
        stmt = (
            select(Chunk)
            .where(Chunk.patient_id == patient_id)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        return [c.to_source() for c in session.execute(stmt).scalars().all()]
