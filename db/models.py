from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Group(Base):
    __tablename__ = "groups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    documents: Mapped[list["Document"]] = relationship(back_populates="group", cascade="all, delete-orphan")
    queries: Mapped[list["Query"]] = relationship(back_populates="group", cascade="all, delete-orphan")


class ChunkingConfig(Base):
    __tablename__ = "chunking_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    separators: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    splitter_type: Mapped[str] = mapped_column(String(64), default="RecursiveCharacterTextSplitter")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[str] = mapped_column(String(64), ForeignKey("groups.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    n_pages: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunking_config_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("chunking_configs.id"), nullable=True)

    group: Mapped["Group"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    answer_sources: Mapped[list["AnswerSource"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id"), nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    vector_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[str] = mapped_column(String(64), ForeignKey("groups.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    k: Mapped[int] = mapped_column(Integer, default=4)
    source_filter: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    federated: Mapped[bool] = mapped_column(Boolean, default=False)
    asked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    group: Mapped["Group"] = relationship(back_populates="queries")
    answer: Mapped[Optional["Answer"]] = relationship(back_populates="query", cascade="all, delete-orphan", uselist=False)


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("queries.id"), nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    prompt_template_version: Mapped[str] = mapped_column(String(32), default="v1")
    tokens_in: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    query: Mapped["Query"] = relationship(back_populates="answer")
    sources: Mapped[list["AnswerSource"]] = relationship(back_populates="answer", cascade="all, delete-orphan")


class AnswerSource(Base):
    __tablename__ = "answer_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(Integer, ForeignKey("answers.id"), nullable=False)
    document_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("documents.id"), nullable=True)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("chunks.id"), nullable=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    remote_group_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    answer: Mapped["Answer"] = relationship(back_populates="sources")
    document: Mapped[Optional["Document"]] = relationship(back_populates="answer_sources")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    results: Mapped[list["ExperimentResult"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id"), nullable=False)
    question_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    experiment: Mapped["Experiment"] = relationship(back_populates="results")
