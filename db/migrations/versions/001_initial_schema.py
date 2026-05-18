"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-17

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "groups",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chunking_configs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(64), unique=True, nullable=False),
        sa.Column("chunk_size", sa.Integer, nullable=False),
        sa.Column("overlap", sa.Integer, nullable=False),
        sa.Column("separators", sa.Text, nullable=True),
        sa.Column("splitter_type", sa.String(64), nullable=False, server_default="RecursiveCharacterTextSplitter"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("group_id", sa.String(64), sa.ForeignKey("groups.id"), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("n_pages", sa.Integer, nullable=False, server_default="0"),
        sa.Column("ingested_at", sa.DateTime, nullable=False),
        sa.Column("chunking_config_id", sa.Integer, sa.ForeignKey("chunking_configs.id"), nullable=True),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("char_count", sa.Integer, nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("vector_id", sa.String(128), nullable=True),
    )

    op.create_table(
        "queries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("group_id", sa.String(64), sa.ForeignKey("groups.id"), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("k", sa.Integer, nullable=False, server_default="4"),
        sa.Column("source_filter", sa.String(512), nullable=True),
        sa.Column("federated", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("asked_at", sa.DateTime, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=True),
    )

    op.create_table(
        "answers",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("query_id", sa.Integer, sa.ForeignKey("queries.id"), nullable=False),
        sa.Column("answer_text", sa.Text, nullable=False),
        sa.Column("confidence", sa.String(16), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("prompt_template_version", sa.String(32), nullable=False, server_default="v1"),
        sa.Column("tokens_in", sa.Integer, nullable=True),
        sa.Column("tokens_out", sa.Integer, nullable=True),
    )

    op.create_table(
        "answer_sources",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("answer_id", sa.Integer, sa.ForeignKey("answers.id"), nullable=False),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id"), nullable=True),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column("chunk_id", sa.Integer, sa.ForeignKey("chunks.id"), nullable=True),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("remote_group_id", sa.String(64), nullable=True),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("config_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "experiment_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("experiment_id", sa.Integer, sa.ForeignKey("experiments.id"), nullable=False),
        sa.Column("question_id", sa.String(64), nullable=True),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("metric_value", sa.Float, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("experiment_results")
    op.drop_table("experiments")
    op.drop_table("answer_sources")
    op.drop_table("answers")
    op.drop_table("queries")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("chunking_configs")
    op.drop_table("groups")
