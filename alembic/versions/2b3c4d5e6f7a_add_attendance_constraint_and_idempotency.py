"""add_attendance_constraint_and_idempotency

Revision ID: 2b3c4d5e6f7a
Revises: 9067670a55af
Create Date: 2026-10-04 00:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2b3c4d5e6f7a'
down_revision: Union[str, Sequence[str], None] = '9067670a55af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add idempotency_key to processing_sessions
    with op.batch_alter_table('processing_sessions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=64), nullable=True))
        batch_op.create_index(batch_op.f('ix_processing_sessions_idempotency_key'), ['idempotency_key'], unique=True)

    # 2. Add UniqueConstraint to attendance
    with op.batch_alter_table('attendance', schema=None) as batch_op:
        batch_op.create_unique_constraint('uq_attendance_lecture_student', ['lecture_id', 'student_id'])


def downgrade() -> None:
    with op.batch_alter_table('attendance', schema=None) as batch_op:
        batch_op.drop_constraint('uq_attendance_lecture_student', type_='unique')

    with op.batch_alter_table('processing_sessions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_processing_sessions_idempotency_key'))
        batch_op.drop_column('idempotency_key')
