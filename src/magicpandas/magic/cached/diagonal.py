from __future__ import annotations

from magicpandas.magic.cached.base import Base


class Diagonal(Base):
    __direction__ = 'diagonal'


property = Diagonal
