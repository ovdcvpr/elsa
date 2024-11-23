
from __future__ import annotations

from magicpandas.magic.cached.base import Base

if False:
    import magicpandas as magic

class Public(Base):
    def __get__(self, instance: magic.Magic, owner: type):
        return getattr(instance, f'__{self.__name__}__')

    def __set__(self, instance: magic.Magic, value):
        setattr(instance, f'__{self.__name__}__', value)

    def __delete__(self, instance: magic.Magic):
        delattr(instance, f'__{self.__name__}__')


property = Public