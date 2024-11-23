from __future__ import annotations

import weakref

from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.cached.base import Base

if False:
    import magicpandas as magic


def __get__(self: Volatile, instance: magic.Magic, owner):
    if instance is None:
        return self
    key = self.from_outer.__name__
    cache = instance.__volatile__
    if key not in cache:
        value = (
            self
            .from_outer
            .__get__(instance, owner)
            .__call__()
        )
        self.__set__(instance, value)
    result = cache[key]
    if isinstance(result, weakref.ref):
        result = result()
        if result is None:
            trace = instance.__trace__
            order = instance.__order__
            msg = f"Weakref to {key} in {trace=} {order=} is None"
            raise ValueError(msg)
    return result


class Volatile(Base):
    """
    properties can change while something like from_outer
    or conjure is run; volatile properties are locally
    cached before a process is run so that they may be renewed
    after the process is done
    """

    locals()['__get__'] = __get__

    def __set__(self, instance: magic.Magic, value):
        cache = instance.__volatile__
        key = self.__name__

        if isinstance(value, ABCMagic):
            value = weakref.ref(value)
            if value() is None:
                msg = f"weakref to {key} in {instance} is None"
                raise ValueError(msg)
        cache[key] = value

    def __delete__(self, instance: magic.Magic):
        cache = instance.__volatile__
        key = self.__name__
        if key in cache:
            del cache[key]


property = Volatile
