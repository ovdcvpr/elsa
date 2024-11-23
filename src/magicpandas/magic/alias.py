from __future__ import annotations

from functools import *

from magicpandas.magic.abc import ABCMagic

"""
@magic.cached.property
def thing(self):
    ...

@magic.proxy.property
@magic.proxy('thing').property
@magic.proxy(thing).property
"""


class Alias:
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, instance, owner):
        if instance is not None:
            return getattr(instance, self.proxy)
        else:
            return getattr(owner, self.proxy)

    def __set__(self, instance, value):
        setattr(instance, self.proxy, value)

    def __delete__(self, instance):
        delattr(instance, self.proxy)

    def __init__(self, ref: ABCMagic | str = None):
        if isinstance(ref, ABCMagic):
            self.proxy = ref.__name__
        elif isinstance(ref, str):
            self.proxy = ref

    def __call__(self, *args, **kwargs):
        return self

    @cached_property
    def proxy(self) -> str:
        return f'__{self.__name__}__'


class Property:
    def __get__(self, instance: alias, owner: type[alias]):
        if instance:
            return instance.proxy
        return Alias

class alias:
    property = Property()

    def __init__(self, ref: ABCMagic | str = None):
        self.proxy = Alias(ref)
