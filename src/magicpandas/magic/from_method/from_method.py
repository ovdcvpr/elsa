from __future__ import annotations

from functools import *

from magicpandas import util
from magicpandas.magic.from_method.outer import Outer
from magicpandas.magic.from_method.inner import Inner

if False:
    from magicpandas.magic.magic import Magic


class FromMethod:
    outer = Outer()
    inner = Inner()

    def __set_name__(self, owner, name):
        owner.__directions__[name] = 'diagonal'
        self.__name__ = name
        self.owner = owner

    @util.weakly.cached_property
    def magic(self) -> Magic:
        ...

    @cached_property
    def Magic(self) -> type[Magic]:
        ...

    def __bool__(self):
        return bool(self.outer or self.inner)

    # todo: from_method versus from_attribute; from_attribute.inner; from_attribute.outer

    def __get__(self, instance: Magic, owner):
        self.magic = instance
        self.Magic = owner
        return self

    def __call__(self, *args, **kwargs):
        func = self.outer or self.inner
        if not func:
            raise AttributeError
        return (
            func
            .__get__(self.magic, self.Magic)
            (*args, **kwargs)
        )

    def __set__(self, instance, value):
        instance.__dict__[self.__name__] = value

    def __delete__(self, instance):
        del instance.__dict__[self.__name__]
