from __future__ import annotations

import importlib
import inspect
import weakref

import magicpandas.util as util
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.cached.base import Base

if False:
    import magicpandas as magic


def __get__(self, instance: magic.Magic, owner: type):
    if instance is None:
        return self
    if instance.__root__ is not None:
        instance = instance.__root__

    return super(Root, self).__get__(instance, owner)


class Root(Base):
    locals()['__get__'] = __get__

    # stores in root attrs
    def __set__(self, instance, value):
        if instance.__root__ is not None:
            instance = instance.__root__
        super().__set__(instance, value)

    def __delete__(self, instance: magic.Magic):
        if instance.__root__ is not None:
            instance = instance.__root__
        super().__delete__(instance)

# class Root(magic.Magic):
#     __order__ = Order.third
#     __subset__ = magic.Magic.__subset__
#     __subdelete__ = magic.Magic.__subdelete__
#
#     # noinspection PyUnresolvedReferences,PyRedeclaration
#     def __subget__(self: Root, instance: magic.Magic, owner):
#         if instance is None:
#             return self
#         if default.context:
#             return self
#         if instance.__root__ is not None:
#             instance = instance.__root__
#         return super().__subget__(instance, owner)
#
#     def __wrap_descriptor__(first, func, outer, *args, **kwargs):
#         if (
#                 outer is not None
#                 and outer.__root__ is not None
#         ):
#             outer = outer.__root__
#         return super().__wrap_descriptor__(func, outer, *args, **kwargs)
#

property = Root