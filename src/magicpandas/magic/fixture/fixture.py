from __future__ import annotations

import os.path

import inspect
import os
from functools import *
from typing import *

from magicpandas.magic.truthy import truthy
from magicpandas import util 
from magicpandas.magic.default import default
from magicpandas.magic.magic import Magic
from magicpandas.magic.order import Order


class Fixture(Magic):
    __order__ = Order.third
    """
    Grid.evaluate.scored.fixtures

    fixtures/evaluate/scored/nlse/.2/nms
    fixtures.evaluate.scored.nlse.twenty.nms

    if it's callable and file does not exist, you "prime" a fixture by calling it

    elsa.evaluate.scored.fixtures.selected.nlse.twenty.nms

    selected.nlse.twenty.nms

    elsa.evaluate.scored.fixtures.selected

    selected.nlse.twenty.nms -> second order
    selected.nlse.twenty.nms(...) -> third order
    selected.nlse.twenty.nms -> second order

    test/fixtures/evaluate/scored/nlse/twenty/nms.parquet

    fixtures/evaluate/scored/nlse/twenty/nms.parquet
    """

    def __fspath__(self):
        result = os.path.join(
            inspect.getfile(self.root.__class__),
            '..',
            'test',
            'fixtures',
            *self.__trace__.split('.')
        )
        result = os.path.abspath(result)
        return result

    def __to_file__(self, value=None):
        ...

    def __from_file__(self) -> Self:
        ...

    @classmethod
    def __fixture_wrapper__(cls, func):
        # noinspection PyUnresolvedReferences
        @wraps(func)
        def wrapper(self: Self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            result = self.__to_file__(result)
            instance = self.__outer__
            result: Self = self.__subset__(instance, result)
            result.__unlink__ = partial(os.unlink, self.__fspath__())
            return result

        return wrapper

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__call__' in cls.__dict__:
            cls.__call__ = cls.__fixture_wrapper__(cls.__call__)

    def __init_func__(self, func=None, *args, **kwargs):
        super().__init_func__(func, *args, **kwargs)
        if not (
                not util.returns(func)
                and not util.contains_functioning_code(func)
        ):
            self.from_outer = self.__fixture_wrapper__(self.from_outer)

    def __subget__(self, outer: Magic, Outer) -> Magic:
        if outer is None:
            return self

        owner: Magic = self.outer
        key = self.__key__

        if self.__configuring__:
            return self
        elif key in owner.__dict__:
            return owner.__dict__[key]
        elif default.context:
            return self

        third = self.__class__()
        third.__order__ = Order.third
        self.__propagate__(third)

        if self.__from_method__:
            return third
        if (
                third.__from_file__
                and os.path.exists(third)
        ):
            # todo: problem is isnt wrapping
            return self.__from_file__()
        elif self.__call__:
            return third
        elif self.__from_method__:
            return third
        elif self.from_outer:
            func = third.from_outer.__func__.__get__(outer, type(outer))
            func = third.__fixture_wrapper__(func)
            outer.__inner__, inner = third, outer.__inner__
            result = third.__log__(func)
            outer.__inner__ = inner
            result = third.__postprocess__(result)
            return result

        elif third.conjure:
            func = third.__fixture_wrapper__(third.conjure)
            result = third.__log__(func)
            result = third.__postprocess__(result)
            return result

        else:
            raise RuntimeError

    # @magic.truthy
    @truthy
    def __call__(self, *args, **kwargs):
        result = self.__from_method__(*args, **kwargs)
        return result

    def __unlink__(self):
        os.unlink(self)

