
from __future__ import annotations

import os.path

import warnings
import weakref

from magicpandas.magic import magic
from magicpandas.magic.cached.sticky import Sticky
from magicpandas.magic.default import default
from magicpandas.magic.order import Order

if False:
    from magicpandas.pandas.ndframe import NDFrame

class Serialized(Sticky):
    __order__ = Order.third
    def __subget__(self, outer: magic.Magic, Outer):
        owner: NDFrame
        if outer is None:
            return self

        elif default.context:
            return self

        elif self.__configuring__:
            key = self.__trace__.__str__()
            if key not in self.__config__:
                self.__config__[key] = self.from_outer()
            result = self.__config__[key]
            return result

        if self.__nothird__:
            return self

        # noinspection PyTypeChecker
        owner = self.__owner__
        key = self.__key__
        trace = self.__trace__.__str__()

        if (
                owner.__metadata__ is not None
                and key in owner.__metadata__
        ):
            result = owner.__metadata__[key]
            owner.__cache__[key] = result
            return result

        if key in owner.__cache__:
            # get from cached instance attr
            result = owner.__cache__[key]
            if isinstance(result, weakref.ref):
                result = result()
            return result

        if trace in owner.__config__:
            # get from config
            return owner.__config__[trace]

        # todo: maybe use volatile in wrap_descriptor instead to minimize user error
        volatile = self.__volatile__.copy()

        if (
                self.__from_file__
                and os.path.exists(self)
        ):
            # load from file
            result = self.__from_file__()
            # todo: could this cause a memory leak?
            try:
                result.__unlink__ = self.__unlink__
            except AttributeError as e:
                warnings.warn(str(e))

        elif self.from_outer:
            # compute from func
            func = self.from_outer.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            if self.__log__:
                result = self.__log__(func)
            else:
                result = func()
            outer.__inner__ = inner

        elif self.conjure:
            # load from inner
            if self.__log__:
                result = self.__log__(self.conjure, outer)
            else:
                result = self.conjure()
        else:
            raise ValueError(
                f'Could not resolve a constructor for {self.__trace__}. '
                f'If get-before-set is acceptable, you must explicitly return None.'
            )

        # noinspection PyUnresolvedReferences
        self.__subset__(outer, result)
        # result = owner.attrs[key]
        result = owner.__cache__[key]
        if isinstance(result, weakref.ref):
            result = result()
        if (
                self.__from_file__
                and not os.path.exists(self)
        ):
            self.__to_file__(result)

        self.__volatile__.update(volatile)

        return result


property = Serialized