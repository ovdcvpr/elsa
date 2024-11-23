from __future__ import annotations

import collections
import inspect
from functools import *

if False:
    from magicpandas.magic.from_method.from_method import FromMethod
    from magicpandas.pandas.ndframe import NDFrame



class Inner(collections.UserDict):
    def __get__(self, instance: FromMethod, owner):
        self.from_params = instance
        return self[instance.Magic]

    def __set__(self, instance: FromMethod, value):
        self[instance.Magic] = self.wrap(value)

    def __delete__(self, instance: FromMethod):
        del self[instance.Magic]

    def __set_name__(self, owner, name):
        self.__name__ = name

    @cached_property
    def names(self):
        second = self.__name__
        first = self.from_params.__name__
        return first, second,

    def __missing__(self, key: type):
        owner = self.from_params.Magic
        for base in key.__bases__:
            if not isinstance(base, owner):
                continue
            obj = base
            for name in self.names:
                obj = getattr(obj, name)
            if obj is None:
                continue
            self[key] = obj
            break
        else:
            self[key] = None

    def wrap(self, func):
        parameters = inspect.signature(func).parameters
        argnames = [
                       param.name
                       for param in parameters.values()
                       # if param.default == param.empty
                   ][1:]
        # skip first
        items = iter(parameters.items())
        next(items)
        defaults = {
            key: param.default
            if param.default != param.empty
            else None
            for key, param in items
        }

        @wraps(func)
        def wrapper(inner: NDFrame, *args, **kwargs):
            owner = inner.__owner__
            outer = inner.__outer__
            passed = defaults.copy()
            passed.update(zip(argnames, args))
            passed.update(kwargs)
            cache = owner.__dict__.setdefault(inner.__key__, {})

            key = tuple(passed.values())
            try:
                return cache[key]
            except KeyError:
                hashable = True
            except (IndexError, TypeError):
                hashable = False

            """
            allow for self.__inner__(frame) 
            while also allowing for super().<inner>(args, kwargs)
            so we replace outer.inner with the constructor function
            while super().<inner> will still cause a recursion
            """

            outer.__inner__, inner_ = inner.enchant, outer.__inner__
            inner.__owner__, owner_ = outer.__third__, inner.__owner__
            # call func with outer.inner(...)
            inner.__outer__ = outer
            volatile = outer.__volatile__.copy()

            result = func(inner, *args, **kwargs)
            # apply inner metadata with outer.__inner__(...)
            outer.__volatile__.update(volatile)
            result = outer.__inner__(result)
            outer.__inner__ = inner_
            inner.__owner__ = owner_

            result = inner.enchant(result)

            # cache result
            if hashable:
                cache[key] = result

            return result

        return wrapper

