from __future__ import annotations

import importlib
import inspect
import sys
import weakref

import magicpandas as magic
import magicpandas.util as util

if False:
    import magicpandas as magic



# def __get__(self, instance: magic.Magic, owner: type[magic.Magic]):
def __get__(
        self: Outer,
        instance: magic.Magic,
        owner: type[magic.Magic]
):
    self.__Outer__ = owner
    self.__magic__ = instance
    if instance is None:
        return self
    cache = self.__cache__
    key = self.__key__

    if key not in cache:
        # todo: can this be optimized?
        # find the class that defined the instance
        mro = owner.mro()
        name = self.__name__
        base = owner

        while mro:
            for base in mro:
                if name in base.__dict__:
                    mro = None
                    break
                if hasattr(base, name):
                    mro = base.mro()[1:]
                    break
            else:
                raise RuntimeError('bad logic')

        module = importlib.import_module(base.__module__)
        try:
            annotation = (
                inspect
                .get_annotations(self.from_outer)
                ['return']
            )
        except KeyError as e:
            msg = (
                f'No return annotation for {self.from_outer}; '
                f'unable to find an outer instance to cache.'
            )
            raise ValueError(msg) from e
        cls = util.resolve_annotation(module, annotation)

        outer = instance
        while True:
            if outer is None:
                msg = (
                    f'Unable to find an outer instance for '
                    f'annotation {annotation} '
                )
                raise ValueError(msg)
            if isinstance(outer, cls):
                break
            outer = outer.__outer__

        self.__set__(instance, outer)

    result = cache[key]
    if isinstance(result, weakref.ref):
        result = result()
        if result is None:
            raise ValueError(
                f"weakref to {key} in {instance} is None"
            )

    return result


# class Outer(Base):
#     locals()['__get__'] = __get__

def walk(outer: magic.Magic, cls: type) -> magic.Magic:
    while True:
        if outer is None:
            msg = (
                f'Unable to find an outer instance for '
                f'annotation {cls} '
            )
            raise ValueError(msg)
        if isinstance(outer, cls):
            break
        outer = outer.__outer__
    return outer


def __get__(self: Outer, instance: magic.Magic, owner: type[magic.Magic]):
    # instance = self.__owner__
    self.__magic__ = instance
    if instance is None:
        result = self
        return result
    key = self.__key__
    cache = instance.__third__.__dict__
    if key in cache:
        result = cache[key]
        result = result()
        return result

    mro = owner.mro()
    name = self.__name__
    base = owner

    while mro:
        for base in mro:
            if name in base.__dict__:
                mro = None
                break
            if hasattr(base, name):
                mro = base.mro()[1:]
                break
        else:
            raise RuntimeError('bad logic')

    module = importlib.import_module(base.__module__)
    annotation = self.__annotation__
    cls = util.resolve_annotation(module, annotation)
    outer = instance

    try:
        result = walk(outer, cls)
    except ValueError as e:
        # it might be imported from "__main__", for which we cannot
        # use importlib.import_module
        try:
            main = sys.modules['__main__']
            name = annotation.rsplit('.', 1)[-1]
            cls = getattr(main, name)
            result = walk(outer, cls)
        except (ValueError, TypeError, AttributeError):
            raise e

    # self.__set__(instance, result)
    if result.__order__ == 3:
        self.__set__(instance, result)


    return result


class Outer:
    # key belongs in attrs
    # instance belongs in dict
    locals()['__get__'] = __get__

    @util.weakly.cached_property
    def __magic__(self) -> magic.Magic:
        ...

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __init__(self, func):
        self.__annotation__: str = func.__annotations__['return']

    @property
    def __cache__(self):
        return self.__magic__.__third__.__cache__

    @property
    def __key__(self):
        magic = self.__magic__
        key = self.__name__
        cache = magic.__cache__
        if key in cache:
            return cache[key]
        result = (
                magic.__trace__
                - magic.__third__.__trace__
                + f'{self.__name__}.__key__'
        )
        cache[key] = result
        return result

    def __set__(self, instance: magic.Magic, value):
        self.__magic__ = instance
        if value is not None:
            value = weakref.ref(value)
        self.__cache__[self.__key__] = value

    def __delete__(self, instance: magic.Magic):
        self.__magic__ = instance
        try:
            del self.__cache__[self.__key__]
        except KeyError:
            ...

"""
If outer is third, it can cache 
Otherwise, it cannot cache; just walk until it's found
"""

"""
elsa.predict.elsa
1    2      3
elsa[predict.elsa] = elsa
"""