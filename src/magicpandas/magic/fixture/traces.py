from __future__ import annotations

from typing import *

if False:
    from magicpandas import Magic

"""
how can we best get the list of traces
e.g. 
selected.nlse.twenty.nms

for bases, we get their traces
for owner, we check all objects in dict, if any are fixtures, add their name
"""


class Traces(set[str]):
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__cache__ = {}

    def __get__(self, instance: Magic, owner: type[Magic]) -> Self:
        # from magicpandas.magic.cached.magic import Magic
        from magicpandas.magic.magic import Magic
        base: type[Magic]
        if owner not in self.__cache__:
            from magicpandas.magic.fixture.fixture import Fixture
            result = Traces(
                trace
                for base in owner.__bases__[::-1]
                if issubclass(base, Magic)
                for trace in getattr(base, self.__name__)
            )
            result.update(
                key
                for key, value in owner.__dict__.items()
                if isinstance(value, Fixture)
            )
            result.update(
                key + '.' + trace
                for key, value in owner.__dict__.items()
                if isinstance(value, Magic)
                and not isinstance(value, Fixture)
                # for trace in value.__fixtures__
                for trace in getattr(value, self.__name__)
            )
            self.__cache__[owner] = result

        result = self.__cache__[owner]
        if instance is None:
            return result

        if self.__name__ not in instance.__cache__:
            instance.__cache__[self.__name__] = result = result.copy()

        return result
