from __future__ import annotations

import collections
from types import *

if False:
    from .magic import Magic



class Options(collections.UserDict):
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__cache__ = {}

    def __get__(
            self,
            instance,
            owner: type[Magic],
    ) -> MappingProxyType[str, bool]:
        from magicpandas.magic.magic import Magic

        if owner not in self.__cache__:
            result = self.copy()
            result.update({
                key: value
                for base in owner.__bases__[::-1]
                if issubclass(base, Magic)
                for key, value in base.__options__.items()
            })
            try:
                from_options = owner.__dict__['from_options']
            except KeyError:
                ...
            else:
                kwdefaults = from_options.__func__.__kwdefaults__
                if kwdefaults is None:
                    raise NotImplementedError
                result.update(kwdefaults)

            result = MappingProxyType(result)
            self.__cache__[owner] = result
        else:
            result = self.__cache__[owner]
        return result
