from __future__ import annotations

import getpass

# import magicpandas as magic
from magicpandas.magic.magic import Magic
from magicpandas.magic.default import default
from magicpandas.magic.magic import Magic
from magicpandas.magic.order import Order

if False:
    from magicpandas.magic.local.locals import Locals

class Local(Magic):
    __root__: Locals
    __order__ = Order.third

    def __subget__(self, outer: Magic, Outer) -> Magic:
        if outer is None:
            return self

        owner = self.__owner__
        key = self.__key__

        if self.__configuring__:
            return self
        elif key in owner.__cache__:
            return owner.__cache__[key]
        elif default.context:
            return self

        # noinspection PyTypeChecker
        trace = self.__trace__.__str__()
        root = self.__root__
        obj = root.dict
        user = getpass.getuser()
        k = None
        for k in trace.split('.'):
            obj = obj[k]
        try:
            result = obj[user]
        except KeyError as e:
            if not self.from_outer:
                msg = (
                    f'{k} was not passed, and could not resolve {trace} '
                    f'with user {user}'
                )
                raise KeyError(msg) from e
            else:
                result = self.from_outer()
        result = self.__subset__(outer, result)
        return result





