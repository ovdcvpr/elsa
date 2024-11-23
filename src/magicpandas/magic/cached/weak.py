

from __future__ import annotations

import weakref

from magicpandas.magic.cached.sticky import Sticky


class Weak(Sticky):
    def __subget__(self: Sticky, outer: Sticky, Outer: type[Sticky]) -> Sticky:
        result = super().__subget__(outer, Outer)
        if isinstance(result, weakref.ref):
            result = result()
            if result is None:
                msg = f'weakref to {self.__trace__} is None'
                raise ValueError(msg)
        return result

    def __subset__(self, instance: Sticky, value):
        if not isinstance(value, weakref.ref):
            value = weakref.ref(value)
        return super().__subset__(instance, value)



property = Weak