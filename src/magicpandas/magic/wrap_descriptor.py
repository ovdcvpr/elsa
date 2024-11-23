from __future__ import annotations

from typing import TypeVar
from magicpandas.magic import globals

T = TypeVar('T')

if False:
    from .magic import Magic

def __wrap_descriptor__(first: Magic, func, outer: Magic, *args, **kwargs):
    """
    wraps __get__, __set__, __delete__ to handle the "magic" operations
    of propagating all the metadata through the attribute chain
    """
    OUTER = outer
    first.__outer__ = outer
    order = first.__class__.__order__
    # todo: problem is this causes config to fail

    if outer is None:
        first.__root__ = root = None
        first.__third__ = None
        first.__owner__ = None
        first.__Root__ = first.__Outer__ = args[0]
        first.__nothird__ = True
    else:
        first.__root__ = root = outer.__root__
        first.__third__ = outer.__third__
        first.__Root__ = outer.__Root__
        first.__nothird__ = (
            outer.__order__ != 3
            and outer.__class__.__order__ == 3
            or outer.__nothird__
        )

        if outer is None:
            first.__Outer__ = args[0]
        else:
            first.__Outer__ = type(outer)
    if (
            outer is None
            or order == 1
            # To access nested metadata from class attributes before set_name is called on them
            # or outer.__wrapped__
    ):
        # Frame.magic
        # Frame.frame
        return first

    if (
        root is None
    ) or (
        first.__nothird__
        and not first.__configuring__
    ):
        try:
            first.__owner__ = outer.__second__
        except AttributeError:
            first.__owner__ = outer.__first__
        match order:
            case 1:
                return first
            case 3 | 2:
                # Frame.frame
                # Frame.frame.magic
                first.__owner__ = outer.__second__
                second = first.__second__
                first.__propagate__(second)
                return second

    match order:
        case 1:
            return first
        case 2:
            # frame.magic.magic
            # frame.frame.magic
            first.__owner__ = outer.__second__
            second = first.__second__
            if globals.chained_safety:
                second.__chained_safety__ = outer.__third__
            first.__propagate__(second)
            return second
        case 3:
            # frame.magic.frame
            # frame.frame.frame
            owner = outer.__second__
            if owner is None:
                owner = outer.__third__
            first.__owner__ = owner
            second = first.__second__
            inner, outer.__inner__ = outer.__inner__, second
            first.__owner__ = outer.__third__
            if globals.chained_safety:
                second.__chained_safety__ = outer.__third__
            first.__propagate__(second)
            third: Magic = func(second, outer, *args, **kwargs)
            outer.__inner__ = inner
            return third

    raise RuntimeError

