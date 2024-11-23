from __future__ import annotations

from functools import *
from types import *
from typing import *

import magicpandas.util as util

if False:
    from magicpandas.magic.magic import Magic


class Test:

    @util.weakly.cached_property
    def __magic__(self) -> Magic:
        ...

    @util.weakly.cached_property
    def __Magic__(self) -> Magic:
        ...

    @cached_property
    def __inner__(self) -> bool:
        """
        If True, the magic is inside self, so magic.outer needs to be
        accessed to run the function with the correct instance.
        """

    @cached_property
    def __func__(self) -> MethodDescriptorType | Callable:
        """
        Test function that has been wrapped by this class
        """

    def __init__(
            self,
            func,
            inner: bool = False,
    ):
        update_wrapper(self, func)
        self.__func__ = func
        self.__inner__ = inner

    def __call__(self, *args, **kwargs):
        instance = self.__magic__
        owner = self.__Magic__
        if self.__inner__:
            instance = instance.__outer__
            owner = instance.__class__

        result = (
            self.__func__
            .__get__(instance, owner)
            .__call__(*args, **kwargs)
        )
        return result

    def __get__(self, instance, owner) -> Self:
        self.__magic__ = instance
        self.__Magic__ = owner
        return self

    def __set_name__(self, owner, name):
        self.__name__ = name

    @property
    def __trace__(self):
        return f'{self.__magic__.__trace__}.{self.__name__}'


class test(Test):
    ...


locals()['test'] = Test
