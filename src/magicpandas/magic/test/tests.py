from __future__ import annotations

from typing import *

import magicpandas.util as util
from magicpandas.magic import globals
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.test.test import Test

if False:
    from magicpandas.magic.magic import Magic


class Tests(set[Test]):

    def __set_name__(self, owner: Magic, name):
        self.__cache__: dict[type[Magic], Self] = {}
        self.__name__ = name

    def __get__(self, instance: Magic, owner: type[Magic]) -> Self:
        INSTANCE = instance
        OWNER = owner
        key = self.__name__
        if owner not in self.__cache__:
            result = self.__cache__[owner] = self.__class__(
                test
                for base in owner.__bases__
                if issubclass(base, ABCMagic)
                for test in getattr(base, key, [])
            )
            result.update({
                value
                for value in owner.__dict__.values()
                if isinstance(value, Test)
            })
        else:
            result = self.__cache__[owner]

        if instance is not None:
            instance = instance.__first__
            cache = instance.__dict__
            if key not in cache:
                cache[key] = result.copy()
            result = cache[key]
        result.__magic__ = INSTANCE
        result.__Magic__ = OWNER

        return result

    @util.weakly.cached_property
    def __magic__(self) -> Magic:
        ...

    @util.weakly.cached_property
    def __Magic__(self) -> Magic:
        ...

    def __call__(self, *args, **kwargs):
        if not globals.test:
            return
        magic = self.__magic__
        Magic = self.__Magic__
        if magic is None:
            raise ValueError('No magic instance found')
        if not self:
            return
        if magic.__order__ != 3:
            return
        header = False
        for test in self:


            try:
                result = (
                    test
                    .__get__(magic, Magic)
                    .__call__(*args, **kwargs)
                )
            except Exception as e:
                if not header:
                    msg = f'Testing {magic.trace}...'
                    magic.logger.info(msg)
                    magic.logger.indent += 1
                    header = True
                msg = f'Test {test.__trace__} failed: {e}'
                magic.logger.warn(msg)

            # todo: very odd: this breaks frame[precision recall]
            # if globals.tests_raise:
            #     result = (
            #         test
            #         .__get__(magic, Magic)
            #         .__call__(*args, **kwargs)
            #     )
            # else:
            #     try:
            #         result = (
            #             test
            #             .__get__(magic, Magic)
            #             .__call__(*args, **kwargs)
            #         )
            #     except Exception as e:
            #         if not header:
            #             msg = f'Testing {magic.trace}...'
            #             magic.logger.info(msg)
            #             magic.logger.indent += 1
            #             header = True
            #         msg = f'Test {test.__trace__} failed: {e}'
            #         magic.logger.warn(msg)
        if header:
            magic.logger.indent -= 1

    def copy(self):
        return self.__class__(self)

    def __set__(self, instance: Magic, value: Self):
        instance.__dict__[self.__name__] = value

    def __delete__(self, instance: Magic):
        try:
            del instance.__dict__[self.__name__]
        except KeyError:
            ...

