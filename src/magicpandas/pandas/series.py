from __future__ import annotations

import pandas
import warnings
import weakref
from pandas import DataFrame
from typing import *

from magicpandas.magic.cached.cached import cached
from magicpandas.magic.order import Order
from magicpandas.pandas.ndframe import NDFrame

if False:
    from .frame import Frame


class Series(
    # NDFrame.Blank,
    NDFrame,
    pandas.Series,
):
    __init_nofunc__ = pandas.Series.__init__

    @classmethod
    def from_options(
            cls,
            *,
            log=True,
            from_file=False,
            align=False,
            dtype=None,
            **kwargs,
    ):
        return super().from_options(
            log=log,
            from_file=from_file,
            align=align,
            dtype=dtype,
            **kwargs,
        )

    def __subset__(self, outer: NDFrame, value):
        # instead of __propagate__, uses __call__
        if self.__configuring__:
            raise NotImplementedError
        owner: NDFrame = self.__owner__
        key = self.__key__
        if (
                isinstance(value, NDFrame)
                and value.__order__ != 3
        ):
            warnings.warn(f"""
            {self.__trace__} is being set to a frame with order 
            {value.__order__}, which is likely unintended. Are you 
            setting to `self.__owner__` instead of `self.__outer__`?
            """)
        result = self.enchant(value, dtype=self.__dtype__, name=self.__key__)
        result = result.__align__(owner)
        result.__trace__ = self.__trace__
        owner.__dict__[key] = result
        owner.attrs[key] = weakref.ref(result)
        del result.__trace__
        _ = result.__trace__
        # result.__third__ = result

        return result

    def __subdelete__(self, instance: Frame):
        super().__subdelete__(instance)
        owner = self.__owner__
        key = self.__key__.__str__()
        if isinstance(owner, DataFrame):
            try:
                del instance[key]
            except KeyError:
                ...

    # @lru_cache()
    def __repr__(self):
        result = self.__trace__.__str__()
        match self.__order__:
            case Order.first:
                result += ' 1st order'
            case Order.second:
                result += ' 2nd order'
            case Order.third:
                if result:
                    result += '\n'
                result += f'{pandas.Series.__repr__(self)}'

        return result

    @cached.diagonal.property
    def __dtype__(self):
        """ The Dtype to assign to the Series, if desired. """
        return None

    if False:
        """
        Here loc and iloc do not return self, they return _LocIndexer
        and _iLocIndexer reprectively. However, there is no way to
        express the fact that the indexer's getitem returns an instance
        of the owning class. The only way to preserve this is to
        annotate that loc returns Self, and Self.__getitem__ returns Self
        """

        @property
        def loc(self) -> Union[Self, _iLocIndexer]:
            ...

        @property
        def iloc(self) -> Union[Self, _iLocIndexer]:
            ...


class series(Series):
    ...


locals()['series'] = Series
