from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import textwrap
import warnings
from typing import *
from typing import TypeVar

from magicpandas.magic.cached.cached import cached
from magicpandas.magic.magic import Magic
from magicpandas.pandas.ndframe import NDFrame
from magicpandas.pandas.series import Series

T = TypeVar('T')

if False:
    from .frame import Frame


class Column(
    # Series.Blank
    Series
):
    __init__ = Series.__init__

    def __subget__(self: Column, outer: Frame, Outer) -> Column:
        # assert self.__owner__.__order__ == 3
        owner = self.__owner__
        key = self.__key__

        if self.__permanent__:
            owner.__permanent__.add(key)
        owner.__columns__.add(key)
        test = False

        if key in owner._item_cache:
            # noinspection PyTypeChecker
            # return owner._item_cache[key]
            result = owner._item_cache[key]
            result.__owner__ = owner
            result.__outer__ = outer
            return result
        if key in owner:
            try:
                result = owner._get_item_cache(key)
            except TypeError as e:
                # TypeError: only integer scalar arrays can be converted to a scalar index
                # During handling of the above exception, another exception occurred:
                result = owner[key]
            if owner._is_copy:
                result = self.enchant(result)
                owner._item_cache[key] = result

                return result
            if not isinstance(result, self.__class__):
                result = self.__subset__(owner, result)
        elif self.from_outer:
            func = self.from_outer.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            with self.__recursion__():
                result = self.__log__(func)
            outer.__inner__ = inner
            result = self.__subset__(owner, result)
            test = True
        elif self.conjure:
            with self.__recursion__():
                result = self.__log__(self.conjure)
            result = self.__subset__(owner, result)
            test = True
        else:
            msg = (
                f'Could not resolve a constructor for {self.__trace__}. '
                f'If attempting to lazy-compute the object, please '
                f'assure the method returns something. Otherwise, the '
                f'object is being get before it is set.'
            )
            index = owner.index
            if (
                key == index.name
                or key in index.names
            ):
                msg = (
                    f'Cannot get the column {key} for {self.__trace__} '
                    f'However the column name has been found in the index names. '
                    f'You may want to use @magic.index instead of @magic.column'
                )
            # if key in owner.index.name
            raise NotImplementedError(msg)

        del result.__trace__
        _ = result.__trace__
        if test:
            result.tests()
        return result

    def __subset__(self, owner, value):
        key = self.__key__

        warnings.simplefilter('error', pd.errors.SettingWithCopyWarning)
        try:
            owner[key] = value
            # owner.attrs[key] = value
        except pd.errors.SettingWithCopyWarning as e:
            stack = inspect.stack()
            frame = stack[4]
            file_path = frame.filename
            line_number = frame.lineno
            msg = textwrap.dedent(f"""
            likely raised at {file_path}:{line_number}:
            
            {self.__trace__} is being set to a {self.__class__.__name__} 
            which is a copy of a slice. Either precompute the column for 
            the original Frame using `_ = frame.{key}`, or create a new 
            copy using `frame = frame.copy()`.
            """)
            warnings.simplefilter('default', pd.errors.SettingWithCopyWarning)
            warnings.warn(msg, pd.errors.SettingWithCopyWarning)

        except pd.errors.SettingWithCopyError as e:
            msg = textwrap.dedent(f"""
            {self.__trace__} is being set to a {owner.__class__.__name__} 
            which is a copy of a slice. Either precompute the column for 
            the original Frame using `_ = frame.{key}` or create a new 
            copy using `frame = frame.copy()`.
            """)

            new = pd.errors.SettingWithCopyError(msg)
            raise new from e

        sliced, owner._sliced_from_mgr = owner._sliced_from_mgr, self._from_mgr

        try:
            result = owner._get_item_cache(key)
        except TypeError:
            # i think this occurs when index is a RangeIndex?
            result = owner[key]
        # todo: this should be enchant
        result = self.enchant(result, index=owner.index, name=self.__key__, dtype=self.__dtype__)
        owner._item_cache[key] = result
        owner._sliced_from_mgr = sliced
        return result

    def __subdelete__(self, instance):
        super().__subdelete__(instance)
        try:
            del self.__owner__[self.__key__]
        except KeyError:
            ...

    @cached.base.property
    def __flush__(self):
        return True

    @cached.diagonal.property
    def __permanent__(self):
        """
        If True, calling del on the attr will do nothing; this is
        for when the column should not be flushed by flush_columns
        """
        return False

    @cached.diagonal.property
    def __postinit__(self):
        """
        If True, the column will be initialized after the initialization
        of the owner, rather than needing to be accessed first.
        """
        return False

    @classmethod
    def from_options(
            cls,
            *,
            log=False,
            # log=True,
            from_file=False,
            align=False,
            dtype=None,
            permanent=False,
            postinit=False,
            no_recursion=False,
            **kwargs,
    ) -> Callable[[T], Union[T, Self]]:
        result: Self = super().from_options(
            log=log,
            from_file=from_file,
            align=align,
            dtype=dtype,
            permanent=permanent,
            postinit=postinit,
            no_recursion=no_recursion,
            **kwargs,
        )
        return result

    def __set_name__(self, owner: type[Magic], name):
        super().__set_name__(owner, name)
        if self.__postinit__:
            if not issubclass(owner, NDFrame):
                raise ValueError(
                    f"Currently {Column.__name__}.__postinit__ is only "
                    f"supported for {NDFrame.__name__} subclasses. "
                    f"Owner is {owner.__class__.__name__}."
                )
            owner.__postinits__.add(name)

    def indexed_on(
            self,
            loc,
            name: str | list[str] = None,
            fill_value=None,
    ) -> Self:
        # todo: promote more frequent use
        #   is Series.reindex the same thing?
        if name is None:
            if isinstance(loc, (pd.Series, pd.Index)):
                name = loc.name
            elif isinstance(loc, pd.DataFrame):
                name = loc.columns
            elif isinstance(loc, pd.MultiIndex):
                name = loc.names
            else:
                msg = f'Cannot infer name from {type(loc)}.'
                raise ValueError(msg)
        elif isinstance(name, (pd.Series, pd.Index)):
            name = name.name
        elif isinstance(name, pd.MultiIndex):
            name = name.names
        elif isinstance(name, pd.DataFrame):
            name = name.columns
        elif isinstance(name, str):
            ...
        else:
            raise TypeError(f'Expected str or list[str], got {type(name)}.')

        owner = self.__owner__
        if isinstance(name, str):
            column: pd.Series = getattr(owner, name)
            axis = pd.Index(column)
        elif isinstance(name, list):
            try:
                columns: pd.DataFrame = owner[name]
            except KeyError:
                columns: pd.DataFrame = owner.reset_index()[name]
            axis = pd.MultiIndex.from_frame(columns)
        else:
            raise TypeError(f'Expected str or list[str], got {type(name)}.')

        result = (
            self
            .set_axis(axis, axis=0)
            .reindex(loc, fill_value=fill_value)
        )
        return result

    @property
    def iunique(self):
        index = self.unique()
        result = (
            pd.Series(np.arange(len(index)), index)
            .loc[self.values]
            .set_axis(self.index, axis=0)
        )
        return result

class column(Column):
    ...



locals()['column'] = Column
