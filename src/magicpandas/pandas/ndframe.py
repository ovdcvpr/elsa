from __future__ import annotations

import inspect
import numpy as np
import os
import pandas as pd
import pandas.core.common
import pandas.core.generic
import warnings
import weakref
from functools import cached_property
from pandas import DataFrame
from pandas import MultiIndex
from pandas.core.generic import (
    ABCDataFrame,
    is_list_like,
    find_stack_level,
)
from typing import *
from typing import TypeVar

from magicpandas.magic.cached.cached import cached
from magicpandas.magic.cached.sticky import Sticky
from magicpandas.magic.classboundstrings import ClassBoundStrings
from magicpandas.magic.default import default
from magicpandas.magic.magic import Magic
from magicpandas.magic.order import Order
from magicpandas.magic.truthy import truthy
from magicpandas.pandas.metadata import MetaDatas
from magicpandas.pandas.records import Record
from magicpandas.pandas.subloc import SubLoc
from pandas._libs.properties import AxisProperty
from magicpandas.magic.alias import alias
from magicpandas.pandas.foresight.foresights import Foresights
from magicpandas.pandas.foresight.foresight import ForesightFactory

T = TypeVar('T')

if False:
    from .series import Series


def __setattr__(self, name: str, value) -> None:
    """
    If it can be found nowhere else than the info_axis, it must be a column.
    This allows simpler access to columns for interactive use.
    """
    attr = getattr(type(self), name, None)
    if (
            name not in self.__dict__
            and name not in self._internal_names_set
            and name not in self._metadata
            # df.index = ... or df.columns = ... should never set a column
            and not isinstance(attr, AxisProperty)
            and name in self._info_axis
    ):
        try:
            self[name] = value
        except (AttributeError, TypeError):
            pass
        else:
            return

    if (
            # only relevant with dataframes
            isinstance(self, ABCDataFrame)
            # ignore if attr already exists
            and name not in self.__dict__
            # ignore if it's a class attribute;
            # might be a prop, axis, cached_property,
            # or other sort of descriptor
            and not hasattr(type(self), name)
            # ignore if internal or metadata
            and name not in self._internal_names
            and name not in self._metadata
            and is_list_like(value)
    ):
        warnings.warn(
            "Pandas doesn't allow columns to be "
            "created via a new attribute name - see "
            "https://pandas.pydata.org/pandas-docs/"
            "stable/indexing.html#attribute-access",
            stacklevel=find_stack_level(),
        )
    object.__setattr__(self, name, value)


# monkey patch my pandas.DataFrame bugfix for now;
# otherwise DataFrame.__setattr__ causes recursion
# see https://github.com/pandas-dev/pandas/pull/56794
pandas.core.generic.NDFrame.__setattr__ = __setattr__

pandas.core.generic.NDFrame.__getattr__


def apply_if_callable(maybe_callable, obj, **kwargs):
    """
    Evaluate possibly callable input using obj and kwargs if it is callable,
    otherwise return as it is.

    Parameters
    ----------
    maybe_callable : possibly a callable
    obj : NDFrame
    **kwargs
    """
    # modified to also check if is_list_like
    if (
            callable(maybe_callable)
            and not is_list_like(maybe_callable)
    ):
        return maybe_callable(obj, **kwargs)

    return maybe_callable


pandas.core.common.apply_if_callable = apply_if_callable


# todo: we need to be able to call __set__ without another wrapper recursion

# # todo maybe inherit from static instead of magic
# # todo: this is a mess; refactor the whole from_params functionality
def __call__(self, *args, **kwargs) -> Self:
    if (
            self.__from_method__
            and not self.__skip_from_params__
    ):
        skip = self.__skip_from_params__
        self.__skip_from_params__ = True
        result = self.__from_method__(*args, **kwargs)
        # todo: raise exception if method didn't return anything

        # todo: problem is, this is modifying after it is already cached
        self.__skip_from_params__ = skip
        # self.__propagate__(result)
        # result = self.__propagate__(result)
        # return result
        # result = self.__enchant__(result)
        # result = self.__propagate__(result)
        return result
    result = self.enchant(*args, **kwargs)
    return result

# def __call__(self, *args, **kwargs) -> Self:
#     # if (
#     #         self.__from_method__
#     #         and not self.__skip_from_params__
#     # ):
#     #     skip = self.__skip_from_params__
#     #     self.__skip_from_params__ = True
#     #     result = self.__from_method__(*args, **kwargs)
#     #     # todo: raise exception if method didn't return anything
#     #     # todo: problem is, this is modifying after it is already cached
#     #     self.__skip_from_params__ = skip
#     #     return result
#     if self.__from_call__:
#         result = self.__from_call__(*args, **kwargs)
#     else:
#         result = self.enchant(*args, **kwargs)
#     return result


class NDFrame(
    Magic,
    pandas.core.generic.NDFrame,
    from_outer=True,
):
    __inner__: pd.Series | pd.DataFrame | NDFrame | Any
    __inner__: Sticky
    __outer__: Sticky
    # __futures__ = cached.root.property(Magic.__futures__)
    # __threads__ = cached.root.property(Magic.__threads__)
    # __done__ = cached.root.property(Magic.__done__)
    __postinits__ = ClassBoundStrings()
    __init_nofunc__ = pandas.core.generic.NDFrame.__init__
    __direction__ = 'horizontal'
    __sticky__ = False
    subloc = SubLoc()
    __metadatas__ = MetaDatas()
    __order__ = Order.third
    __record__ = Record()
    locals()['__call__'] = __call__
    # __getitem__: Callable[[...], Self]
    # __foresight__ = Foresight
    __foresight__ = ForesightFactory()
    __foresights__ = Foresights()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # concat.__name__ is stored in concat;
        # concat.attrs gets concat.__name__;
        # concat.attrs changes length during concat.attrs.repr()
        # this is to avoid a runtime error
        self._attrs['__name__'] = ''

    def __subget__(self, outer: Sticky, Outer) -> NDFrame:
        from magicpandas.pandas.column import Column
        result: NDFrame
        if outer is None:
            return self

        if self.__configuring__:
            return self
        elif self.__from_method__:
            return self
        elif default.context:
            return self
        if self.__nothird__:
            return self

        owner: NDFrame | Series | DataFrame = self.__owner__
        key = self.__key__
        # cache = owner.__cache__
        cache = owner.__dict__
        if key in cache:
            result = cache[key]
            result.__owner__ = owner
            result.__outer__ = outer
            return result

        tests = False

        # todo problem is we get here, and second gets new owner
        #   todo probably not propagate every time accessing second
        _ = self.__second__
        volatile = self.__volatile__.copy()

        if (
                key in owner.attrs
                and self.__align__
        ):
            # subindex from already existing frame
            result: Self = owner.attrs[key]
            result = result.__align__(owner)

        elif (
                isinstance(owner, pd.DataFrame)
                and isinstance(self, Column)
                and key in owner.columns
        ):
            # noinspection PyTypeChecker
            result = self.enchant(owner[key])

        elif (
                self.__from_file__
                and os.path.exists(self)
        ):
            result = self.__log__(self.__from_file__)
            # result = self.__postprocess__(result)
            tests = True

        elif self.from_outer:
            func = self.from_outer.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            result = self.__log__(func)
            outer.__inner__ = inner
            # result = self.__postprocess__(result)
            if (
                    self.__from_file__
                    and not os.path.exists(self)
            ):
                self.__to_file__(result)
            tests = True

        elif self.conjure:
            # load from inner
            result = self.__log__(self.conjure)
            if (
                    self.__from_file__
                    and not os.path.exists(self)
            ):
                raise NotImplementedError  # should we pass through subset?
                self.__to_file__(result)

            tests = True

        else:
            raise NotImplementedError(
                f'Could not resolve a constructor for {self.__trace__}'
            )

        result = self.__subset__(outer, result)
        result.__volatile__.update(volatile)
        assert isinstance(result, self.__class__)
        # todo: this seems like a suboptimal solution;
        #   the problem is trace is horizontal, while self is 2nd order
        #   and result is 3rd order
        del result.__trace__
        _ = result.__trace__
        if tests:
            result.tests()
        return result

    def __propagate__(self, obj: NDFrame | T) -> T:
        """ set metadata from another object """
        # todo: _ = self.second resets owner?
        if self.__order__ == 2:
            _ = self.__second__
        if obj.__order__ == self.__order__:
            # cache = self.__horizontal__
            cache = self.__directions__.horizontal
        elif obj.__order__ > self.__order__:
            # cache = self.__vertical__
            cache = self.__directions__.vertical
        else:
            raise ValueError(f'obj.order < self.order')
        diagonal = self.__directions__.diagonal
        obj.attrs.update(
            (key, value)
            for key, value in self.attrs.items()
            if key in cache
            or key in diagonal
        )
        obj.__dict__.update(
            (key, value)
            for key, value in self.__dict__.items()
            if key in cache
            or key in diagonal
        )
        obj.__volatile__.update(self.__volatile__)
        return obj

    @property
    def __cache__(self):
        return self.attrs

    @cached.volatile.property
    def __skip_from_params__(self):
        return False

    # def __call__(self, func: T=None, *args, **kwargs) -> Union[Self, T]:
    #     ...

    # 1. propagate from frame to result
    # 2. propagate from result to frame

    def enchant(self, *args, **kwargs) -> Self:
        if args:
            frame = args[0]
            if (
                    isinstance(frame, NDFrame)
                    and frame.__order__ != 3
            ):
                warnings.warn(f"""
                {self.__trace__} is being called on a frame with order 
                {frame.__order__}, which is likely unintended. Are you 
                calling on `self.__outer__` instead of `self.__owner__`?
                """)

            if inspect.isfunction(frame):
                self.__init_func__(frame, *args, **kwargs)
                return self
            result = self.__class__(*args, **kwargs)

            # sticky or same trace
            if (
                    isinstance(frame, NDFrame)
                    and frame.__trace__ == self.__trace__
            ):
                frame.__propagate__(result)
            if not inspect.isfunction(frame):
                self.__propagate__(result)
        else:
            result = self.__class__(*args, **kwargs)
            self.__propagate__(result)
        result.__trace__ = self.__trace__
        return result

    def __flush_references(self) -> Self:
        result = self.copy()
        dropping = {
            key
            for key, value in self.__cache__.items()
            if isinstance(value, weakref.ReferenceType)
        }
        dropping.update(
            key
            for key, value in self.__dict__.items()
            if isinstance(value, weakref.ReferenceType)
        )
        if dropping:
            warnings.warn(
                f'{self.__trace__} is being aligned but contains weakreferences {dropping}'
            )
        result.attrs = {
            key: value
            for key, value in self.attrs.items()
            if not isinstance(value, weakref.ReferenceType)
        }
        result.__dict__ = {
            key: value
            for key, value in self.__dict__.items()
            if not isinstance(value, weakref.ReferenceType)
        }
        return result

    @truthy
    def __align__(self, owner: NDFrame = None) -> Self:
        """ align such that self.index ⊆ outer.index """
        if not self.__align__:
            return self.copy()
        if owner is None:
            owner = self.__owner__
        result = (
            self
            .__subalign__(owner)
            .copy()
        )
        result.__owner__ = owner
        return result

    # todo: we need to modify subalign so that owner is passed;
    #   we cannot use self.__owner__ because the reference might be lost

    def __subalign__(self, owner: NDFrame) -> Self:
        """ align such that self.index ⊆ outer.index """
        haystack: MultiIndex = owner.index
        needles: MultiIndex = self.index
        if (
                isinstance(needles, pd.Index)
                and not needles.name
        ):
            # If index unnamed, assume it is the same as the owner's index
            ...
        elif set(needles.names).intersection(haystack.names):
            # If some index names are shared, align on those
            try:
                names = haystack.names.difference(needles.names)
                haystack = haystack.droplevel(names)
                names = needles.names.difference(haystack.names)
                needles = needles.droplevel(names)
            except ValueError as e:
                raise ValueError(
                    f'Could not align {self.__trace__} with {owner.__trace__};'
                    f' be sure that the index names are compatible.'
                ) from e
        elif (
                isinstance(self, DataFrame)
                and set(haystack.names).intersection(self.columns)
        ):
            # If some columns are in the owner's index names, align on those
            columns = set(haystack.names).intersection(self.columns)
            needles = pd.MultiIndex.from_frame(self[columns])
        else:
            raise NotImplementedError(
                f'Could not resolve how to align {self.__trace__} with '
                f'owner {self.__owner__.__trace__}; you may define how '
                f'to align by overriding {self.__class__}.__align__.'
            )

        loc = needles.isin(haystack)
        result = self.loc[loc]
        return result

    def __subset__(self, outer: NDFrame, value: NDFrame):
        if self.__configuring__:
            raise NotImplementedError
        owner: NDFrame = self.__owner__
        key = self.__key__
        # result = self.__call__(value)
        result = self.enchant(value)

        if self.__align__:
            result = result.__align__(owner)
            owner.attrs[key] = result
        owner.__dict__[key] = result

        for postinit in self.__postinits__:
            if postinit in self:
                continue
            getattr(result, postinit)

        return result

    def __subdelete__(self, outer: NDFrame):
        if self.__configuring__:
            raise NotImplementedError
        owner = self.__owner__
        key = self.__key__
        if key in owner.__dict__:
            del owner.__dict__[key]
        if key in owner.attrs:
            del owner.attrs[key]

    @cached_property
    def _constructor(self):
        return type(self)

    @classmethod
    def from_options(
            cls,
            *,
            postinit=False,
            # log=True,
            log=False,
            from_file=False,
            align=False,
            **kwargs
    ) -> Callable[[T], Union[Self, T]]:
        return super().from_options(
            postinit=postinit,
            log=log,
            from_file=from_file,
            align=align,
            **kwargs
        )

    @alias.property
    def record(self) -> Record:
        return self.__record__

    @alias.property
    def metadatas(self) -> MetaDatas:
        return self.__metadatas__

    @alias.property
    def metadata(self) -> dict:
        return self.__metadata__


    def __init_subclass__(
            cls,
            __call__=False,
            **kwargs
    ):
        super().__init_subclass__(**kwargs)
        if (
                # if programmer did not specifically override this functionality
                not __call__
                # and parameter constructor has been defined
                and '__call__' in cls.__dict__
                # and 0 parameter constructor has not been defined
                # and not cls.conjure
        ):
            """
            use __call__ from bases instead of subclass
            cls.__from_params__.cache[cls] = cls.__call__
            when the user calls the object, it is then instead
            handled by NDFrame.__call__, which then uses self.__from__method__
            to call the parameter-constructor that the user defined.
            """
            cls.__from_method__.inner = cls.__call__
            for base in cls.__bases__:
                if hasattr(base, '__call__'):
                    # cls.__call__ = base.__call__
                    setattr(cls, '__call__', base.__call__)
                    break
            else:
                raise NotImplementedError

        # incase we are inheriting, where the super class
        # uses conjure, but we use call
        # if (
        #     '__conjure__' in cls.__dict__
        #     and not cls.__dict__['__conjure__']
        # ):
        #     cls.__from_method__ = False

    # @cached.static.property
    @property
    def irange(self) -> np.ndarray:
        """An array of integers from 0 to len(self)"""
        return np.arange(len(self))

    # @cached.static.property
    @property
    def __original__(self):
        original = self
        while original._is_copy:
            original = original._is_copy()
        return original

    @classmethod
    def from_parquet(cls, file) -> Self:
        return pd.read_parquet(file).pipe(cls)

    @property
    def softcopy(self) -> Self:
        """If _is_copy, return .copy(deep=False), else return just self"""
        if self._is_copy is None:
            return self
        else:
            return self.copy(deep=False)

    def __getstate__(self):
        # there are weakrefs in the frame's attrs; clear these or pickling will fail
        result = super().__getstate__()
        result['attrs'].clear()
        return result

    @alias.property
    def foresight(self) -> ForesightFactory:
        return self.__foresight__

    @alias.property
    def foresights(self) -> Foresights:
        return self.__foresights__

    @final
    def __finalize__(self, other, method: str | None = None, **kwargs) -> Self:
        """
        Propagate metadata from other to self.

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate
        method : str, optional
            A passed method name providing context on where ``__finalize__``
            was called.

            .. warning::

               The value passed as `method` are not currently considered
               stable across pandas releases.
        """
        if isinstance(other, NDFrame):
            for name in other.attrs:
                self.attrs[name] = other.attrs[name]

            self.flags.allows_duplicate_labels = other.flags.allows_duplicate_labels
            # For subclasses using _metadata.
            for name in set(self._metadata) & set(other._metadata):
                assert isinstance(name, str)
                object.__setattr__(self, name, getattr(other, name, None))

        if method == "concat":
            allows_duplicate_labels = all(
                x.flags.allows_duplicate_labels
                for x in other.objs
            )
            self.flags.allows_duplicate_labels = allows_duplicate_labels

        return self
