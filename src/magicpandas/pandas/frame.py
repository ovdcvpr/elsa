from __future__ import annotations

import json
import pandas
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame
from pandas._typing import FilePath, WriteBuffer
from typing import *

from magicpandas import util
from magicpandas.magic.cached.abc import CachedABC
from magicpandas.magic.cached.cached import cached
from magicpandas.magic.classboundstrings import ClassBoundStrings
from magicpandas.magic.order import Order
from magicpandas.magic.alias import alias
from magicpandas.magic.truthy import truthy
from magicpandas.pandas.column import Column
from magicpandas.pandas.defined_columns import DefinedColumns
from magicpandas.pandas.ndframe import NDFrame

T = TypeVar('T')


# def set_index(self: Frame, *args, **kwargs):
#     # if it failed, try to dynamically load each potential column
#     try:
#         return super(Frame, self).set_index(*args, **kwargs)
#     except KeyError:
#         keys = kwargs.get('keys', args[0])
#         _ = self[keys]
#         return super(Frame, self).set_index(*args, **kwargs)
#

class Frame(
    NDFrame,
    pandas.DataFrame,
):
    __init_nofunc__ = pandas.DataFrame.__init__
    __flush__ = ClassBoundStrings()
    __defined_columns__ = DefinedColumns()

    def __fspath__(self):
        # cwd/magic/magic.pkl
        return self.__directory__ + '.parquet'

    @truthy
    def __from_file__(self):
        return pandas.read_parquet(self)

    def __to_file__(self, value=None):
        if value is None:
            value = self
        future = self.__root__.__threads__.submit(self.to_parquet, value)
        self.__root__.__futures__.append(future)

    @cached.sticky.property
    def __permanent__(self) -> set[str]:
        return set()

    @cached.sticky.property
    def __columns__(self) -> set[str]:
        return set()

    def __flush_columns__(self, columns: str | list[str] = None) -> Self:
        result = self.copy()
        permanent = self.__permanent__
        __columns__ = self.__columns__
        if columns is None:
            columns = self.columns
        elif isinstance(columns, str):
            columns = {columns}
        else:
            columns: set[str] = set(columns)
        # todo: only delete ones that are magic columns

        for column in columns:
            # todo: x permanent is not properly propagating
            try:
                self.__getnested__(column)
            except AttributeError:
                continue
            if (
                    column in __columns__
                    and column not in permanent
            ):
                del result[column]

        return result

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
                result += f'{pandas.DataFrame.__repr__(self)}'

        return result

    def __eq__(self, other):
        # temporary solution to an annoying problem:
        # pd.concat wants to do this:
        # check_attrs = all(objs.attrs == attrs for objs in other.objs[1:])
        # when we have magic frames cached in the attrs, even if weakrefed
        # as of python 3.10, they are compared, and this raises an exception
        # todo: add this to base magicpandas
        if isinstance(other, (pd.DataFrame, pd.Series)):
            return False
        return super().__eq__(other)

    def to_parquet(
            self,
            path: FilePath | WriteBuffer[bytes] | None = None,
            *args,
            **kwargs,
    ) -> bytes | None:
        result = super().to_parquet(
            path=path,
            *args,
            **kwargs,
        )
        table = pq.read_table(path)
        new = {
            attr: getattr(self, attr)
            for attr in self.__metadatas__
        }
        old = table.schema.metadata or {}
        combined = {
            **old,
            **{'magic': new}
        }
        schema = table.schema.with_metadata(combined)
        pq.write_table(table, path, schema=schema)
        return result

    @cached.sticky.property
    def __passed__(self) -> Optional[str]:
        """The path to the file that was passed"""
        return None

    @alias.property
    def passed(self):
        return self.passed

    @cached.sticky.property
    def __metadata__(self) -> Optional[dict[str]]:
        if self.passed is None:
            return None
        match str(self.passed).split('.')[-1]:
            case 'parquet':
                result = (
                    pq
                    .ParquetFile(self.passed)
                    .metadata.metadata
                )
                try:
                    byte_string = result[b'magic']
                    # Convert bytestring to string
                    json_string = byte_string.decode('utf-8')
                    result = json.loads(json_string)
                except KeyError:
                    result = {}
            case _:
                raise NotImplementedError
        return result

    @alias.property
    def metadata(self):
        return self.__metadata__

    def indexed_on(
            self,
            loc=None,
            name: str | list[str] = None,
    ) -> Self:
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

        if isinstance(name, str):
            column: pd.Series = getattr(self, name)
            index = pd.Index(column)
        elif isinstance(name, list):
            try:
                columns: pd.DataFrame = self[name]
            except KeyError:
                columns: pd.DataFrame = self.reset_index()[name]
            index = pd.MultiIndex.from_frame(columns)
        else:
            raise TypeError(f'Expected str or list[str], got {type(name)}.')

        # if loc is not None:
        #     iloc = index.get_indexer_for(target=loc)
        #     index = index[iloc]
        #     result = self.iloc[iloc]
        # result = result.set_axis(index, axis=0)
        iloc = index.get_indexer_for(target=loc)
        result = self.iloc[iloc]
        return result

    # def __getitem__(self, item) -> Union[Self, pd.Series]:
    #     """
    #     If it failed, the columns may be magic; access them through
    #     attribute access and then try again.
    #     """
    #     try:
    #         return super().__getitem__(item)
    #     except KeyError as e:
    #         # if it fails, try to dynamically load each potential column
    #         cols = item
    #         if (
    #                 isinstance(item, str)
    #                 or isinstance(item, tuple)
    #                 or not isinstance(item, Iterable)
    #         ):
    #             cols = item,
    #         for col in cols:
    #             try:
    #                 util.getattrs(self, col)
    #             except AttributeError:
    #                 raise e
    #     return super().__getitem__(item)

    def __getitem__(self, item) -> Union[Self, pd.Series]:
        """
        If it failed, the columns may be magic; access them through
        attribute access and then try again.
        """
        try:
            return super().__getitem__(item)
        except KeyError as k1:
            # if it fails, try to dynamically load each potential column
            cols = item
            if (
                    isinstance(item, str)
                    or isinstance(item, tuple)
                    or not isinstance(item, Iterable)
            ):
                cols = item,
            for col in cols:
                obj = self
                try:
                    for attr in str.split(col, '.'):
                        obj = getattr(obj, attr)
                except Exception as e2:
                    raise k1
        except pandas.errors.InvalidIndexError as e:
            """
            If slice(None) is first, return self[self.columns, *item[1:]]
            If slice(None) is last, return self[item[:-1], self.columns]
            
            frame['col1', :]
            frame[:, 'col2']
            frame['col1', :, 'col2']
            """
            expanders = [
                i
                for i, item in enumerate(item)
                if (
                        item == slice(None)
                        or item is ...
                )
            ]
            if len(expanders) != 1:
                raise e
            i = expanders[0]
            columns = [
                *item[:i],
                *self.columns,
                *item[i + 1:],
            ]
        return super().__getitem__(item)

    def itercolumns(self, values=False) -> Iterator[pd.Series]:
        if values:
            yield from (
                self[column].values
                for column in self.columns
            )
        else:
            yield from (
                self[column]
                for column in self.columns
            )

    # def prepare(self, *columns: str) -> Self:
    #     """
    #     Ensure that the required columns are computed and present in the
    #     DataFrame. Rather than computing them in the current frame, it
    #     computes them in the original frame and returns and returns the
    #     current subset, with the prepared columns added.
    #     """
    #     old = self.columns
    #     original = self.__original__
    #
    #     for col in columns:
    #         if col not in original:
    #             getattr(original, col)
    #
    #     new = original[list(columns)].columns
    #     result = original.loc[self.index, new.union(old)]
    #     return result

    # @property
    # def requires(self) -> __requires__:
    #     return self.__requires__
    #
    # requires: requires

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

    def __setitem__(self, key, value):
        if key == 'index':
            ...
        super(Frame, self).__setitem__(key, value)

    def __set_index(self: Frame, *args, **kwargs):
        # if it failed, try to dynamically load each potential column
        try:
            return super(Frame, self).set_index(*args, **kwargs)
        except KeyError:
            if 'keys' in kwargs:
                keys = kwargs['keys']
            else:
                keys = args[0]
            _ = self[keys]
            return super(Frame, self).set_index(*args, **kwargs)

    locals()[DataFrame.set_index.__name__] = __set_index

    def assign(self, **kwargs) -> Self:
        """
        Extends DataFrame.assign, so that if any keys exist as Magic
        for the class, other than magic.Column, they are simply assigned,
        rather than set as columns.
        """
        cls = self.__class__
        candidates: Iterator[object] = (
            getattr(cls, key, None)
            for key in kwargs
        )
        it_setter = (
            candidate is not None
            and isinstance(candidate, CachedABC)
            and not isinstance(candidate, Column)
            for candidate in candidates
        )
        it = zip(kwargs.keys(), kwargs.values(), it_setter)
        properties = {
            key: value
            for key, value, property_like in it
            if property_like
        }
        columns = {
            key: value
            for key, value in kwargs.items()
            if key not in properties
        }
        result = super().assign(**columns)
        # todo: how do we determine
        for key, value in properties.items():
            setattr(result, key, value)
        return result


class frame(Frame):
    ...


locals()['frame'] = frame

