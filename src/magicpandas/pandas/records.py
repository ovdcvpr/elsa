from __future__ import annotations

from functools import *

from magicpandas import util 


if False:
    from magicpandas.pandas.ndframe import NDFrame


class Record:
    @util.weakly.cached_property
    def __instance__(self) -> NDFrame:
        """The instance that owns this record"""

    @cached_property
    def __owner__(self):
        """The class associated with this record"""
        return self.__owner__

    def __set_name__(self, owner, name):
        self.__name__ = name

    @property
    def __last__(self) -> dict[str, NDFrame]:
        """
        Stores the last record as a string, value dict to aid in easy
        access to the last record associated with a particular key.
        """
        cache = self.__instance__.__cache__
        key = f'{self.__name__}.__last__'
        if key not in cache:
            cache[key] = {}
        return cache[key]

    @property
    def __records__(self) -> list[tuple[str, NDFrame]]:
        """
        Stores the records as a list in the NDFrame's attrs. So that
        previous instances do not have their records wrongfully modified,
        each new instance's records is a copy of its parent's records.
        """
        cache = self.__instance__.__cache__
        key = f'{self.__name__}'
        if key not in cache:
            cache[key] = []
        records: list = cache[key]
        cache = self.__instance__.__dict__
        if key not in cache:
            cache[key] = records.copy()
        records = cache[key]
        return records

    def __get__(self, instance: NDFrame, owner):
        self.__instance__ = instance
        self.__owner__ = owner
        return self

    def __call__(self):
        """self.record() appends a new record with the instance's trace"""
        instance = self.__instance__
        trace = self.__instance__.__trace__.__str__()
        record = (trace, instance)
        self.__records__.append(record)
        self.__last__[trace] = instance

    def __getitem__(self, item):
        """
        self.record[0] returns last record
        self.record['name'] returns last record with the key 'name'
        """
        if isinstance(item, str):
            return self.__last__[item]
        elif isinstance(item, int):
            return self.__records__[-item]
        else:
            raise TypeError(f'unsupported type {type(item)}')

    def __setitem__(self, key, value):
        """
        self.record['name'] = ... appends a new record
        """
        if not isinstance(key, str):
            raise TypeError(f'unsupported type {type(key)}')
        self.__records__.append((key, value))
        self.__last__[key] = value

    def __delitem__(self, key):
        """
        del self.record[0] deletes the last record
        del self.record['name'] deletes the last record with the key 'name'
        """
        if isinstance(key, str):
            del self.__last__[key]
            for string, value in self.__records__:
                if string == key:
                    self.__records__.remove((string, value))
                    break
        elif isinstance(key, int):
            string, value = self.__records__[-key]
            del self.__records__[-key]
            if value is self.__last__[string]:
                del self.__last__[string]
            for key, value in self.__records__:
                if key == string:
                    self.__last__[key] = value

    def __iter__(self):
        yield from reversed(self.__records__)

    def __repr__(self):
        """Print the records, one per line"""
        return '\n'.join(map(str, self.__records__[::-1]))
