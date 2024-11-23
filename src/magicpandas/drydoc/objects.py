from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
from pandas import Series
from typing import *

import magicpandas as magic
import magicpandas.drydoc.util as util
from magicpandas.drydoc.docstring import DocString
from magicpandas.drydoc.write import Write
from magicpandas.magic.magic import Magic
from magicpandas.drydoc.old import Old


class Objects(magic.Frame):

    @magic.column
    def cls(self) -> magic[type[Magic]]:
        ...

    @magic.column
    def drydoc(self):
        cls: Magic
        result = np.fromiter((
            cls.drydoc
            for cls in self.cls
        ), dtype=object, count=len(self))
        result = pd.Categorical(result)
        return result

    @magic.column
    def name(self) -> magic[str]:
        result = []
        it = zip(self.cls, self.object)
        for cls, obj in it:
            if obj is None:
                name = cls.__name__
            else:
                name = obj.__name__
            result.append(name)
        return result

    @magic.column
    def object(self) -> magic[Callable]:
        """The object for which the docstring is to be checked"""

    @DocString
    def docstring(self):
        """
        docstring.magic:
            DocString from the magic;
            if object is None, it is the cls docstring
            else, it is the object docstring
        docstring.drydoc:
            docstring from the DryDoc for the magic's name, if it exists.
        """

    @Old
    def old(self):
        """
        DataFrame encapsulating the old, or previous docstrings
        associated with magics.
        """

    # @magic.column
    # def old(self) -> Series[str]:
    #     ...

    @Write
    def write(self):
        """"""

    @classmethod
    def from_files(cls, files: list[str]) -> Self:
        """Each class has 1 methodless, and 0+ methodful rows."""
        list_magics: list[type[Magic]] = []
        list_objects: list[Optional[Callable]] = []

        for file in files:
            module = util.module(file)
            MAGICS: list[type[Magic]] = [
                obj
                for obj in module.__dict__.values()
                if inspect.isclass(obj)
                   and issubclass(obj, Magic)
                   and inspect.getfile(obj) == file
            ]
            for magic in MAGICS:
                list_magics.append(magic)
                list_objects.append(None)

                def criteria(obj):
                    """
                    An object meets the criteria if it is a function
                    with a docstring, or wraps a function with a
                    docstring.
                    """
                    try:
                        if not inspect.getfile(obj) == file:
                            return False
                    except TypeError:
                        return False
                    if inspect.isfunction(obj):
                        return isinstance(obj.__doc__, str)
                    wrapped = getattr(obj, '__wrapped__', None)
                    if wrapped:
                        return isinstance(wrapped.__doc__, str)
                    return False

                objects = [
                    obj
                    for obj in magic.__dict__.values()
                    if criteria(obj)
                ]
                list_magics.extend([magic] * len(objects))
                list_objects.extend(objects)

        result = cls({
            'cls': list_magics,
            'object': list_objects,
        })
        return result


if __name__ == '__main__':
    path = '/home/redacted/PycharmProjects/sirius/src/magicpandas/drydoc/example.py'
    # from magicpandas.drydoc import util
    #
    # module = util.module(path)
    # file = inspect.getfile(module.Parent)
    # inspect.getsourcefile(module.Parent)
    # file

    # todo: cached outer not working?
