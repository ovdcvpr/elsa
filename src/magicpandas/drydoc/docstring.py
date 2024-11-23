from __future__ import annotations

import magicpandas as magic
from magicpandas.drydoc.resource import Resource


class DocString(
    Resource,
    magic.Frame
):
    """
    docstring.magic:
        DocString from the magic;
        if object is None, it is the cls docstring
        else, it is the object docstring
    docstring.drydoc:
        docstring from the DryDoc for the magic's name, if it exists.
    """

    @magic.column
    def drydoc(self):
        drydocs = self.magics.drydoc
        names = self.magics.name
        result = []
        it = zip(drydocs, names)
        for drydoc, name in it:
            try:
                method = getattr(drydoc, name)
            except AttributeError:
                result.append(None)
            else:
                doc = method.__doc__
                result.append(doc)

        return result

    @magic.column
    def magic(self):
        magics = self.magics
        result = []
        it = zip(magics.cls, magics.object)
        for cls, obj in it:
            if obj is None:
                doc = cls.__doc__
            else:
                doc = obj.__doc__
            result.append(doc)

        return result
