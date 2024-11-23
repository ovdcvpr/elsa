from __future__ import annotations

if False:
    from magicpandas.pandas.frame import Frame
class DefinedColumns:
    """
    A set of all the defined magic.columns for the given owner
    """
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__cache__: dict[type, set[str]] = {}


    def __get__(self, instance: Frame, owner: type[Frame]):
        from magicpandas.pandas.frame import Frame
        from magicpandas.pandas.column import Column
        cache = self.__cache__
        name = self.__name__
        if owner not in cache:
            result = {
                key
                for base in owner.__bases__
                if issubclass(base, Frame)
                for key in getattr(base, name)
            }
            result.update(
                key
                for key, value in owner.__dict__.items()
                if isinstance(value, Column)
            )
            cache[owner] = result
        result = cache[owner]
        return result



