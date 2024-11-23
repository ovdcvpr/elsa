from magicpandas.magic.cached.sticky import Sticky


class Static(Sticky):
    @property
    def __owner_cache__(self) -> dict:
        return self.__owner__.__dict__

# locals()['property'] = Static

property = Static