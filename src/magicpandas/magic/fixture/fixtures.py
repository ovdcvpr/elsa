from __future__ import annotations
# from magicpandas.magic.cached.magic import Magic
from magicpandas.magic.magic import Magic
from magicpandas.magic.fixture.traces import Traces
from magicpandas.magic.fixture.fixture import Fixture

"""
Fixtures:
    traverses all nested Magic
"""

class Fixtures(Magic):
    __fixture_traces__ = Traces()
    def __call__(self, *args, **kwargs):
        """Primes all the fixtures that are yielded"""
        for fixture in self:
            if not isinstance(fixture, Fixture):
                continue
            fixture(*args, **kwargs)

    def __iter__(self):
        for trace in self.__fixture_traces__:
            obj = self
            for attr in trace.split('.'):
                obj = getattr(obj, attr)
            yield obj






