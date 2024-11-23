from __future__ import annotations
from os import PathLike

from typing import *

from magicpandas import util
from magicpandas.logger.logger import logger
from magicpandas.magic.cached.cached import cached
from magicpandas.magic.default import default
from magicpandas.magic.fixture.fixture import Fixture
from magicpandas.magic.fixture.fixtures import Fixtures
from magicpandas.magic.imports import imports
from magicpandas.magic.local.locals import Locals
from magicpandas.magic.log import log
from magicpandas.magic.magic import Magic
from magicpandas.magic.alias import alias
from magicpandas.magic.test.test import Test
from magicpandas.magic.test.test import test
from magicpandas.magic.truthy import truthy
from magicpandas.pandas.column import Column
from magicpandas.pandas.column import column
from magicpandas.pandas.frame import Frame
from magicpandas.pandas.frame import frame
from magicpandas.pandas.index import Index
from magicpandas.pandas.index import index
from magicpandas.pandas.series import Series
from magicpandas.pandas.series import series
from magicpandas.util import LazyModuleLoader
from magicpandas.pandas.foresight.foresight import foresight

if False:
    from magicpandas.pandas import geo
locals()['geo'] = LazyModuleLoader('magicpandas.pandas.geo')

if False:
    pass
locals()['graph'] = LazyModuleLoader('magicpandas.graph')

if False:
    pass
locals()['raster'] = LazyModuleLoader('magicpandas.raster')


def __getitem__(*args, **kwargs):
    # allows def func() -> magic[int]
    ...

import warnings

warnings.filterwarnings("ignore", message=".*You are adding a column named 'geometry' to a GeoDataFrame.*")


T = TypeVar('T')


def delayed(func: T = None, *args) -> T:
    return func


def _portal(func: T) -> T:
    return func

def portal(file: object | PathLike[str] | PathLike[bytes]):
    return _portal
