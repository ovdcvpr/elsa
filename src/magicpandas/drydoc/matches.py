from __future__ import annotations

import numpy as np
import pandas as pd
from typing import *
import magicpandas as magic
from magicpandas.magic.drydoc.drydoc import DryDoc

if False:
    from magicpandas.drydoc.objects import Objects


class Matches(magic.Frame):
    outer: Objects
