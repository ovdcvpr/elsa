from __future__ import annotations

from magicpandas.magic.cached.base import Base
from magicpandas.magic.cached.cmdline import CmdLine
from magicpandas.magic.cached.diagonal import Diagonal
from magicpandas.magic.cached.sticky import Sticky
from magicpandas.magic.cached.outer import Outer
from magicpandas.magic.cached.outer2 import  Outer as Outer2
from magicpandas.magic.cached.public import Public
from magicpandas.magic.cached.root import Root
from magicpandas.magic.cached.volatile import Volatile
from magicpandas.magic.local.local import Local
from magicpandas.magic.cached.serialized import Serialized
from magicpandas.magic.cached.static import Static
from magicpandas.magic.cached.weak import Weak


class cached:
    class base:
        property = Base

    class cmdline:
        property = CmdLine

    class diagonal:
        property = Diagonal

    class local:
        property = Local

    class outer:
        # property = Outer
        property = Outer2

    class public:
        property = Public

    class root:
        property = Root

    class serialized:
        property = Serialized

    class static:
        property = Static

    class sticky:
        property = Sticky

    class volatile:
        property = Volatile

    class weak:
        property = Weak
