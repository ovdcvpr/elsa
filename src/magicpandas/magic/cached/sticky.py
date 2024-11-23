
from __future__ import annotations

from magicpandas.magic.order import Order
import magicpandas.magic.magic as magic




class Sticky(magic.Magic):
    __order__ = Order.third


property = Sticky