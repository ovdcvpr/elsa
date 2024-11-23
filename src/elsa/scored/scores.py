

from __future__ import annotations

import magicpandas as magic
from elsa.prediction.scores import Selection

if False:
    from elsa.scored.scored import Scored

class Selection(magic.Magic):
    outer: Scores

    @magic.column
    @magic.portal(Selection.loglse)
    def loglse(self):
        ...

    @magic.column
    @magic.portal(Selection.nlse)
    def nlse(self):
        ...

    @magic.column
    @magic.portal(Selection.argmax)
    def argmax(self):
        ...


class Selected(Selection):
    ...



class Scores(magic.Magic):
    outer: Scored
    whole = Selection()
    selected = Selected()
