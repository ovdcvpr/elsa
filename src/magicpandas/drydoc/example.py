from magicpandas.magic.magic import Magic
from magicpandas.magic.drydoc import drydoc


class DryDoc(drydoc.DryDoc):

    def parent(self):
        """drydoc"""

    def child(self):
        """drydoc"""

    def Parent(self):
        """drydoc"""


class Parent(Magic):
    """magic"""
    drydoc = DryDoc

    def parent(self):
        """magic"""

    def ignore(self):
        ...


class Child(Parent):
    """magic"""
    drydoc = DryDoc

    def child(self):
        """magic"""

    def ignore(self):
        ...


class OtherParent:

    def parent(self):
        """magic"""


class OtherChild(OtherParent):
    """magic"""

    def child(self):
        """MAGIC"""


def method():
    """magic"""


def func():
    ...
