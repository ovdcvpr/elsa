# todo:
"""

@norme.register
def norme(self)
    ...

@norme.register(thing=list)
def norme(self, thing: list | str):
    ...

@magic.column
def col(self) -> magic[str]
    # tries default

@col.register
def col(self) -> magic[str]:
    # if previous raised AttributeErrore tries this

@col.register
def col(self) -> magic[str]:
    # if previous raised AttributeError tries this


@magic.frame.register(hello=str)
def func(self, hello: str) -> magic[str]:
    ...

@magic.frame.register(hello=list)
def func(self, hello: list) -> magic[str]:
    ...

@magic.frame.register(hello=str)
def func(self, hello: str) -> magic[str]:
    # if previous raised AttributeError tries this



__registers__: list[function]  = magic.cached.diagonal.property

@norme.register
def norme(self):
    return self.normx.values + self.normwidth.values / 2

@norme.register
def norme(self):
    return self.normw.values + self.normwidth.value

@norme.register
def norme(self):
    return self.fe.values - self.nfile


@magic.register(param=int)
def __call__(self, param):
    ...

@magic.register(param=str)
def __call__(self, param):
    ...

@magic.register(param=(str, int))


def register(func=None, **kwargs):
    if kwargs:
        ...
    else
        ...
"""
