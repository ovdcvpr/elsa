from __future__ import annotations

import functools
import importlib
import inspect
import warnings
from functools import *
from typing import *
from typing import TypeVar

from magicpandas.magic.delayedimport import DelayedImport
from magicpandas.magic.options import Options
from magicpandas.magic.order import Order
from magicpandas.magic.setter import __setter__
from magicpandas.magic.trace import Trace
from magicpandas.magic.wrap_descriptor import __wrap_descriptor__
from magicpandas.magic.cached.abc import CachedABC

T = TypeVar('T')

if False:
    from .magic import Magic


class delayed_import:
    method = None

    @classmethod
    def from_params(cls, func: T) -> T:
        result = delayed_import(func)
        result.method = 'from_params'
        return result

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, instance, owner):
        name = self.__name__
        *module, cls = (
            inspect.get_annotations(owner)
            [name]
            .rsplit('.', 1)
        )
        if not module:
            module = owner.__module__
        else:
            module = importlib.import_module(module[0])
        if cls == 'Self':
            cls = owner.__name__
        cls = getattr(module, cls)
        if self.method is not None:
            cls = getattr(cls, self.method)
        result = cls(*self.args, **self.kwargs)
        setattr(owner, name, result)
        result.__set_name__(owner, name)
        return result.__get__(instance, owner)


def __get__(self: ABCMagic, instance, owner) -> ABCMagic:
    # func = self.__class__.__subget__
    # result = self.__wrap_descriptor__(func, instance, owner)
    # return result
    try:
        func = self.__class__.__subget__
        result = self.__wrap_descriptor__(func, instance, owner)
        return result
    except Exception as e:
        stack = inspect.stack()
        frame = stack[1]
        file_path = frame.filename
        line_number = frame.lineno
        if not getattr(e, 'raising', False):
            e.raising = True
            msg = f'likely raised at {file_path}:{line_number}\n'
            try:
                msg += f':\n {e.args[0]}'
            except IndexError:
                ...
            e.args = msg, *e.args[1:]
        raise


class ABCMagic(
    CachedABC
):
    """
    mostly has to do with Magic type construction rather than implementation
    """
    __name__: str = ''
    __trace__: Trace
    __order__ = Order.second
    __toggle__: dict[str, bool] = {}
    __setter__ = __setter__()
    locals()['__get__'] = __get__
    __direction__: str
    __options__ = Options()
    __sticky__ = True
    __wrap_descriptor__ = __wrap_descriptor__

    def __subget__(self, instance, owner):
        """Override to handle __get__ for a subclass"""

    def __subset__(self, instance, value):
        """Override to handle __set__ for a subclass"""

    def __subdelete__(self, instance):
        """Override to handle __delete__ for a subclass"""

    def __subinit__(self, *args, **kwargs):
        """Override to handle __init__ for a subclass"""

    def __set__(self, instance, value):
        func = self.__class__.__subset__
        value = (
            self.__setter__
            .__get__(instance, instance.__class__)
            (value)
        )
        return self.__wrap_descriptor__(func, instance, value)

    def __delete__(self, instance):
        func = self.__class__.__subdelete__
        return self.__wrap_descriptor__(func, instance)

    # @cached_property
    # def __wrapped__(self):
    #     """The function that has been wrapped"""

    def __init_func__(self, func, *args, **kwargs):
        raise NotImplementedError

    def __init_nofunc__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        """
        Initialize the Magic object, supporting both use as a wrapper,
        and the normal use as intended by the base class.
        """
        if (
                args
        ) and (
                inspect.isfunction(args[0])
                or isinstance(args[0], functools.partial)
        ):
            # Magic instance is decorating, and then being instantiated
            # with no arguments. If the base class doesn't support
            # instantiation without arguments, this fails.
            self.__init_nofunc__(*args[1:], **kwargs)
            self.__init_func__(*args, **kwargs)
        else:
            self.__init_nofunc__(*args, **kwargs)

    # third, second, first, root, etc. should be propagated; can this be done without
    #   explicit assignments?
    # noinspection PyMethodParameters

    # todo: get second if outer.__wrapped__

    """
    how do we get second.edges when second.outer is third order?
    """

    # noinspection PyUnresolvedReferences
    def __init_subclass__(
            cls,
            from_outer=False,
            cached_property=False,
            **kwargs
    ):
        if (
                not from_outer
                and 'from_outer' in cls.__dict__
        ):
            warnings.warn(f"""
                {cls} defined from_outer, which is meant to be used as a
                variable. You most likely intended to define conjure,
                which is the constructor defined for the particular class.

                If you intended to define from_outer, you can suppress this 
                warning by setting from_outer=True in the class definition e.g. 
                class MyClass(Magic, from_outer=True):
                    ...
            """, category=UserWarning)

        try:
            from_options = cls.__dict__['from_options']
        except KeyError:
            ...
        else:

            try:
                func = from_options.__func__
            except AttributeError as e:
                raise AttributeError(f'{cls.__name__}.from_options must be a classmethod') from e
            kwdefaults = func.__kwdefaults__
            if kwdefaults is None:
                raise NotImplementedError

        # if from_options is defined, assign the defaults
        if 'from_options' in cls.__dict__:
            from_options = cls.__dict__['from_options']
            kwdefaults = cls.from_options.__func__.__kwdefaults__
            for key, value in kwdefaults.items():
                name = f'__{key}__'
                attr = getattr(cls, name)
                try:
                    setattr(cls, name, attr.__ior__(value))
                except (TypeError, AttributeError):
                    ...

            if not isinstance(from_options, classmethod):
                raise ValueError(
                    f"{cls.__module__}.{cls.__name__}.from_options"
                    f" must be a classmethod"
                )

            # noinspection PyUnresolvedReferences
            if from_options.__func__.__code__.co_argcount > 1:
                raise ValueError(f"""
                {cls.__module__}.{cls.__name__}.from_options
                must not have any positional arguments!
                be sure this method looks like this, with the
                'cls', *, and **kwargs:
                @classmethod
                def from_options(cls, *, ..., **kwargs):
                    ...
                """)

        # resolve delayed imports indicated by annotations
        for attr, hint in cls.__annotations__.items():
            # if a method is also annotated as Magic,
            #   it's to be wrapped later.
            # If an ellipsis is annotated as Magic,
            #   it's to be instantiated later.
            attr: str
            try:
                already = getattr(cls, attr)
            except AttributeError:
                continue
            if already is Ellipsis:
                already = None
            elif inspect.isfunction(already):
                ...
            else:
                continue
            delayed = DelayedImport(already)
            delayed.__set_name__(cls, attr)
            setattr(cls, attr, delayed)

        if (
                not cached_property
                and cls.__order__ != 3
        ):
            name = f'{cls.__module__}.{cls.__name__}'
            for key, value in cls.__dict__.items():
                if isinstance(value, functools.cached_property):
                    # todo: uncomment this once we add proper magic.config properties
                    msg = (
                        f'Using a {value.__class__.__name__} for {key} in '
                        f'{cls.__order__} order Magic subclass {name} '
                        f' may result in unintended behavior. '
                    )
                    # stack = inspect.stack()
                    # frame = stack[1]
                    # file_path = frame.filename
                    # line_number = frame.lineno
                    # warnings.warn(msg, category=UserWarning)

        super().__init_subclass__(**kwargs)

    def enchant(self, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def from_options(
            cls,
            **kwargs,
    ) -> enchant:
        result = cls()
        if cls.__order__ == 1:
            raise ValueError(f"""
            {cls.from_options} is not a valid option for {cls}
            with order {cls.__order__}
            """)

        for key, value in kwargs.items():
            name = f'__{key}__'
            setattr(result, name, value)

        return result.enchant
