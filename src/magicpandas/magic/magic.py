from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import contextlib
import functools
import inspect
import os
import pickle
import tempfile
import time
import warnings
import weakref
from functools import cached_property
from types import FunctionType
from typing import *
from typing import Self, Optional, Any, Callable
from typing import TypeVar

import magicpandas.magic.cached as cached
import magicpandas.magic.cached.base
import magicpandas.magic.cached.diagonal
import magicpandas.magic.cached.root
import magicpandas.magic.cached.volatile
import magicpandas.magic.fixture.traces
from magicpandas.logger.logger import ConsoleLogger
from magicpandas.magic import blank
from magicpandas import util
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.cached.base import Base
from magicpandas.magic.config import Config
from magicpandas.magic.default import default
from magicpandas.magic.delayed import Delayed
from magicpandas.magic.directions import Directions, Direction
from magicpandas.magic.from_method.from_method import FromMethod
from magicpandas.magic.options import Options
from magicpandas.magic.order import Order
from magicpandas.magic.propagating import Propagating
from magicpandas.magic.alias import alias
from magicpandas.magic.root import Root
from magicpandas.magic.second import Second
from magicpandas.magic.stickies import Stickies
from magicpandas.magic.third import Third
from magicpandas.magic.trace import Trace
from magicpandas.magic.truthy import truthy
from magicpandas.magic.volatile import Volatile
from magicpandas.magic.test.tests import Tests
from magicpandas.magic.test.test import Test
from magicpandas.magic.thirds import Thirds

if False:
    from magicpandas.pandas.ndframe import NDFrame
    from magicpandas.magic.drydoc.drydoc import DryDoc

T = TypeVar('T')


class Magic(
    ABCMagic,
    Base,
    from_outer=True,
    cached_property=True,
):
    def conjure(self) -> Self:
        ...

    conjure = None
    __direction__: str | Direction = Directions.horizontal
    __directions__ = Directions()
    __second__: Self = Second()
    __third__ = Third()
    __root__ = Root()
    __volatile__ = Volatile()
    __config__ = Config()
    __from_method__ = FromMethod()
    __options__ = Options()
    __sticky__ = True
    __stickies__ = Stickies()
    __propagating__ = Propagating()
    __trace__ = Trace()
    __delayed__ = Delayed()
    __outer__: Self
    __inner__: Self
    __fixture_traces__ = magicpandas.magic.fixture.traces.Traces()
    Blank = blank.Blank()
    __tests__ = Tests()
    __thirds__ = Thirds()
    drydoc: DryDoc = None

    def __subget__(self, outer: Magic, Outer):
        owner: Self
        if outer is None:
            return self

        elif default.context:
            return self

        elif self.__configuring__:
            key = self.__trace__.__str__()
            if key not in self.__config__:
                self.__config__[key] = self.from_outer()
            result = self.__config__[key]
            return result

        elif self.__nothird__:
            return self

        # noinspection PyTypeChecker
        owner = self.__owner__
        key = self.__key__
        trace = self.__trace__.__str__()
        # cache = self.__owner__.__cache__
        cache = self.__owner_cache__
        if key in cache:
            # get from cached instance attr
            result = cache[key]
            if isinstance(result, weakref.ref):
                result = result()
                if result is None:
                    message = f'weakref to {trace} is None'
                    raise ValueError(message)

            return result

        if trace in owner.__config__:
            # get from config
            return owner.__config__[trace]

        # todo: maybe use volatile in wrap_descriptor instead to minimize user error
        volatile = self.__volatile__.copy()

        if (
                self.__from_file__
                and os.path.exists(self)
        ):
            # load from file
            result = self.__from_file__()
            # todo: could this cause a memory leak?
            try:
                result.__unlink__ = self.__unlink__
            except AttributeError as e:
                warnings.warn(str(e))

        elif self.from_outer:
            # compute from func
            func = self.from_outer.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            if self.__log__:
                result = self.__log__(func)
            else:
                result = func()
            outer.__inner__ = inner

        elif self.conjure:
            # load from inner
            if self.__log__:
                result = self.__log__(self.conjure, outer)
            else:
                result = self.conjure()
        else:
            raise ValueError(
                f'Could not resolve a constructor for {self.__trace__}. '
                f'If get-before-set is acceptable, you must explicitly return None.'
            )

        # noinspection PyUnresolvedReferences
        self.__subset__(outer, result)
        result = cache[key]
        if isinstance(result, weakref.ref):
            result = result()
        if (
                self.__from_file__
                and not os.path.exists(self)
        ):
            self.__to_file__(result)

        self.__volatile__.update(volatile)

        return result

    def __subset__(self, instance: Magic, value):
        # if isinstance(value, ABCMagic):
        #     value = weakref.ref(value)

        if self.__configuring__:
            cache = self.__config__
            key = self.__trace__.__str__()
        else:
            # cache = self.__owner__.__cache__
            cache = self.__owner_cache__
            key = self.__key__
        cache[key] = value

        return value

    def __subdelete__(self, instance: Magic):
        if self.__configuring__:
            cache = self.__config__
            key = self.__trace__.__str__()
        else:
            # cache = self.__owner__.__cache__
            cache = self.__owner_cache__
            key = self.__key__
        if key in cache:
            del cache[key]

    @property
    def __owner_cache__(self) -> dict:
        return self.__owner__.__cache__

    def __propagate__(self, obj: Magic | T) -> T:
        """ set metadata from another object """
        if obj is None:
            return
        if self.__order__ == 2:
            _ = self.__second__
        if obj.__order__ == self.__order__:
            cache = self.__directions__.horizontal
        elif obj.__order__ > self.__order__:
            cache = self.__directions__.vertical
        else:
            raise ValueError(f'obj.order < self.order')
        diagonal = self.__directions__.diagonal
        obj.__dict__.update(
            (key, value)
            for key, value in self.__dict__.items()
            if key in cache
            or key in diagonal
        )
        obj.__volatile__.update(self.__volatile__)

        return obj

    def __set_name__(self, Outer: Magic, name):
        self.__name__ = name
        self.__first__: Optional[Magic] = self
        self.__order__ = Order.first
        self.__Outer__ = Outer
        self.__third__: Optional[Magic] = None
        Base.__set_name__(self, Outer, name)

        if (
            self.__class__.__order__ != 3
        ):
            # propagate nested third-order magic
            ours = thirds = self.__thirds__
            theirs = Outer.__thirds__
            theirs.update({
                f'{name}.{key}': value
                for key, value in ours.items()
            })
            ours = self.__directions__
            theirs = Outer.__directions__
            theirs.update({
                f'{name}.{key}': value
                for key, value in ours.items()
                if key in thirds
            })
            # transferring from nested class to outer


    # @lru_cache()
    def __repr__(self):
        try:
            result = self.__trace__.__str__()
            match self.__order__:
                case Order.first:
                    result += ' 1st'
                case Order.second:
                    result += ' 2nd'
                case Order.third:
                    result += ' 3rd'

            return result
        except AttributeError as e:
            return super().__repr__()

    @cached_property
    def __directory__(self):
        """
        the file path of the current instance; this is used for caching
        """
        # cwd/magic/magic
        return os.path.join(
            self.__rootdir__,
            self.__trace__.replace('.', os.sep),
        )

    @cached.root.property
    def __timeit__(self) -> float:
        """
        If a subprocess took more than this long to complete,
        log the time taken. If 0, every time is logged.
        If -1, no time is logged. This can be overridden with
        @cached.property so that the time is specific to instance.
        """
        return 1.

    @property
    def __cache__(self):
        return self.__dict__

    @cached.volatile.property
    def __outer__(self) -> Self | ABCMagic | NDFrame:
        """
        The object immediately outside this nested object;

        class Outer(magic):
            @Inner
            def inner(self):
                ...

        outer = Outer()
        outer.inner

        Here, inner is nested in outer
        """
        return

    @cached.volatile.property
    def __owner__(self) -> Union[Self, ABCMagic, NDFrame, Magic]:
        """
        The object that the __dict__ containing this attribute is attached to;

        class Outer(magic):
            @Inner
            def inner(self):
                ...

        class Owner(frame):
            outer = Outer()

        owner = Owner()
        owner.outer.inner

        Here, inner is owned by owner;
        if you look in owner.__dict__ you will find 'outer.inner'
        """
        return

    @cached.volatile.property
    def __Outer__(self) -> type[Self]:
        """
        The type of the outer class;

        class Outer(magic):
            @Inner
            def inner(self):
                ...
        Outer.inner

        Here, inner.__outer__ is None, but inner.__Outer__ is Outer
        """
        return

    @cached.volatile.property
    def __inner__(self) -> Self | ABCMagic | NDFrame:
        """
        The instance of the object for which the current method is being called;

        class Outer(magic):
            @Inner
            def inner(self: Outer):
                self.__inner__: Outer
                return self.__inner__({
                    'a': [1,2,3],
                })

        Here, we have a simple method with the typical bound self,
        but for some reason we may need to access metadata about the
        inner instance to be constructed. For that purpose we have self.__inner__
        """
        return

    @cached_property
    def __nothird__(self) -> bool:
        """
        True if current instance's nested instances cannot return
        third order instance
        """
        # false if already third
        if self.__order__ == Order.third:
            return False
        outer = self.__outer__
        if outer is None:
            return True
        if outer.__nothird__:
            return True
        # if (
        #     self.__class__.__order__ is Order.third
        #     and self.__order__ is not Order.third
        # ):
        #     return True
        if (
                outer.__class__.__order__ is Order.third
                and outer.__order__ is not Order.third
        ):
            return True
        return False

        # if outer.__order__ < outer.__class__.__order__:
        #     return True
        # return False
        # return (
        #     self.__class__.__order__ == Order.third
        #     and self.__order__ != Order.third
        # )
        # return (
        #
        # ) or (
        #
        # )
        # return (
        #     self.__outer__ is None
        #     and se
        # ) or (
        #     self.__outer__.__nothird__
        # ) or (
        #     self.__order__ != Order.third
        #     and self.__class__.__order__ == Order.third
        # )

        # return (
        #     self.__Outer__ is not None
        #     and self.__outer__ is None
        # )

        # return (
        #     # if it's root, it is a third
        #     self.__Outer__ is not None
        #     and self.__order__ != Order.third
        # ) or (
        #     self.__Outer__ is not None
        # )

    @cached.diagonal.property
    def __name__(self) -> str:
        """
        The name of the attribute in the access chain;
        For first.second.third, the __name__ values are
        '', 'second', 'third', respectively; the root has no name.
        """
        if self.__outer__ is None:
            return ''
        return self.__first__.__name__

    @alias.property
    def name(self):
        return self.__name__

    @cached.volatile.property
    def __Root__(self) -> type[Self]:
        """
        Root.frame.magic.__Root__ is Root
        """
        return

    # @cached.property
    @cached.diagonal.property
    def __first__(self) -> Self | Any | ABCMagic | NDFrame:
        """
        Commutative metadata
        a.b is c.b

        The first order instance of the hierarchy

        class Outer(frame):
            inner = Inner()

        Here, the object literally created by the line
        `inner = Inner()` is the first order instance,
        and is stored in Outer.__dict__['inner']

        The first order instance only contains metadata,
        and does not contain data associated with the process.

        First.second:
        name:           First   second
        cls.order:      3       2
        instance.order: -       1

        """
        if self.__class__.__order__ == 1:
            return self

    @cached.root.property
    def __futures__(self) -> list[Future]:
        """
        The futures that are currently running for this process;
        all futures are awaiting before the process ends.
        """
        return []

    @cached.root.property
    def __threads__(self) -> ThreadPoolExecutor:
        """
        The thread pool associated with this process
        """
        return ThreadPoolExecutor()

    @cached.root.property
    def __logger__(self) -> ConsoleLogger:
        from magicpandas.logger.logger import logger
        return logger

    @cached.root.property
    def __done__(self) -> bool:
        """
        Whether the process is done;
        this is used to determine whether to await the process
        """
        return True

    @cached.root.property
    def __rootdir__(self) -> str:
        # Magic.__init__
        """
        root directory of specific instance;
        all cached files are stored in subdirectories
        """
        # tempfile.tempdir
        # return os.getcwd()

    @cached.base.property
    def __is_limited__(self) -> bool:
        return False

    @cached.root.property
    def __configuring__(self) -> bool:
        return False

    # @cached.base.property
    # def __configuring_first__(self) -> bool:
    #     return False

    # @cached.property
    @cached.base.property
    def __calling_from_params__(self) -> bool:
        return bool(self.__from_method__)

    @property
    def __key__(self) -> str:
        """
        The key for the current instance; this is used for caching
        Has to be string or else pandas.NDFrmae.__getitem__ will fail
        because Trace is a callable.
        """
        trace = self.__trace__
        owner = self.__owner__
        if owner is None:
            return str(trace - self.__outer__.__trace__)
        return str(trace - owner.__trace__)

    @truthy
    def from_outer(self):
        """
        the function belonging to outer that will generate the attribute

        class Outer(Frame):

            @Inner
            def inner(self: Outer) -> Inner:
                return self.__inner__(self)

        here, def thing becomes thing.from_outer, because self is outer
        Regardless, it returns an instance of inner
        """
        raise NotImplementedError(
            f"No constructor has been defined for {self.__trace__}. If the "
            f"method is intended to return None,"
        )

    @property
    def configure(self):
        @contextlib.contextmanager
        def configure():
            root = self.__root__
            config = root.__configuring__
            root.__configuring__ = True
            yield self
            root.__configuring__ = config

        return configure()

    @property
    def freeze(self):
        raise NotImplementedError

        @contextlib.contextmanager
        def freeze():
            yield self

        return freeze()

    def __getnested__(self, key: str) -> Magic:
        """
        get a nested object from the current object
        """
        obj = self
        # column access adds to permanent
        for piece in key.split('.'):
            obj = getattr(obj, piece)
        return obj

    def __test__(self, func):
        test = Test(func, inner=True)
        self.__first__ = self
        self.__tests__.add(test)
        return func

    @alias.property
    def test(self):
        return self.__test__

    @alias.property
    def outer(self):
        return self.__outer__

    @alias.property
    def owner(self):
        return self.__owner__

    @alias.property
    def inner(self):
        return self.__inner__

    @alias.property
    def root(self):
        return self.__root__

    @alias.property
    def trace(self):
        return self.__trace__

    @alias.property
    def logger(self):
        return self.__logger__

    @alias.property
    def unlink(self):
        return self.__unlink__

    @alias.property
    def tests(self):
        return self.__tests__

    @alias.property
    def third(self):
        return self.__third__

    def __init_func__(self, func=None, *args, **kwargs):
        if isinstance(func, Base):
            # todo: when is this still occurring?
            self.__dict__.update(func.__dict__)
        else:
            super().__subinit__(func, *args, **kwargs)

        parameters = inspect.signature(func).parameters
        functools.update_wrapper(self, func)

        if len(parameters) > 1:
            # case from params
            # self.__from_params__ = func
            self.__from_method__.outer = func
        elif (
                not util.returns(func)
                and not util.contains_functioning_code(func)
        ):
            if not self.conjure:
                self.__permanent__ = True
        else:
            self.from_outer = func

    @cached.base.property
    def __postinit__(self):
        """
        If True, the column will be initialized after the initialization
        of the owner, rather than needing to be accessed first.
        """
        return False

    @cached.root.property
    def __rootfile__(self) -> Optional[str]:
        return None

    @cached.root.property
    def __rootdir__(self) -> str:
        rootfile = self.__rootfile__
        if rootfile is None:
            raise AttributeError(f'{self.__trace__}.__rootfile__ is not set')
        filename = (
            self.__rootfile__
            .rsplit(os.sep, 1)[-1]
            .split('.')[0]
        )
        dir = tempfile.tempdir
        module = self.__class__.__module__
        result = os.path.join(dir, module, filename)
        return result

    @cached.diagonal.property
    def __no_recursion__(self) -> bool:
        """If True, raises AttributeError if recursively accessed."""
        return False

    @cached.base.property
    def __is_recursion__(self) -> bool:
        """If True, the cached property is already being accessed."""
        return False

    @cached_property
    def __recursions__(self) -> set[str]:
        return set()

    @contextlib.contextmanager
    def __recursion__(self):
        cache = self.__owner__.__recursions__
        key = self.__key__
        if (
                self.__no_recursion__
                and key in cache
        ):
            raise RecursionError(
                f'{self.__trace__} is recursively defined.'
            )
        empty = not cache
        cache.add(key)
        try:
            yield
        except Exception as e:
            cache.remove(key)
            raise e
        else:
            try:
                cache.remove(key)
            except KeyError as e:
                raise
            if empty:
                cache.clear()

    @classmethod
    def __from_pipe__(cls, *args, **kwargs) -> Self:
        """
        # todo: create this option from commandline pipe e.g.
        <other process> | python <project> first.second.third
        """
        raise NotImplementedError

    @classmethod
    def __from_commandline__(cls, *args, **kwargs) -> Self:
        """
        in commandline:
        python <project> first.second.third
        """
        raise NotImplementedError

    @truthy
    def __from_file__(self):
        with open(self, 'rb') as file:
            result: Self = pickle.load(file)

    def __to_file__(self, value=None):
        if value is None:
            value = self

        def serialize():
            with open(self, 'wb') as file:
                pickle.dump(value, file)

        future = self.__root__.__threads__.submit(serialize)
        self.__root__.__futures__.append(future)

    def __unlink__(self):
        os.unlink(self)

    def __fspath__(self):
        # cwd/magic/magic.pkl
        return self.__directory__ + '.pkl'

    @truthy
    def __log__(
            self,
            func: FunctionType | Any,
            *args,
            **kwargs,
    ):
        """
        log information about the current subprocess
        """
        # todo: allow user to change log level
        if not self.__log__:
            return func(*args, **kwargs)
        logger = self.__logger__
        T = self.__timeit__
        t = time.time()
        logger.info(self.__trace__)
        logger.indent += 1
        result = func(*args, **kwargs)
        t = time.time() - t
        if (
                T is not None
                and 0 <= T <= t
        ):
            logger.info(f'{t=:.2f}s')

        logger.indent -= 1
        return result

    @classmethod
    def from_options(
            cls,
            *,
            postinit=False,
            # log=True,
            log=False,
            from_file=False,
            no_recursion=False,
            **kwargs
    ) -> Callable[[...], Self]:
        return super().from_options(
            postinit=postinit,
            log=log,
            from_file=from_file,
            no_recursion=no_recursion,
            **kwargs
        )

    F = TypeVar('F', bound=Callable)

    @classmethod
    def __method__(cls, func: F) -> Union[Self, F]:
        return func

    # todo: we should just use enchant instead of method
    method = __method__

    @cached_property
    def __chained_safety__(self) -> Self:
        """
        Creates a strong reference to a third-order instance to prevent
        it from being prematurely garbage collected during nested access.

        For example:
        third = (
            eval
            .iloc[:0]
            .f1
            .multilabel
            .__third__
        )

        In this scenario, eval.iloc[:0] creates a new instance, which
        is weakreferenced as __outer__ for f1. However, because there
        are no strong references, by the time we get to __third__,
        eval.iloc[:0] has already been garbage collected. Every
        second-order access assigns to chained_safety to prevent it
        from being cached.
        """

    def __copy__(self):
        return self

    # noinspection PyDefaultArgument
    def __deepcopy__(self, memodict={}):
        return self


if __name__ == '__main__':
    assert Trace('a.b.c') - 'a' == 'b.c'
    assert Trace('a.b.c') - 'a.b' == 'c'

if __name__ == '__main__':
    class Fourth(Magic):
        first = Magic()
        second = Magic()


    class Third(Magic):
        first = Fourth()
        second = Fourth()


    class Second(Magic):
        first = Third()
        second = Third()


    class First(Magic):
        first = Second()
        second = Second()


    print(f'{First.first.__trace__=}')
    second = First.first.second
    print(f'{second.__trace__=}')
    print(f'{First.second.first.second.__trace__=}')
    print(f'{First.second.first.second.first.__trace__=}')
    print(f'{First.second.first.first.__trace__=}')

    assert First.first.__trace__ == 'first'
    assert First.first.first.__trace__ == 'first.first'
    assert First.second.first.second.__trace__ == 'second.first.second'
    assert First.second.first.second.first.__trace__ == 'second.first.second.first'
    assert First.second.first.first.__trace__ == 'second.first.first'
    from magicpandas.pandas.ndframe import NDFrame
