from __future__ import annotations

from functools import *

import numpy as np
import shapely
from pandas import Series

import magicpandas as magic
from elsa import resource

E = RecursionError, AttributeError


def norm(func):
    """Assure that the result is correctly normalized."""

    tolerance = 1e-1

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        loc = result < 0
        loc &= result >= -tolerance
        result = np.where(loc, 0., result)

        loc = result > 1
        loc &= result <= 1 + tolerance
        result = np.where(loc, 1., result)

        loc = result >= 1 + tolerance
        loc |= result < -tolerance
        # assert not loc.any(), 'result is not correctly normalized'
        n = loc.sum()
        if n:
            msg = f'{n} values out of {len(loc)} are not correctly normalized'
            self.logger.warning(msg)

        return result

    return wrapper


def positive(func):
    """Assure that the result is positive."""

    @wraps(func)
    def wrapper(self: magic.Magic, *args, **kwargs):
        result: np.ndarray = func(self, *args, **kwargs)

        tolerance = -1e-1
        loc = result > tolerance
        loc &= result < 0
        result = np.where(loc, 0., result)

        loc = result < 0
        # assert not loc.any(), 'result is not positive'
        # loc.sum()
        n = loc.sum()
        if n:
            msg = f'{n} values out of {len(loc)} are not positive'
            self.logger.warning(msg)

        return result

    return wrapper


class Boxes(
    resource.Resource,
    magic.geo.Frame,
):
    """
    Base class which contains dynamic columns relating to the spatial
    qualities of subclasses. For a given method, it tries to calculcate
    the result from whichever columns are available.
    """

    @magic.column.from_options(no_recursion=True)
    @positive
    def xmin(self) -> Series[float]:
        try:
            return self.w.values
        except E:
            ...
        try:
            return self.x.values - self.width.values / 2
        except E:
            ...
        try:
            return self.image_width * self.normxmin.values

        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def ymin(self) -> Series[float]:
        try:
            return self.s.values
        except E:
            ...
        try:
            return self.y.values - self.height.values / 2
        except E:
            ...
        try:
            return self.image_height * self.normymin.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def xmax(self) -> Series[float]:
        try:
            return self.e.values
        except E:
            ...
        try:
            return self.x.values + self.width.values / 2
        except E:
            ...
        try:
            return self.image_width * self.normxmax.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def ymax(self) -> Series[float]:
        try:
            return self.n.values
        except E:
            ...
        try:
            return self.y.values + self.height.values / 2
        except E:
            ...
        try:
            return self.image_height * self.normymax.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def w(self) -> Series[float]:
        return self.xmin.values

    @magic.column.from_options(no_recursion=True)
    @positive
    def s(self) -> Series[float]:
        return self.ymin.values

    @magic.column.from_options(no_recursion=True)
    @positive
    def e(self) -> Series[float]:
        return self.xmax.values

    @magic.column.from_options(no_recursion=True)
    @positive
    def n(self) -> Series[float]:
        return self.ymax.values

    @magic.column.from_options(no_recursion=True)
    @positive
    def x(self) -> Series[float]:
        try:
            return (self.xmin.values + self.xmax.values) / 2
        except E:
            ...
        try:
            return self.image_width * self.normx.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def y(self) -> Series[float]:
        try:
            return (self.ymin.values + self.ymax.values) / 2
        except E:
            ...
        try:
            return self.image_height.values * self.normy.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def height(self) -> Series[float]:
        try:
            return self.ymax.values - self.ymin.values
        except E:
            ...
        try:
            return self.image_height * self.normheight
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def width(self) -> Series[float]:
        try:
            return self.xmax.values - self.xmin.values
        except E:
            ...
        try:
            return self.image_width * self.normwidth
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normxmin(self) -> Series[float]:
        try:
            return self.normw.values
        except E:
            ...
        try:
            return self.normx.values - self.normwidth.values / 2
        except E:
            ...
        try:
            return self.xmin.values / self.image_width.values
        except E:
            ...
        try:
            return self.normxmax.values - self.normwidth.values
        except E:
            ...
        try:
            return self.fw.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normymin(self) -> Series[float]:
        try:
            return self.norms.values
        except E:
            ...
        try:
            return self.normy.values - self.normheight.values / 2
        except E:
            ...
        try:
            return self.ymin.values / self.image_height.values
        except E:
            ...
        try:
            return self.normymax.values - self.normheight.values
        except E:
            ...
        try:
            return self.fs.values - self.nfile
        except E:
            ...

        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normxmax(self) -> Series[float]:
        try:
            return self.norme.values
        except E:
            ...
        try:
            result = self.normx.values + self.normwidth.values / 2
            return result
        except E:
            ...
        try:
            result = self.xmax.values / self.image_width.values
            return result
        except E:
            ...
        try:
            # return self.normw.values + self.normwidth.values
            result = self.normxmin.values + self.normwidth.values
            return result
        except E:
            ...
        try:
            # return self.fe.values - self.nfile
            result = self.fe.values - self.nfile
            return result
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normymax(self) -> Series[float]:
        try:
            return self.normn.values
        except E:
            ...
        try:
            return self.normy.values + self.normheight.values / 2
        except E:
            ...
        try:
            return self.ymax.values / self.image_height.values
        except E:
            ...
        try:
            return self.normymin.values + self.normheight.values
        except E:
            ...
        try:
            return self.fn.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normw(self) -> Series[float]:
        return self.normxmin.values

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def norms(self) -> Series[float]:
        return self.normymin.values

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def norme(self) -> Series[float]:
        return self.normxmax.values

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normn(self) -> Series[float]:
        return self.normymax.values

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normx(self) -> Series[float]:
        try:
            return (self.normxmin.values + self.normxmax.values) / 2
        except E:
            ...
        try:
            return self.x.values / self.image_width.values
        except E:
            ...
        try:
            return self.fw.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normy(self) -> Series[float]:
        try:
            return (self.normymin.values + self.normymax.values) / 2
        except E:
            ...
        try:
            return self.y.values / self.image_height.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normwidth(self) -> Series[float]:
        try:
            return self.width.values / self.image_width.values
        except E:
            ...
        try:
            return self.normxmax.values - self.normxmin.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normheight(self) -> Series[float]:
        try:
            return self.height.values / self.image_height.values
        except E:
            ...
        try:
            return self.normymax.values - self.normymin.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def fn(self) -> Series[float]:
        return self.normymax + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fs(self) -> Series[float]:
        return self.normymin + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fe(self) -> Series[float]:
        return self.normxmax + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fw(self) -> Series[float]:
        return self.normxmin + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fx(self) -> Series[float]:
        return self.normx + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fy(self) -> Series[float]:
        return self.normy + self.nfile

    @magic.column
    def area(self) -> magic[float]:
        result = self.width.values * self.height.values
        return result

    @magic.column
    def image_height(self) -> Series[float]:
        return self.images.height.loc[self.ifile].values

    @magic.column
    def image_width(self) -> Series[float]:
        return self.images.width.loc[self.ifile].values

    @magic.geo.column
    def geometry(self):
        return shapely.box(self.fw, self.fs, self.fe, self.fn)
