from __future__ import annotations

import functools
import itertools
import numpy as np
import pandas as pd
import re
import warnings
from dataclasses import dataclass
from typing import *

from elsa.annotation.prompts import Prompts
from elsa.annotation.stacked import Stacked
from magicpandas import util

consuming = []


@dataclass
class consume:
    consumer: str | list[str]
    consumed: str | list[str]
    func = None

    def __post_init__(self):
        consuming.append(self)
        self.consumer = self.consumer.split(', ')
        self.consumed = self.consumed.split(', ')

    def __call__(self, func):
        self.func = func
        return func

    @property
    def staticmethod(self):
        return self


class Struct(str):
    cat: str
    label: str

    @classmethod
    def from_row(
            cls,
            string: str,
            label: str,
            cat: str,
    ) -> Self:
        result = cls(string)
        result.label = label
        result.cat = cat
        return result


@consume(
    consumer='pushing stroller',
    consumed='baby',
)
def with_stroller(consumer: Struct, consumed: list[Struct]):
    # stroller consumes baby
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

@consume(
    consumer='person',
    consumed='baby'
)
def baby(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = f'with {consumed[0]}'
    cat = ' ' * len(string)
    label = ' ' * len(string)
    yield string, cat, label


@consume(
    consumer='kid, teenager, elderly, police, laborer, vendor',
    consumed='baby'
)
def baby(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = f'with {consumed[0]}'
    cat = ' ' * len(string)
    label = ' ' * len(string)
    yield string, cat, label



@consume(
    consumer='standing, sitting',
    consumed='pushing cart',
)
def with_cart(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = 'with a cart'
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label


@consume(
    consumer='standing, sitting',
    consumed='pushing stroller',
)
def with_stroller(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = 'with a stroller'
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label


@consume(
    consumer='standing, sitting',
    consumed='pushing stroller',
)
def with_stroller(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = 'with a stroller'
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label


@consume(
    consumer='crossing crosswalk',
    consumed='standing, walking, running',
)
def crossing_crosswalk(consumer: Struct, consumed: list[Struct]):
    string = ' and '.join(consumed)
    cat = '    '.join(
        struct.cat * len(struct)
        for struct in consumed
    )
    label = '    '.join(
        struct.label * len(struct)
        for struct in consumed
    )
    yield string, cat, label

    string = 'to cross a crosswalk'
    cat = consumer.cat * len(string)
    label = consumer.label * len(string)
    yield string, cat, label

@consume(
    consumer='running',
    consumed='sports'
)
def running(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

@consume(
    consumer='walking, standing',
    consumed='mobility aid',
)
def walking(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = consumed[0]
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label

    # string = 'with a'
    # cat = ' ' * len(string)
    # label = ' ' * len(string)
    # yield string, cat, label
    #
    # string = 'cane'
    # cat = consumed[0].cat * len(string)
    # label = consumed[0].label * len(string)
    # yield string, cat, label

@consume(
    consumer='elderly',
    consumed='mobility aid',
)
def walking(consumer: Struct, consumed: list[Struct]):
    # prioritize the elderly getting the mobility aid
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = consumed[0]
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label


@consume(
    consumer='person',
    consumed='mobility aid',
)
def walking(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = consumed[0]
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label

@consume(
    consumer='person, kid, teenager, elderly, police, laborer, vendor',
    consumed='mobility aid',
)
def walking(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = consumed[0]
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label


    # string = 'with a'
    # cat = ' ' * len(string)
    # label = ' ' * len(string)
    # yield string, cat, label
    #
    # string = 'cane'
    # cat = consumed[0].cat * len(string)
    # label = consumed[0].label * len(string)
    # yield string, cat, label
    #

@consume(
    consumer='sitting',
    consumed='on wheelchair',
)
def sitting(consumer: Struct, consumed: list[Struct]):
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = 'on a'
    cat = ' ' * len(string)
    label = ' ' * len(string)
    yield string, cat, label

    string = 'wheelchair'
    cat = consumed[0].cat * len(string)
    label = consumed[0].label * len(string)
    yield string, cat, label



@consume(
    consumer='group, pair',
    consumed='kid, teenager, elderly, police, laborer, vendor, person',
)
def group(consumer: Struct, consumed: list[Struct]):
    """
    before:
        where consumed are consumers
    """
    # group or pair consumes subjects so that to handle plural cases
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label

    string = 'including'
    cat = ' ' * len(string)
    label = ' ' * len(string)
    yield string, cat, label

    string = ' and '.join(consumed)
    cat = '    '.join(
        struct.cat * len(struct)
        for struct in consumed
    )
    label = '    '.join(
        struct.label * len(struct)
        for struct in consumed
    )
    yield string, cat, label


@consume(
    # consumer='kid, teenager, elderly, police, laborer, vendor, person',
    consumer='kid, teenager, police, laborer, vendor, senior',
    consumed='person',
)
def person(consumer: Struct, consumed: list[Struct]):
    """
    This must happen after group!
    If not pair or group, and both laborer and person in group,
    laborer should consume person
    """
    cat = consumer.cat * len(consumer)
    label = consumer.label * len(consumer)
    yield consumer, cat, label


class Consumed(Stacked):
    outer: Stacked
    prompts = Prompts()

    def conjure(self) -> Self:
        """Called when accessing stacked.consumed to instantiate Consumed"""
        stacked = self.outer
        _ = (
            stacked.natural, stacked.ilabel, stacked.ilabels,
            stacked.cat_char, stacked.catchars,
        )
        result = stacked.reset_index('ilabels')
        warnings.simplefilter('ignore', SyntaxWarning)

        for consume in consuming:
            ...

            for consumer in consume.consumer:

                # select where includes consumer
                # a = (
                #     result
                #     .includes(consumer)
                #     .loc[result.ilabels]
                #     .values
                # )
                # a = (
                #     result
                #     .includes(consumer)
                #     .groupby(level='iprompt')
                #     .loc[result.iprompt]
                #     .values
                # )
                loc_consumer = result.includes(consumer)
                if not loc_consumer.any():
                    continue

                # select where includes consumed
                # loc_consumed = np.zeros_like(loc_consumer, bool)
                loc_consumed = pd.Series(False, index=result.index)
                for consumed in consume.consumed:
                    loc_consumed |= result.includes(consumed)

                # c = a & loc_consumed
                # if not c.any():
                #     continue
                # loc_relevant = (
                #     loc_consumer.groupby(level=0).any()
                #     & loc_consumed.groupby(level=0).any()
                # )
                loc_relevant = loc_consumer.groupby(level=0).any()
                loc_relevant &= loc_consumed.groupby(level=0).any()
                loc = result.ilabels == (0, 5, 32)
                result.prompt.loc[loc]
                if not loc_relevant.any():
                    continue
                loc_relevant = loc_relevant.loc[result.iprompt].values

                # todo consume should receive tuples as parameters
                loc_consumer = result.synonymous(consumer)
                loc_consumer &= loc_relevant
                loc_consumer = loc_consumer.values

                loc_consumed = result.synonymous(consume.consumed)
                loc_consumed &= loc_relevant
                loc_consumed = loc_consumed.values

                loc_consumer &= ~loc_consumed
                loc_consumed &= ~loc_consumer

                if (2, 5, 31) in result.ilabels.loc[loc_relevant].values:
                    loc = result.ilabels == (2, 5, 31)
                    ...


                ilocs = (
                    result
                    .assign(iloc=np.arange(len(result)))
                    .loc[loc_consumer]
                    .groupby(level='iprompt', sort=False)
                    .iloc
                    .first()
                    .values
                )
                # get a list of all unique consumed
                groupby = (
                    result
                    .loc[loc_consumed]
                    .groupby('iprompt', sort=False)
                )
                list_consumed = (
                    groupby
                    .natural
                    .unique()
                )
                assert len(list_consumed) == len(ilocs)

                it = zip(
                    result.natural.loc[loc_consumed],
                    result.cat_char.loc[loc_consumed],
                    result.labelchar.loc[loc_consumed],
                )
                data = np.fromiter((
                    Struct.from_row(
                        string=string,
                        label=label,
                        cat=cat,
                    )
                    for string, cat, label in it
                ), dtype=object, count=loc_consumed.sum())
                STRUCT_CONSUMED = (
                    pd.Series(data)
                    .groupby(result.iprompt[loc_consumed], sort=False)
                    .apply(list)
                )
                STRUCT_CONSUMER = pd.Series(np.fromiter(
                    (
                        Struct.from_row(
                            string=string,
                            label=label,
                            cat=cat,
                        )
                        for string, cat, label in zip(
                        result.natural.loc[loc_consumer],
                        result.cat_char.loc[loc_consumer],
                        result.labelchar.loc[loc_consumer],
                    )
                    ), dtype=object, count=loc_consumer.sum()
                ), index=result.iprompt[loc_consumer].values)
                loc = STRUCT_CONSUMER.index == STRUCT_CONSUMED.index
                assert np.all(loc)

                CAT_CONSUMED = result.cat_char.loc[loc_consumed]
                assert CAT_CONSUMED.nunique() == 1
                CONSUMER = result.prompt.values
                CATCHARS = result.catchars.values
                LABELCHARS = result.labelchars.values
                ILABELS = result.ilabels.values

                it = zip(ilocs, STRUCT_CONSUMER, STRUCT_CONSUMED)
                for iloc, consumer, consumed in it:
                    new_prompt = []
                    new_catchars = []
                    new_labelchars = []
                    ilabels = ILABELS[iloc]
                    if ilabels == (2,5,31):
                        ...


                    it = consume.func(consumer, consumed)
                    for string, cat, label in it:
                        new_prompt.append(string)
                        new_catchars.append(cat)
                        new_labelchars.append(label)

                    prompt = ' '.join(new_prompt)
                    cat_char = ' '.join(new_catchars)
                    labelchars = ' '.join(new_labelchars)

                    CONSUMER[iloc] = prompt
                    CATCHARS[iloc] = cat_char
                    LABELCHARS[iloc] = labelchars

                if (2, 5, 31) in result.ilabels.loc[loc_relevant].values:
                    loc = result.ilabels == (2, 5, 31)
                    subloc = loc & loc_relevant
                    ...

                result.prompt = CONSUMER
                result.catchars = CATCHARS
                result.labelchars = LABELCHARS
                result = result.loc[~loc_consumed].copy()

                if (2, 5, 31) in result.ilabels.loc[loc_relevant[~loc_consumed]].values:
                    loc = result.ilabels == (2, 5, 31)
                    ...

        warnings.simplefilter('default', SyntaxWarning)
        prompts = result.prompt
        catchars = result.catchars
        labelchars = result.labelchars
        assert prompts.str.len().eq(catchars.str.len()).all()
        assert prompts.str.len().eq(labelchars.str.len()).all()

        prompt_string = '*'.join(iter(prompts))
        cat_string = '*'.join(iter(catchars))
        label_string = '*'.join(iter(labelchars))
        buffer = cat_string.encode()
        cat_array = np.frombuffer(buffer, dtype='S1').copy()
        buffer = label_string.encode()
        label_array = np.frombuffer(buffer, dtype='S1').copy()

        # find each pattern, and where the group is, clear the cat_char
        list_ifirsts = []
        list_ilasts = []

        def contains(string: str, spaced=True):
            if spaced:
                yield rf'( {string} )'
                yield rf'\*({string} )'
            else:
                yield rf'({string})'

        spaced = (
            'a above an and at below for on or the with to'
        ).split()
        unspaced = ' ;-;'.split(';')
        partial = functools.partial(contains, spaced=False)
        it = itertools.chain(
            map(contains, spaced),
            map(partial, unspaced),
        )
        for pattern in itertools.chain.from_iterable(it):
            # noinspection RegExpRedundantEscape
            sub = r'\\.|[\^\$\.\|\?\*\+\(\)\[\]\{\}]'
            literal_str = re.sub(sub, '', pattern)
            ifirsts = np.fromiter((
                match.start(1)
                for match in
                re.finditer(pattern, prompt_string)
            ), dtype=int)
            ilasts = ifirsts + len(literal_str)
            list_ifirsts.append(ifirsts)
            list_ilasts.append(ilasts)

        ifirsts = np.concatenate(list_ifirsts)
        ilasts = np.concatenate(list_ilasts)
        iloc = util.slices(ifirsts, ilasts)
        cat_array[iloc] = b' '
        label_array[iloc] = b' '

        catchars = (
            cat_array
            .tobytes()
            .decode()
            .split('*')
        )
        labelchars = (
            label_array
            .tobytes()
            .decode()
            .split('*')
        )

        result.catchars = catchars
        result.labelchars = labelchars

        ilabels = self.elsa.classes.ilabels[1:]
        assert ilabels.isin(result.ilabels).all()

        return result
