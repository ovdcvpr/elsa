from __future__ import annotations
from typing import *
from PIL.Image import Image
import pyarrow.parquet as pq
import pyarrow as pa
import tempfile
import gc

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from functools import *
from pathlib import Path
from typing import *

import pandas as pd
import tqdm
from pandas import Index, MultiIndex
from pandas import Series

from magicpandas.magic import globals
import magicpandas as magic
from elsa import boxes
from elsa.prediction.from_file import from_file
from elsa.prediction.maxlogit import MaxLogit
from elsa.prediction.ranks import Ranks
from elsa.prediction.scores import Scores
from elsa.prediction.view import View
from elsa.prediction.to_svg import ToSvg

E = RecursionError, AttributeError

if False:
    from elsa.root import Elsa


class Prediction(boxes.Boxes):
    outer: Elsa
    columns: MultiIndex
    max_logit = MaxLogit()
    scores = Scores()
    ranks = Ranks()
    without_extraneous_tokens: Self
    without_irrelevant_files: Self
    levels = 'token ifirst label cat'.split()

    # the columns which are to be written to file
    names = (
        'normx normy normwidth normheight file prompt ifile '
        'ilabels_string level w s e n normxmin normxmax normymin normymax '
        'xmin xmax ymin ymax x y width height image_width image_height '
    ).split()
    names: list[str]

    @View
    @magic.portal(View.__call__)
    def view(
            self,
            score: Union[
                str,
                str | Literal[
                    'whole.nlse',
                    'whole.argmax',
                    'selected.nlse',
                    'selected.argmax',
                ],
            ] = 'selected.nlse',
            compare: list[str] = None,
            file: str | int = None,
            heat: str = 'blue',
            top: int = 5,
            background: str = 'black',
            buffer=500,
    ) -> Image:
        """
        score:
            Main scores to use when thresholding the logits
        compare:
            Additional scores to display
        file:
            filename, ifile, or integer representative of the file to display
        heat:
            color of the heatmap; None or '' for no heatmap
        top:
            number of boxes to display
        background:
            color of the background
        buffer:
            size of the background buffer where the text is written
        """

    # noinspection PyMethodOverriding
    def __call__(
            self,
            file: str | Path,
            check=True,
    ) -> Self:
        # result = self.from_file(file).pipe(self.enchant)
        result: Self = (
            self
            .from_file(file)
            .pipe(self.enchant)
        )
        loc = result.confidence.notna().all(axis=1).values
        if not loc.all():
            self.logger.warn(f"dropping logits with NaN confidence: {loc.sum()}")
            result = result.loc[loc]
        # if check:
        #     elsa = self.elsa
        #     # assures the prediction's ilabels and iclass match its prompt;
        #     #   this is to assure updates to the prompt generation
        #     #   are reflected in the predictions
        #     loc = result.prompt.isin(elsa.prompts.natural.values)
        #     result: Self = result.loc[loc].copy()
        #     iclass = (
        #         elsa.prompts.iclass
        #         .indexed_on(result.prompt, name='natural')
        #         .values
        #     )
        #     result.iclass = iclass
        #     del result.ilabels

        loc = result.ifile.isin(self.elsa.ifile).values
        result = result.loc[loc]
        return result

    @magic.cached.static.property
    def confidence(self) -> Self:
        """Select only the columns that describe label confidence"""
        loc = self.ifirst != ''
        result = self.loc[:, loc]
        return result

    @magic.index
    def ilogit(self) -> magic[int]:
        """index of the logits"""

    @magic.cached.static.property
    def token(self) -> Index:
        """
        return the column names that describe the token
        token is the substring of the prompt that is represented by a token
        """
        return self.columns.get_level_values('token')

    @magic.cached.static.property
    def ifirst(self) -> Index:
        """return the column names that describe the start of the token"""
        return self.columns.get_level_values('ifirst')

    @magic.cached.static.property
    def cs(self) -> Self:
        loc = self.cat.isin(['condition', 'state'])
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def csa(self) -> Self:
        loc = self.cat.isin(['condition', 'state', 'activity'])
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def cso(self) -> Self:
        loc = self.cat.isin(['condition', 'state', 'other'])
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def csao(self) -> Self:
        loc = self.cat.isin(['condition', 'state', 'activity', 'other'])
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def condition(self) -> Self:
        loc = self.cat == 'condition'
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def state(self) -> Self:
        loc = self.cat == 'state'
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def activity(self) -> Self:
        loc = self.cat == 'activity'
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def other(self) -> Self:
        loc = self.cat == 'other'
        result = self.loc[:, loc]
        return result

    @magic.column.from_options(dtype='category')
    def file(self) -> magic[str]:
        """filename"""

    @magic.column.from_options(dtype='category')
    def path(self) -> magic[str]:
        """path to image"""
        _ = self.ifile
        result = (
            self.elsa.files.path
            .loc[self.ifile]
            .values
        )
        return result

    @magic.cached.sticky.property
    def passed(self) -> Optional[str]:
        """the passed path to the logits parquet"""
        return None

    @magic.column
    def prompt(self) -> magic[str]:
        """the (synonymous) prompt that was used to generate the logits"""

    @classmethod
    def from_file(
            cls,
            file: str | Path,
            elsa: Optional[Elsa] = None,
            as_cls: bool = True,
    ) -> Self:
        """
        construct from a single file;
        elsa carries metadata
        """
        file = Path(file)
        result = (
            pd
            .read_parquet(file)
            # .pipe(cls)
        )
        if as_cls:
            result = cls(result)
        if '.' in result.columns[0]:
            # I was experiencing an issue with to_parquet with multi-level columns
            # so I compressed them into one-level a.b.c.d; this expands it back to [a, b, c, d]
            # names = 'cat label token ifirst'.split()
            names = 'token ifirst'.split()
            columns = (
                result.columns
                .to_frame()
                [0]
                .str.split('.')
                .pipe(pd.MultiIndex.from_tuples, names=names)
            )
            result.columns = columns

        result.passed = file
        if elsa is not None:
            # result.outer = result.owner = elsa
            result.elsa = elsa
            # someone else might have generated the logits, need to
            #   get the paths from the user's files
            try:
                result.path = (
                    elsa.files.path
                    .set_axis(elsa.files.file)
                    .loc[result.file]
                    .values
                )
            except KeyError:
                ...
        if 'prompt' not in result:
            result['prompt'] = file.stem
            result['prompt'] = result['prompt'].astype('category')

        if not result.index.name:
            result.index.name = 'ilogit'
        if 'cat' in result.columns.names:
            result.columns.droplevel('cat label'.split())
        if (
                isinstance(result.columns, pd.MultiIndex)
                and not result.columns.names[0]
        ):
            try:
                result.columns.names = Prediction.levels
            except ValueError:
                # noinspection PyTypeChecker
                n = len(Prediction.levels) - len(result.columns.levels)
                empty = [''] * n
                new = pd.MultiIndex.from_tuples((
                    (*old, *empty)
                    for old in result.columns.tolist()
                ), names=Prediction.levels)
                result.columns = new

        assert not result.ilogit.duplicated().any(), "ilogit must be unique"
        result['logit_file'] = str(file)

        return result

    @magic.column
    def logit_file(self):
        """Output path to the scored"""

    @magic.column
    def score(self) -> magic[float]:
        """The score to be used in evaluating the logits"""
        return self.scores.selected.nlse.values

    # @cached_property
    @magic.cached.static.property
    def scored(self) -> Self:
        """
        Return a subframe with the score and essential columns;
        to be used in concatenation for the evaluation
        """
        _ = self.score
        columns = (
            'prompt file score '
            'normx normy normwidth normheight'
        ).split()
        result = self.loc[:, columns]
        return result

    @magic.cached.sticky.property
    def extraneous(self) -> set[str]:
        """A set of tokens that should be dropped if they appear in the output"""
        result = (
            'a an on at ing with or the ed s to and up down including'
        ).split()
        result = set(result)
        return result

    @magic.cached.static.property
    def without_extraneous_tokens(self) -> Self:
        """Logits without the extraneous tokens e.g. a, an, to, the"""
        loc = self.label != ''
        result = self.loc[:, loc]
        return result

    @magic.cached.static.property
    def without_irrelevant_files(self) -> Self:
        """
        Logits without the files that don't actually contain the prompt
        according to the truth.
        """
        loc = ~self.is_irrelevant_file
        result = self.loc[loc]
        return result

    @magic.column
    def is_irrelevant_file(self) -> magic[bool]:
        """
        Whether the file is does not actually contain the prompt
        according to the truth
        """
        prompt = self.prompt
        files = self.elsa.files
        loc = files.implicated(prompt)
        file = files.file.loc[loc]
        loc = self.file.isin(file)
        result = ~loc
        return result

    @property
    def groupby_files(self) -> Iterator[Self]:
        yield from self.groupby(
            'file',
            as_index=False,
            sort=False,
        )

    @classmethod
    def scored_to_directory(
            cls,
            indir: Path | str,
            outdir: Path | str,
            elsa: Elsa = None,
    ):
        """
        indir: directory of logits
        outdir: directory of scores
        """

        # each infile is indir/subdir/file.parquet
        # each outfile is outdir/subdir/file.parquet
        indir = Path(indir)
        inpaths = list(indir.rglob('*.parquet'))
        outpaths = [
            outdir / infile.relative_to(indir)
            for infile in inpaths
        ]

        with ThreadPoolExecutor() as threads:
            from_file = cls.from_file

            def it_logits():
                it: Iterator[Future] = (
                    threads.submit(from_file, infile, elsa=elsa)
                    for infile in inpaths
                )
                prev = next(it)
                while True:
                    try:
                        curr = next(it)
                    except StopIteration:
                        break
                    yield prev.result()
                    prev = curr
                yield prev.result()

            def it_logits():
                yield from (
                    from_file(infile, elsa=elsa)
                    for infile in inpaths
                )

            it = zip(it_logits(), outpaths)
            futures = []
            for logits, outfile in it:
                outfile.parent.mkdir(parents=True, exist_ok=True)
                scores = logits.scores
                ranks = logits.ranks
                if 'prompt' in logits:
                    prompt = logits.prompt.values
                else:
                    prompt = outfile.stem
                data = {
                    'file': logits.file.values,
                    'ilogit': logits.index.values,
                    'prompt': prompt,
                    'scores.whole.argmax': scores.whole.argmax.values,
                    'scores.whole.nlse': scores.whole.nlse.values,
                    'scores.whole.avglse': scores.whole.avglse.values,
                    'scores.selected.nlse': scores.selected.nlse.values,
                    'scores.selected.avglse': scores.selected.avglse.values,
                    'ranks.whole.argmax': ranks.whole.argmax.values,
                    'ranks.whole.nlse': ranks.whole.nlse.values,
                    'ranks.whole.avglse': ranks.whole.avglse.values,
                    'ranks.selected.nlse': ranks.selected.nlse.values,
                    'ranks.selected.avglse': ranks.selected.avglse.values,
                    'normx': logits.normx.values,
                    'normy': logits.normy.values,
                    'normwidth': logits.normwidth.values,
                    'normheight': logits.normheight.values,
                }
                frame = pd.DataFrame(data)
                frame.file = frame.file.astype('category')
                frame.prompt = frame.prompt.astype('category')
                future = threads.submit(frame.to_parquet, outfile)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

    @classmethod
    def from_directory(
            cls,
            directory: Path | str,
            score: str = 'selected.nlse',
            threshold: float = .3,
            elsa: Elsa = None,
            nms: bool = False,
    ) -> Iterator[Self]:
        """
        yields logits where the score is above the threshold

        prompt  file    ilogit    score   normx   normy   normwidth   normheight
        """
        directory = (
            Path(directory)
            .expanduser()
            .resolve()
        )
        inpaths = list(directory.rglob('*.parquet'))
        assert len(inpaths)
        score = score.split('.')
        columns = (
            'prompt file normx normy normwidth normheight '
        ).split()
        # columns = 'normx normy normwidth normheight file path prompt ifile ilabels'

        with ThreadPoolExecutor() as threads:
            from_file = cls.from_file

            def it_logits():
                it: Iterator[Future] = (
                    threads.submit(from_file, infile, elsa=elsa)
                    for infile in inpaths
                )
                prev = next(it)
                while True:
                    try:
                        curr = next(it)
                    except StopIteration:
                        break
                    yield prev.result()
                    prev = curr
                yield prev.result()

            it = zip(it_logits(), inpaths)
            for logits, infile in tqdm.tqdm(it, total=len(inpaths)):
                logits = cls(logits)
                obj = logits.scores
                for s in score:
                    obj = getattr(obj, s)
                obj = obj.values
                loc = obj >= threshold
                if 'prompt' not in logits:
                    logits['prompt'] = infile.stem
                    logits['prompt'] = logits['prompt'].astype('category')
                argmax = logits.scores.whole.argmax.values
                axis = logits.columns.get_level_values(0)
                scores = logits.scores.everything
                scores.columns = [
                    'scores.' + col
                    for col in scores.columns
                ]

                logits = (
                    logits
                    .set_axis(axis, axis=1)
                    [Prediction.names]
                )

                concat = logits, scores
                result = pd.concat(concat, axis=1)

                yield (
                    result
                    .reset_index()
                    .assign(score=obj, argmax=argmax)
                    .loc[loc]
                )

    @classmethod
    def from_inpaths(
            cls,
            inpaths: list[Path] | Series | list,
            score: str = 'selected.nlse',
            threshold: float = .3,
            multiprocessing=True,
            compare: list[str] = None
    ) -> Self:

        globals.chained_safety = False
        outdir = Path(tempfile.mkdtemp())
        func = partial(
            from_file,
            score=score,
            threshold=threshold,
            outdir=outdir,
            compare=compare,
        )

        # todo: why do I have to iterate and create a new process pool or it consumes all my memory?
        if multiprocessing:
            processes = ProcessPoolExecutor()
            workers = processes._max_workers
            try:
                for i in tqdm.tqdm(range(0, len(inpaths), workers), total=len(inpaths) // workers):
                    with ProcessPoolExecutor() as processes:
                        futures = [processes.submit(func, infile) for infile in inpaths[i:i + workers]]
                        for future in futures:
                            future.result()
            except Exception as e:
                # just try again
                for i in tqdm.tqdm(range(0, len(inpaths), workers), total=len(inpaths) // workers):
                    with ProcessPoolExecutor() as processes:
                        futures = [
                            processes.submit(func, infile)
                            for infile in inpaths[i:i + workers]
                        ]
                        for future in futures:
                            future.result()
        else:
            for infile in tqdm.tqdm(inpaths):
                func(infile)

        # globals.chained_safety = True
        # dataset = pq.ParquetDataset(outdir)
        # table = dataset.read()
        # result: Prediction = table.to_pandas().pipe(cls)

        # globals.chained_safety = True
        # dataset = pq.ParquetDataset(outdir)
        # table = dataset.read()
        #
        # # Check if the table is empty
        # if table.num_rows == 0:
        #     # Extract column names from the schema
        #     columns = table.schema.names
        #     # Create an empty DataFrame with the correct columns
        #     df = pd.DataFrame(columns=columns)
        # else:
        #     # Convert the table to a DataFrame as usual
        #     df = table.to_pandas()
        #
        # # Proceed with your processing
        # result: Prediction = df.pipe(cls)

        globals.chained_safety = True
        dataset = pq.ParquetDataset(outdir)

        # Read the table from the dataset
        table = dataset.read()

        # Convert the table to a pandas DataFrame
        df = table.to_pandas()

        if not len(df):
            raise ValueError("No Parquet files found in the directory to infer schema.")

        # If the DataFrame is empty, extract the columns from the dataset schema
        if df.empty:
            # Ensure the dataset has a schema
            if dataset.schema is not None:
                columns = dataset.schema.names
                # Create an empty DataFrame with the correct columns
                df = pd.DataFrame(columns=columns)
            else:
                # If schema is not available, read from any Parquet file in outdir
                import os
                parquet_files = [f for f in os.listdir(outdir) if f.endswith('.parquet')]
                if parquet_files:
                    first_file = os.path.join(outdir, parquet_files[0])
                    schema = pq.read_schema(first_file)
                    columns = schema.names
                    df = pd.DataFrame(columns=columns)
                else:
                    # If no Parquet files are found, raise an error
                    raise ValueError("No Parquet files found in the directory to infer schema.")

        # Proceed with your processing
        result: Prediction = df.pipe(cls)

        # filter rows where any confidence column is NaN
        loc = (
            result.score
            .notna()
            .values
        )
        if not loc.all():
            result.logger.warn(f"dropping logits with NaN confidence: {loc.sum()}")
            result = result.loc[loc]

        result.logit_file = result.logit_file.astype('category').values
        # result.path = result.path.astype('category').values

        return result

    @magic.column
    def level(self) -> magic[str]:
        if self.elsa is None:
            raise ValueError('elsa must not be None to determine level')
        result = (
            self.elsa.prompts.natural2level
            .loc[self.prompt]
            .values
        )
        return result

    @magic.column
    def ilabels(self) -> magic[tuple[int]]:
        """"""
        try:
            def apply(string: str):
                # return tuple(str.split(string))
                return tuple(map(int, string.split(' ')))

            result = (
                self.ilabels_string
                .set_axis(self.ilabels_string.values)
                .drop_duplicates()
                .astype(str)
                .map(apply)
                .loc[self.ilabels_string.values]
                .values
            )
        except (AttributeError, NotImplementedError):
            result = (
                self.elsa.prompts
                .reset_index()
                .set_axis('natural')
                .ilabels
                .loc[self.prompt]
                .values
            )
        return result

    @ilabels.setter
    def ilabels(self, value: Series):
        """
        If a Series of strings was passed, split them to tuples of ints
        If a Series of lists was passed, convert them to tuples of ints
        """
        if (
                isinstance(value, Series)
                and isinstance(value.iloc[0], list)
        ):
            value = value.apply(tuple)
        return value

    @magic.column
    def ilabels_string(self) -> Series:
        """
        Tuples are not easily serialized; store them as strings instead.
        """
        from elsa.prediction.prompt2ilabels_string import mapping_sorted
        mapping = {
            key: ' '.join(map(str, value))
            for key, value in mapping_sorted.items()
        }
        result = Series(mapping).loc[self.prompt].values
        return result

    @magic.cached.static.property
    def cat(self):
        """return the column names that describe the category"""
        return self.columns.get_level_values('cat')

    @magic.cached.static.property
    def label(self):
        """return the column names that describe the label"""
        return self.columns.get_level_values('label')

    @magic.column
    def relevant(self) -> Series[bool]:
        names = 'ilabels ifile'.split()
        index = partial(pd.MultiIndex.from_arrays, names=names)
        arrays = self.ilabels, self.ifile
        needles = index(arrays)
        truth = self.elsa.truth.combos
        arrays = truth.ilabels, truth.ifile
        haystack = index(arrays)
        result = needles.isin(haystack)
        return result

    # todo: why was this here?
    # @magic.column
    # def width(self) -> magic[int]:
    #     """Width of the image in pixels"""
    #     path: Series[str] = self.path
    #     assert path.str.endswith('.png').all()
    #
    #     def submit(path):
    #         with open(path, 'rb') as f:
    #             f.seek(16)  # PNG width is stored at byte 16-19
    #             width = int.from_bytes(f.read(4), 'big')
    #         return width
    #
    #     with ThreadPoolExecutor() as threads:
    #         it = threads.map(submit, path)
    #         result = list(it)
    #     return result
    #
    # @magic.column
    # def height(self) -> Series[int]:
    #     """Height of the image in pixels"""
    #     path: Series[str] = self.path
    #     assert path.str.endswith('.png').all()
    #
    #     def submit(path):
    #         with open(path, 'rb') as f:
    #             f.seek(20)
    #             height = int.from_bytes(f.read(4), 'big')
    #         return height
    #
    #     with ThreadPoolExecutor() as threads:
    #         it = threads.map(submit, path)
    #         result = list(it)
    #     return result

    @magic.column
    def width(self) -> magic[int]:
        """Width of the image in pixels"""
        result = self.normxmax.values - self.normxmin.values
        result *= self.image_width.values
        return result

    @magic.column
    def height(self) -> magic[int]:
        """Height of the image in pixels"""
        result = self.normymax.values - self.normymin.values
        result *= self.image_height.values
        return result

    @ToSvg
    @magic.portal(ToSvg.__call__)
    def to_svg(self):
        ...

    @magic.series
    def path(self) -> Series[str]:
        result = (
            self.elsa.files.path
            .loc[self.ifile]
            .values
        )
        return result

    @magic.column
    def ntokens(self) -> magic[int]:
        """Number of tokens in the prompt"""



def get_names(*columns: magic.column | Series):
    return {
        column.name
        for column in columns
    }

# cls = Prediction
# Prediction.names = get_names(
#     cls.normx,
#     cls.normy,
#     cls.normwidth,
#     cls.normheight,
#     cls.normxmax,
#     cls.normymax,
#     cls.normymin,
#     cls.normxmin,
#     cls.xmin,
#     cls.xmax,
#     cls.ymin,
#     cls.ymax,
#     cls.x,
#     cls.y,
#     cls.file,
#     cls.prompt,
#     cls.ilabels_string,
#     cls.level,
#     cls.ifile,
# )
