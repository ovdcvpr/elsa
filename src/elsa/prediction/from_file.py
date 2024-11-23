from __future__ import annotations

import weakref

import gc

import pandas as pd
from pathlib import Path
from magicpandas.magic import globals

import gc
import sys

if False:
    from elsa.prediction.prediction import Prediction


def from_file(
        file: Path,
        score: str,
        threshold: float,
        outdir: Path,
        compare: list[str] = None,
) -> Prediction:
    from elsa.prediction.prediction import Prediction
    score_name = score
    globals.chained_safety = False
    outpath = outdir / file.name
    if isinstance(compare, str):
        compare = [compare]

    file = Path(file)
    result = (
        pd
        .read_parquet(file)
        .pipe(Prediction)
    )
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
        result.columns.names = Prediction.levels

    if result.index.name != 'ilogit':
        msg = (
            f'There is a non-prediction parquet at {file}. '
            f'Please ensure you have segregated the predictions from '
            f'the scores.'
        )
        raise ValueError(msg)
    assert not result.index.duplicated().any(), "ilogit must be unique"

    result['logit_file'] = str(file)
    try:
        score = result.scores[score_name]
    except KeyError:
        try:
            score = result[score_name]
        except KeyError:
            raise KeyError(f"score {score_name} not found in {file}")

    score = score.values
    loc = score >= threshold
    if 'prompt' not in result:
        result['prompt'] = file.stem
        result['prompt'] = result['prompt'].astype('category')
    axis = result.columns.get_level_values(0)
    # also include score_ columns
    # columns: pd.Index = result.token.intersection(Prediction.names)
    assert 'w' in Prediction.names
    columns = result.token.intersection(Prediction.names).tolist()
    # score_columns = [col for col in result.token if col.startswith('score_')]
    # columns.extend(score_columns)
    columns = pd.Index(columns)

    scores = {score_name}
    scores.update(compare or [])
    score_columns = [col for col in result.token if col.startswith('score_')]
    scores.update(score_columns)
    appendix = {}
    # for key in scores:
    #     try:
    #         appendix[f'scores.{key}'] = result.scores[key].values
    #     except KeyError:
    #         try:
    #             appendix[key] = result[key].values
    #         except KeyError:
    #             raise KeyError(f"score {key} not found in {file}")
    for key in scores:
        if '.' in key:
            selection, name = key.split('.')
            selection = getattr(result.scores, selection)
            try:
                appendix[f'scores.{key}'] = getattr(selection, name).values
            except TypeError as e:
                raise KeyError(f"score {key} not found in {file}") from e
        else:
            try:
                appendix[f'scores.{key}'] = result.scores[key].values
            except KeyError:
                try:
                    appendix[key] = result[key].values
                except KeyError:
                    raise KeyError(f"score {key} not found in {file}")

    result = (
        result
        .set_axis(axis, axis=1)
        [columns]
    )
    appendix = pd.DataFrame(appendix)
    result = pd.concat([result, appendix], axis=1)

    score_name = pd.Categorical([score_name] * len(result), categories=[score_name])
    result: pd.DataFrame = (
        result
        .reset_index()
        .assign(
            score=score,
            score_name=score_name,
            logit_file=str(file),
        )
        .loc[loc]
        .pipe(pd.DataFrame)
    )
    # force garbage collector or process pool ends up taking an enormous
    #  amount of memory
    result.attrs.clear()
    if len(result):
        result.to_parquet(outpath)
    del score
    del result
    gc.collect()
