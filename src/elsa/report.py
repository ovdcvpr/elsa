from __future__ import annotations

import gc
import os.path
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import pandas as pd
import pyarrow
from tqdm import tqdm

from elsa.resource import Resource

if False:
    from elsa.scored.cdba.summary import Summary


def reorder_columns(df):
    return df[['file', 'method'] + [col for col in df.columns if col not in ['file', 'method']]]


class Report(
    Resource,
):
    def __call__(
            self,
            *files: str,
            outdir: str | Path = './report',
            overwrite=False,
            metrics: tuple[LiteralString[
                'ap.ap',
                'map',
                'si',
                'ss',
                'tp_counts',
                'counts',
            ], ...] | list = (
                    'ap.ap',
                    'map',
                    'tp_counts',
                    'counts',
            ),
            method_parameters: dict = None,
            parallel=False,
    ):
        """
        files:
            List of concatencated parquet files to score, or
            directories containing parquet files to score. See
            `Elsa.scored` to how to generate these.
        outdir:
            The output directory in which a subdirectory for each file
            will be created. For example, passing `gdino_nlse`
            results in a gdino_nlse directory.
        overwrite:
            if True, existing CSV files will be overwritten.
        metrics:
            A list of which metrics to report on. For each metric,
            CSVs are generated, and then at the end of the iteration,
            those CSVs are concatenated.
        method_parameters:
            A dictionary of method parameters. The key is the method
            name, and the value is a dictionary of parameters to pass
            to the method. For example, to change CDBA IOU threshold,
            pass `{'cdba': {'iou': .7}}`.

        outdir/
            filename/
                method/
                    score.csv

        e.g.

        outdir/
            gdino_swinB_zeroshot_whole_argmax/
                cdba/
                    ap.csv
                    ap.by_level.csv
                nms/
                    ap.csv
                    ap.by_level.csv
        """
        outdir = (
            Path(outdir)
            .expanduser()
            .resolve()
        )
        elsa = self.elsa
        futures = []
        methods = 'nms normal cdba'.split()
        total = len(files) * len(metrics) * len(methods)
        counter = tqdm(total=total, desc='Report')
        method_parameters = method_parameters or {}
        for method in method_parameters:
            assert method in methods, f'Unknown method: {method}'

        _files = [
            os.path.join(dir, file)
            for dir in files
            if os.path.isdir(dir)
            for file in os.listdir(dir)
            if file.endswith('.parquet')
        ]
        _files.extend(
            file
            for file in files
            if not os.path.isdir(file)
        )
        files = _files

        def skip(
                files: str | list[str] | tuple[str],
                methods: str | list[str] | tuple[str],
                metrics: str | list[str] | tuple[str],
        ):
            if isinstance(files, str):
                files = [files]
            if isinstance(methods, str):
                methods = [methods]
            if isinstance(metrics, str):
                metrics = [metrics]
            return all(
                outdir.joinpath(
                    Path(file).stem,
                    method,
                    f'{metric}.csv',
                ).exists()
                for file in files
                for method in methods
                for metric in metrics
            )

        with ThreadPoolExecutor() as threads:
            for file in files:
                if (
                        not overwrite
                        and skip(file, methods, metrics)
                ):
                    counter.update((len(methods) * len(metrics)))
                    continue
                try:
                    scored = elsa.scored(outfile=file)
                except pyarrow.lib.ArrowInvalid as e:
                    elsa.scored.__skip_from_params__ = False
                    self.logger.error(f'Parquet file failed to load:')
                    self.logger.error(f'{file}: {e}')
                    counter.update(len(methods) * len(metrics))
                    continue
                del elsa.scored

                filename = Path(file).stem

                for method in methods:
                    parameters = method_parameters.get(method, {})
                    if (
                            not overwrite
                            and skip(file, method, metrics)
                    ):
                        counter.update(len(metrics))
                        continue

                    match method:
                        case 'nms':
                            summary = scored.nms(**parameters).summary
                        case 'normal':
                            summary = scored.normal(**parameters).summary
                        case 'cdba':
                            summary = scored.cdba(**parameters).summary
                        case _:
                            raise ValueError(f'Unknown method: {method}')

                    for metric in metrics:
                        if (
                                not overwrite
                                and skip(file, method, metric)
                        ):
                            counter.update(1)
                            continue
                        path = outdir.joinpath(
                            filename,
                            method,
                            f'{metric}.csv',
                        )
                        self.logger.info(f'{path}')
                        path.parent.mkdir(parents=True, exist_ok=True)

                        result = summary
                        for attr in metric.split('.'):
                            result = getattr(result, attr)

                        if isinstance(result, pd.Series):
                            result = (
                                result
                                .reset_index()
                                .rename(columns={0: metric})
                            )
                        if not isinstance(result, pd.DataFrame):
                            result = (
                                pd.DataFrame(
                                    data=[result],
                                    columns=[metric],
                                )
                            )
                        result['file'] = filename
                        result['method'] = method
                        cols = ['file', 'method'] + [col for col in result.columns if col not in ['file', 'method']]
                        result = result[cols]
                        result = result.reset_index()
                        future = threads.submit(result.to_csv, path, index=False)
                        futures.append(future)
                    delattr(scored, method)
                    gc.collect()
                del elsa.scored
                gc.collect()
            for future in futures:
                future.result()

        self.combine_csvs(outdir)

    def combine_csvs(
            self,
            outdir: str | Path,
    ):
        """
        outdir/
            filename/
                method/
                    score.csv

        outdir/
            gdino_swinB_zeroshot_whole_argmax/
                cdba/
                    ap.csv
                    ap.by_level.csv
                nms/
                    ap.csv
                    ap.by_level.csv

        becomes

        outdir/
            ap.csv
            ap.by_level.csv
            ap.by_condition.csv
            map.csv
            si.csv
            si.by_level.csv
            si.by_condition.csv
        """
        from collections import defaultdict

        name2scores = defaultdict(list)

        # Collect CSV files, excluding those immediately in outdir
        for score_path in outdir.rglob('*.csv'):
            if score_path.parent != outdir:  # Exclude files directly in outdir
                name2scores[score_path.name].append(score_path)

        for name, scores in name2scores.items():
            concat = []
            for score_path in scores:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(score_path)
                concat.append(df)
            combined_df = pd.concat(concat, ignore_index=True)
            combined_path = outdir / name
            combined_df.to_csv(combined_path, index=False)
