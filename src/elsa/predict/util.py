from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import logging

from pathlib import Path
from typing import *

import pandas as pd


def parse(
        chars: list[str],
        offset_mapping: list[Tuple[int, int]],
        natural: str
) -> list[str]:
    CHARS = chars
    result = []
    for ifirst, ilast in offset_mapping:
        chars = set(CHARS[ifirst:ilast]) - {' '}
        match len(chars):
            case 0:
                result.append(' ')
            case 1:
                result.append(chars.pop())
            case _:
                token = CHARS[ifirst:ilast]
                substring = natural[ifirst:ilast]
                msg = f'In the prompt {CHARS!r}, there is a token '
                msg += f'{token} with multiple chars: {chars} for '
                msg += f"the substring '{substring}' in '{natural}'"
                raise ValueError(msg)
    return result

def replace_column_names(
        outpath: str | Path,
        names: pd.MultiIndex
):
    frame = pd.read_parquet(outpath)
    frame.columns = names
    frame.to_parquet(outpath)



def resolve_duplicates(
        outdir: str | Path,
        logger: logging.Logger = None,
):
    rglob = (
        Path(outdir)
        .expanduser()
        .resolve()
        .rglob('*.parquet')
    )
    natural2path = defaultdict(list)

    # Group paths by filename (without directory)
    for path in rglob:
        natural2path[path.stem].append(path)

    if logger:
        duplicates = {
            filename: paths
            for filename, paths in natural2path.items()
            if len(paths) > 1
        }
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicates")

    # Iterate over each group of paths with the same name
    for filename, paths in natural2path.items():
        if len(paths) > 1:
            # Sort paths by modification time, keeping the most recent file last
            paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Keep the most recent file and delete the others
            for path_to_delete in paths[1:]:
                if logger:
                    logger.warning(f"Deleting {path_to_delete}")
                try:
                    path_to_delete.unlink()  # Deletes the file
                except Exception as e:
                    ...

