from __future__ import annotations

import ast
from collections import Counter
from collections import UserDict

import geopandas as gpd
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from numpy import ndarray
from pandas import Series, Index
from pathlib import Path
from shapely import *
from typing import *

from elsa import boxes
from PIL import ImageFont

if False:
    from elsa.annotation.annotation import Annotated

colors = (
    'red orange green blue purple '
    'brown pink grey cyan magenta lime navy teal '
    'gold silver violet indigo maroon olive fuchsia '
    'aquamarine turquoise khaki lavender tan coral salmon '
    'sienna beige plum wheat orchid tomato yellowgreen '
    'seagreen skyblue lightblue powderblue '
    'royalblue mediumblue azure chartreuse mediumseagreen '
    'springgreen palegreen mediumspringgreen lawngreen '
    'lightgreen darkgreen forestgreen limegreen '
    'greenyellow aqua deepskyblue dodgerblue steelblue '
    'slateblue lightskyblue lightseagreen darkcyan '
    'cadetblue darkturquoise mediumturquoise paleturquoise '
    'darkslateblue midnightblue cornflowerblue lightslategray '
    'slategray lightslategrey slategrey '
    'lightsteelblue mediumslateblue lightgray '
    'lightgrey gainsboro '
    'honeydew mintcream aliceblue seashell '
    'oldlace ivory '
    'lavenderblush mistyrose'
).split()

colors2 = (
    'red orange green blue purple'
    'brown pink gray grey cyan magenta lime navy teal '
    'gold silver violet indigo maroon olive fuchsia '
    'aquamarine turquoise khaki lavender tan coral salmon '
    'sienna beige plum wheat orchid tomato yellowgreen '
    'seagreen skyblue lightblue powderblue '
    'royalblue mediumblue azure chartreuse mediumseagreen '
    'springgreen palegreen mediumspringgreen lawngreen '
    'lightgreen darkgreen forestgreen limegreen '
    'greenyellow aqua deepskyblue dodgerblue steelblue '
    'slateblue lightskyblue lightseagreen darkcyan '
    'cadetblue darkturquoise mediumturquoise paleturquoise '
    'darkslateblue midnightblue cornflowerblue lightslategray '
    'slategray lightslategrey slategrey '
    'lightsteelblue mediumslateblue lightgray '
    'lightgrey gainsboro '
    'honeydew mintcream aliceblue seashell '
    'oldlace ivory '
    'lavenderblush mistyrose'
).split()


@dataclass
class Constituents:
    unique: ndarray
    ifirst: ndarray
    ilast: ndarray
    istop: ndarray
    repeat: ndarray

    @cached_property
    def indices(self) -> ndarray:
        return np.arange(len(self)).repeat(self.repeat)

    def __len__(self):
        return len(self.unique)

    def __repr__(self):
        return f'Constituents({self.unique}) at {hex(id(self))}'

    def __getitem__(self, item) -> Constituents:
        unique = self.unique[item]
        ifirst = self.ifirst[item]
        ilast = self.ilast[item]
        istop = self.istop[item]
        repeat = self.repeat[item]
        con = Constituents(unique, ifirst, ilast, istop, repeat)
        return con


def constituents(self: Union[Series, ndarray, Index], monotonic=True) -> Constituents:
    try:
        monotonic = self.is_monotonic_increasing
    except AttributeError:
        pass
    if monotonic:
        if isinstance(self, (Series, Index)):
            assert self.is_monotonic_increasing
        elif isinstance(self, ndarray):
            assert np.all(np.diff(self) >= 0)

        unique, ifirst, repeat = np.unique(self, return_counts=True, return_index=True)
        istop = ifirst + repeat
        ilast = istop - 1
        # constituents = Constituents(unique, ifirst, ilast, istop, repeat)
        constituents = Constituents(
            unique=unique,
            ifirst=ifirst,
            ilast=ilast,
            istop=istop,
            repeat=repeat,
        )
    else:
        counter = Counter(self)
        count = len(counter)
        repeat = np.fromiter(counter.values(), dtype=int, count=count)
        unique = np.fromiter(counter.keys(), dtype=self.dtype, count=count)
        val_ifirst: dict[int, int] = dict()
        val_ilast: dict[int, int] = {}
        for i, value in enumerate(self):
            if value not in val_ifirst:
                val_ifirst[value] = i
            val_ilast[value] = i
        ifirst = np.fromiter(val_ifirst.values(), dtype=int, count=count)
        ilast = np.fromiter(val_ilast.values(), dtype=int, count=count)
        istop = ilast + 1
        constituents = Constituents(unique, ifirst, ilast, istop, repeat)

    return constituents


import PIL.Image
import PIL.ImageDraw


def view_detection(detections, image: str) -> PIL.Image:
    # Load the image
    result = PIL.Image.open(image)
    draw = PIL.ImageDraw.Draw(result)
    color_map = {
        i: colors[i]
        for i in set(detections.class_id)
    }
    print(color_map)

    # Iterate over detections and draw each one
    for box, class_id in zip(detections.xyxy, detections.class_id):
        # Convert box coordinates to integer, since draw.rectangle expects integer tuples
        box = tuple(map(int, box))
        # Fetch the color corresponding to the class_id from util.colors
        color_map = colors[class_id]

        # Draw the rectangle
        draw.rectangle(box, outline=color_map)

    return result


def get_ibox(
        boxes: Annotated,
        label=False,
        round=4,
) -> Series[int]:
    _ = boxes.xmin, boxes.ymin, boxes.xmax, boxes.ymax, boxes.file
    columns = 'file w s e n'.split()
    if label:
        _ = boxes.ilabel
        columns.append('isyn')
    needles = boxes[columns].copy()
    columns = 'w s e n'.split()
    # round the columns w s e n to 4 decimal places
    needles[columns] = needles[columns].round(round)
    needles = needles.pipe(pd.MultiIndex.from_frame)
    haystack = needles.unique()
    ibox = (
        pd.Series(np.arange(len(haystack)), index=haystack)
        .loc[needles]
    )
    return ibox


T = TypeVar('T')


def trim_path(path: T) -> T:
    """Removes the directories and the extension from a path."""
    if isinstance(path, Series):
        if len(path):
            result = (
                pd.Series(path)
                .str
                .rsplit(os.sep, expand=True, n=1)
                .iloc[:, -1]
                .str
                .split('.', expand=True, n=1)
                .iloc[:, 0]
                .values
            )
        else:
            result = path
    elif isinstance(path, str):
        result = (
            str(path)
            .rsplit(os.sep, 1)[-1]
            .split('.', 1)[0]
        )
    elif isinstance(path, Path):
        result = (
            Path(path)
            .name
            .split('.', 1)[0]
        )
    else:
        raise TypeError(f'path must be Series, str or Path, not {type(path)}')
    return result


class LocalFile(UserDict):
    def __get__(self, instance: LocalFiles, owner) -> Optional[str]:
        key = os.environ.get('sirius')
        return self.get(key)


class LocalFiles:
    """
    Allows user to map users to local file maps. Here, if os.env[elsa]
    is redacted, LocalFiles.config returns the respective path.
    This way we can better share scripts we're working on.

    class LocalFiles(LocalFiles_):
        config: str = dict(
            redacted='/home/redacted/PycharmProjects/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py',
            redacted="/home/redacted/projects/elsa/configs/cfg_odvg.py"
        )

    To achieve this, I used
    echo 'export elsa="redacted"' >> venv/bin/activate && source venv/bin/activate
    """

    def __init_subclass__(cls, **kwargs):
        for key, value in cls.__dict__.items():
            if not isinstance(value, dict):
                continue
            setattr(cls, key, LocalFile(value))


def sjoin(left: Combo, right: Combo) -> tuple[ndarray, ndarray]:
    _ = left.geometry, right.geometry
    left = left.assign(iloc=np.arange(len(left)))
    right = right.assign(iloc=np.arange(len(right)))
    matches = (
        left
        .reset_index()
        ['geometry iloc'.split()]
        .sjoin(right.reset_index()['geometry iloc'.split()])
    )
    ileft = matches.iloc_left.values
    iright = matches.iloc_right.values
    return ileft, iright


def intersection(
        left: boxes.Boxes,
        right: boxes.Boxes,
) -> ndarray[float]:
    """area of the intersection"""
    assert len(left) == len(right), f'left and right must have the same length, got {len(left)} and {len(right)}'
    w = np.maximum(left.xmin.values, right.xmin.values)
    e = np.minimum(left.xmax.values, right.xmax.values)
    s = np.maximum(left.ymin.values, right.ymin.values)
    n = np.minimum(left.ymax.values, right.ymax.values)
    width = np.maximum(0, e - w)
    height = np.maximum(0, n - s)
    result = width * height
    return result


def area(
        left: boxes.Boxes,
) -> ndarray[float]:
    """area of the boxes"""
    width = left.xmax.values - left.xmin.values
    height = left.ymax.values - left.ymin.values
    result = width * height
    return result


def union(
        left: boxes.Boxes,
        right: boxes.Boxes,
) -> ndarray[float]:
    """area of the union"""
    intersection_area = intersection(left, right)
    area_left = (left.xmax.values - left.xmin.values) * (left.ymax.values - left.ymin.values)
    area_right = (right.xmax.values - right.xmin.values) * (right.ymax.values - right.ymin.values)
    union_area = area_left + area_right - intersection_area
    return union_area


def match(
        left: boxes.Boxes,
        right: boxes.Boxes,
        threshold: float = .9,
) -> ndarray:
    """
    Using REDACTED's suggested approach:

    For each GT box, we find one box across all scored
    that have the highest overlap with it and then find other
    prediction boxes that have more than 90% overlap with that box
    found in scored.
    """
    LEFT = left
    RIGHT = right
    _ = left.area, right.area
    ileft, iright = sjoin(left, right)
    left = LEFT.iloc[ileft]
    right = RIGHT.iloc[iright]

    if left is right:
        loc = ileft != iright
        ileft = ileft[loc]
        iright = iright[loc]

    # box in right which has the highest overlap with left
    area = intersection(left, right)
    intersection = area / left.area.values
    ibest = (
        Series(intersection)
        .groupby(ileft)
        .idxmax()
        .loc[ileft]
        .values
    )
    ibest = iright[ibest]
    best = RIGHT.iloc[ibest]

    # boxes in right that have more than 90% overlap with best
    area = intersection(best, right)
    intersection = area / right.area.values
    loc = intersection >= threshold

    ileft = ileft[loc]
    iright = ibest[loc]
    result = np.concatenate([ileft, iright])
    return result


def to_file(self: pd.DataFrame, path: Path, *args, **kwargs):
    path = Path(path)
    try:
        func = getattr(self, f'to_{path.extension}')
    except AttributeError:
        raise ValueError(f'Extension {path.extension} not supported')
    func(path, *args, **kwargs)


def from_file(cls, path, *args, **kwargs):
    # if isinstance(cls, gpd.DataFrame):
    try:
        return getattr(cls, f'from_{path.suffix[1:]}')
    except AttributeError:
        ...
    else:
        if isinstance(cls, gpd.GeoDataFrame):
            func = getattr(gpd, f'read_{path.suffix[1:]}')
        else:
            func = getattr(pd, f'read_{path.suffix[1:]}')
        frame = func(path, *args, **kwargs)
    return cls(frame)


def compute_average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """ Compute the Average Precision (AP). """
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def draw_text_with_outline(draw, xy, text, font, text_color, outline_color, outline_width):
    x, y = xy
    # Draw the outline by drawing the text in the outline color shifted in all directions
    draw.text((x - outline_width, y - outline_width), text, font=font, fill=outline_color)
    draw.text((x + outline_width, y - outline_width), text, font=font, fill=outline_color)
    draw.text((x - outline_width, y + outline_width), text, font=font, fill=outline_color)
    draw.text((x + outline_width, y + outline_width), text, font=font, fill=outline_color)
    # Draw the main text on top
    draw.text((x, y), text, font=font, fill=text_color)



try:
    font = ImageFont.truetype("Courier New", 12)
except OSError:
    try:
        font = ImageFont.truetype("DejaVuSansMono", 12)
    except OSError:
        try:
            font = ImageFont.truetype("LiberationMono", 12)
        except OSError:
            try:
                font = ImageFont.truetype('Arial', 12)
            except OSError:
                font = ImageFont.truetype('C:\\Windows\\Fonts\\arial.ttf', 12)




def stack_ilabel(
        # file: str,
        frame: pd.DataFrame
) -> pd.DataFrame:
    # frame: pd.DataFrame = pd.read_csv(file, index_col=0)
    ilabels = np.fromiter((
        ast.literal_eval(ilabel)
        for ilabel in frame.ilabel
    ), dtype=object, count=len(frame))
    repeat = np.fromiter(map(len, ilabels), dtype=int, count=len(frame))
    ilabel = np.fromiter((
        ilabel
        for lst in ilabels
        for ilabel in lst
    ), dtype=int, count=repeat.sum())
    iloc = np.arange(len(frame)).repeat(repeat)
    frame = frame.iloc[iloc].copy()
    frame['ilabel'] = ilabel
    return frame
    # frame.to_csv(file)


def unstack_ilabel(
        frame: pd.DataFrame,
        columns=tuple('normx normy normwidth normheight ifile data_source'.split())
) -> pd.DataFrame:
    columns = list(columns)

    def apply(frame: pd.DataFrame) -> pd.Series:
        result = frame.iloc[0]
        result.ilabel = frame.ilabel.tolist()
        return result

    result = (
        frame
        .groupby(columns, as_index=False)
        .apply(apply)
    )
    return result


def frame2index(
        frame: pd.DataFrame,
        columns: list[str],
):
    return (
        frame[columns]
        .pipe(pd.MultiIndex.from_frame)
    )



def process_file(filename, indir, outdir, n):
    import os
    import re
    import pandas as pd

    filepath = os.path.join(indir, filename)
    print(f"Reading: {filepath}")
    frame = pd.read_parquet(filepath)
    result = (
        frame
        .sort_values('score', ascending=False)
        .groupby(['prompt', 'ifile'], as_index=False, sort=False, observed=True)
        .head(n)
        .reset_index()
    )
    if 'logit_file' in result:
        del result['logit_file']
    if 'path' in result:
        del result['path']

    output_filepath = os.path.join(outdir, filename)
    print(f"Writing: {output_filepath}")
    result.to_parquet(output_filepath)


def top_n(
        indir: str,
        outdir: str,
        n: int = 25,
        regex: str = '.*mdeter.*\.parquet$',
        nworkers: int = 1,
):
    import os
    import re
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    pattern = re.compile(regex)
    os.makedirs(outdir, exist_ok=True)

    files_to_process = [filename for filename in os.listdir(indir) if pattern.search(filename)]

    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        list(tqdm(
            executor.map(lambda filename: process_file(filename, indir, outdir, n), files_to_process),
            total=len(files_to_process),
            desc="Processing files"
        ))

def append_tp_fp_to_map(
    indir: str,
):
    pd.set_option('display.max_columns', 20)
    map_path = Path(indir) / 'map.csv'
    tp_counts_path = Path(indir) / 'tp_counts.csv'
    counts_path = Path(indir) / 'counts.csv'
    # Read the CSVs into DataFrames
    tp_counts_df = pd.read_csv(tp_counts_path)
    map_df = pd.read_csv(map_path)
    counts_df = pd.read_csv(counts_path)

    # Remove prefixes from column names
    # For map_df, remove 'map_' prefix from columns
    map_df.columns = ['file' if col=='file' else 'method' if col=='method' else col.replace('map_', '') for col in map_df.columns]
    # Rename 'map' column to 'overall'
    map_df = map_df.rename(columns={'map': 'overall'})

    # For counts_df, remove 'counts_' prefix from columns
    counts_df.columns = ['file' if col=='file' else 'method' if col=='method' else col.replace('counts_', '') for col in counts_df.columns]
    counts_df = counts_df.rename(columns={'counts': 'overall'})

    # For tp_counts_df, remove 'tp_' prefix from columns
    tp_counts_df.columns = ['file' if col=='file' else 'method' if col=='method' else col.replace('tp_', '') for col in tp_counts_df.columns]
    tp_counts_df = tp_counts_df.rename(columns={'tp': 'overall'})

    # Convert numeric columns to numeric types
    def convert_numeric(df):
        for col in df.columns:
            if col not in ['file', 'method']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    map_df = convert_numeric(map_df)
    tp_counts_df = convert_numeric(tp_counts_df)
    counts_df = convert_numeric(counts_df)

    # Compute 'fp' as 'counts' - 'tp'
    cols_to_subtract = [col for col in counts_df.columns if col not in ['file', 'method']]

    # Create 'fp' DataFrame by subtracting tp_counts_df from counts_df
    fp_df = counts_df.copy()
    for col in cols_to_subtract:
        if col in tp_counts_df.columns:
            fp_df[col] = counts_df[col] - tp_counts_df[col]
        else:
            fp_df[col] = counts_df[col]

    # Set index of each DataFrame to the desired index (the 'name')
    map_df = map_df.set_index(pd.Index(['map']))
    tp_counts_df = tp_counts_df.set_index(pd.Index(['tp']))
    fp_df = fp_df.set_index(pd.Index(['fp']))

    # Define the desired columns order
    columns_order = ['file', 'method', 'overall', 'c', 'cs', 'csa', 'cso', 'csao', 'person', 'people', 'pair'] + [str(i) for i in range(115)]

    # Reindex DataFrames to have all columns in columns_order
    map_df = map_df.reindex(columns=columns_order)
    tp_counts_df = tp_counts_df.reindex(columns=columns_order)
    fp_df = fp_df.reindex(columns=columns_order)

    # Concatenate the DataFrames
    final_df = pd.concat([map_df, tp_counts_df, fp_df])
    return final_df

# Example usage:
# final_df = process_csv_files('tp_counts.csv', 'map.csv', 'counts.csv')
# print(final_df)

if __name__ == '__main__':
    append_tp_fp_to_map('/home/redacted/Downloads/scores/scores/gdino_swinB_zeroshot_selected_nlse/nms')
    # append_tp_fp_to_map('/home/redacted/Downloads/scores/scores/gdino_swinT_zeroshot_whole_argmax/normal/')
