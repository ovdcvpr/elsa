from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from matplotlib.colors import to_rgb
from pandas import DataFrame

import elsa.util as util
import magicpandas as magic

if False:
    import elsa.scored.scored
    from elsa.scored.scored import Scored


class View(magic.Magic):
    @magic.cached.outer.property
    def _scored(self) -> elsa.scored.scored.Scored:
        ...

    def _string(
            self,
            file: str | int,
            prompt: str | int,
            background: str,
            top: int,
            heat: str,
            buffer: int = 500,
            colors=None,
            **kwargs
    ) -> Image:
        COLORS = colors or util.colors
        pred = self._scored
        wsen = 'w s e n'.split()
        _ = pred[wsen]

        if not top:
            top = 0
        top = min(top, 10)

        loc = np.full_like(pred.file, True)
        if isinstance(file, str):
            a = pred.file.values == file
            a |= pred.ifile.values == file
            loc &= a
        elif isinstance(file, int):
            file = pred.ifile.loc[loc].unique()[file]
            loc &= pred.ifile.values == file
        elif file is None:
            if pred.file.nunique() != 1:
                raise ValueError("file must be specified if there are multiple files")
        else:
            raise ValueError(f"file must be str or int, not {type(file)}")

        if isinstance(prompt, str):
            loc &= pred.prompt.values == prompt
        elif isinstance(prompt, int):
            prompt = pred.prompt.loc[loc].unique()[prompt]
            loc &= pred.prompt.values == prompt
        elif prompt is None:
            if pred.prompt.nunique() != 1:
                raise ValueError("prompt must be specified if there are multiple prompts")
        else:
            raise ValueError(f"prompt must be str or int, not {type(prompt)}")

        pred: Self = pred.loc[loc]
        path = pred.path.iat[0]
        label = pred.label.iat[0]
        score = pred.score
        iloc = score.argsort()[::-1]
        pred = pred.iloc[iloc]
        score = score.iloc[iloc]
        max = score.max()

        # Apply heatmap if heat is specified
        if heat:
            alpha = (
                score.values[::-1]
                .__truediv__(max)
                .__mul__(255 // 2)
                .astype(int)
            )
            image = (
                Image.open(path)
                .convert("RGBA")
            )
            rgba = (*to_rgb(heat), 0)
            shape = (*image.size[::-1], 4)
            overlay = np.full(shape, rgba, dtype=np.uint8) * 255

            W, S, E, N = (
                pred.loc[::-1, wsen]
                .pipe(magic.Frame)
                .astype(int)
                .itercolumns()
            )
            for w, s, e, n, a in zip(W, S, E, N, alpha):
                overlay[s:n, w:e, 3] = a

            overlay_img = Image.fromarray(overlay, mode="RGBA")
            combined_img = Image.alpha_composite(image, overlay_img)
        else:
            combined_img = Image.open(path).convert("RGBA")

        pred: Scored = pred.iloc[:top]
        # Add extra space for the filename and score
        width, height = combined_img.size
        new_width = width + buffer
        new_image = Image.new('RGBA', (new_width, height), background)
        new_image.paste(combined_img, (0, 0))
        draw = ImageDraw.Draw(new_image)
        font = util.font
        text_color = 'white' if background == 'black' else 'black'

        draw.text((width + 10, 10), f'ifile={file}', fill=text_color, font=font)
        draw.text((width + 10, 30), f'label={label}', fill=text_color, font=font)

        w = pred.w
        e = pred.e
        n = pred.n
        s = pred.s

        labels = pred.confidence.round(2)
        draw = ImageDraw.Draw(new_image)

        # Print the prompt once at the top of the image
        y_offset = 10
        y_offset += 20
        y_offset += 20

        iloc = np.arange(len(pred)) % len(COLORS)
        colors = np.array(COLORS)[iloc]

        boxes = magic.Frame({
            'w': w,
            'e': e,
            'n': n,
            's': s,
            'color': colors[:len(iloc)],
        })

        # Draw the bounding boxes
        it = zip(
            *boxes
            ['w e n s color'.split()]
            .itercolumns()
        )
        for wi, ei, ni, si, color in it:
            draw.rectangle([wi, si, ei, ni], outline=color, width=3)
        labels: DataFrame

        ipred = labels.reset_index()['ipred'].astype(str)
        COLORS = ['white', 'white'] + COLORS[:len(ipred)]
        if background == 'white':
            COLORS = ['black', 'black'] + COLORS[:len(ipred)]

        pred: Scored
        rows = (
            pred.confidence
            .round(2)
            .__repr__()
            .split('\n')
            [1::]  # ignore magicpandas trace
        )
        if top:
            for row, color in zip(rows, COLORS):
                draw.text((width + 10, y_offset), row, fill=color, font=font)
                y_offset += 20
        y_offset += 10

        return new_image

    def __call__(
            self,
            file: str | int = None,
            prompt: str | int = None,
            heat: str = 'blue',
            top: int = 10,
            background: str = 'black',
            buffer=500,
    ) -> Image:
        with pd.option_context('mode.chained_assignment', None):
            return self._string(
                file=file,
                prompt=prompt,
                heat=heat,
                top=top,
                background=background,
                buffer=buffer,
            )

    def to_directory(
            self,
            outdir: Path | str,
            heat: str = 'blue',
            top: int = 10,
            background: str = 'black',
            buffer=500,
            threshold: float = None,
            observed=True,
            **kwargs
    ):
        """
        Saves each of the predictions to SVG files.
        /outdir/filename/prompt.svg
        """
        outdir = (
            Path(outdir)
            .expanduser()
            .resolve()
        )
        scored = self._scored
        _ = scored['xmin ymin xmax ymax score'.split()]
        if threshold:
            loc = scored.score >= threshold
            scored = scored.loc[loc].copy()

        # Group data by prompt and file, using multithreading to create SVGs
        with concurrent.futures.ThreadPoolExecutor() as threads:
            futures = []
            for (prompt, ifile), group in scored.groupby('prompt ifile'.split(), sort=False, observed=observed):
                view = scored.view(
                    file=ifile,
                    prompt=prompt,
                    heat=heat,
                    top=top,
                    background=background,
                    buffer=buffer,
                )
                # outpath = outdir.joinpath(ifile, f'{prompt}.svg')
                outpath = outdir.joinpath(ifile, f'{prompt}.png')
                outpath.parent.mkdir(parents=True, exist_ok=True)
                future = threads.submit(view.save, outpath, )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result()

