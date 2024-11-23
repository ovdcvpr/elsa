
from __future__ import annotations

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
    from elsa.prediction.prediction import Prediction
    import elsa.prediction.prediction


class View(magic.Magic):
    @magic.cached.outer.property
    def _prediction(self) -> elsa.prediction.prediction.Prediction:
        ...

    def _string(
            self,
            score: str,
            compare: list[str],
            file: str | int,
            background: str,
            top: int,
            heat: str,
            buffer: int = 500,
            ilogit: int | list[int] = None,
            colors=None,
            show_tokens=True,
            show_scores=True,
            **kwargs
    ) -> Image:
        COLORS = colors or util.colors
        pred = self._prediction
        wsen = 'w s e n'.split()

        _ = pred[wsen]

        if not top:
            top = 0
        top = min(top, 10)

        if isinstance(file, str):
            loc = pred.file == file
            loc |= pred.ifile == file
        elif isinstance(file, int):
            file = pred.file.unique()[file]
            loc = pred.file == file
        elif file is None:
            if pred.file.nunique() != 1:
                raise ValueError("file must be specified if there are multiple files")
            loc = slice(None)
        else:
            raise ValueError(f"file must be str or int, not {type(file)}")

        pred: Self = pred.loc[loc]

        # Apply ilogit filtering if specified
        if ilogit is not None:
            if isinstance(ilogit, int):
                ilogit = [ilogit]
            pred = pred[pred.ilogit.isin(ilogit)]

        score_name = score
        path = pred.path.iat[0]

        try:
            score = pred.scores[score_name]
        except KeyError:
            try:
                score = pred[score_name]
            except KeyError:
                raise ValueError(f"score {score_name} not found")

        iloc = score.argsort()[::-1]
        pred = pred.iloc[iloc]
        score = score.iloc[iloc]

        max_score = score.max()

        if heat:
            alpha = (
                score.values[::-1]
                .__truediv__(max_score)
                .__mul__(255 // 2)
                .astype(int)
            )
            image = Image.open(path).convert("RGBA")
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

        prompt_text = pred.prompt.iat[0]
        pred = pred.iloc[:top]
        score = score.iloc[:top]
        width, height = combined_img.size
        new_width = width + buffer
        new_image = Image.new('RGBA', (new_width, height), background)
        new_image.paste(combined_img, (0, 0))
        draw = ImageDraw.Draw(new_image)
        font = util.font
        text_color = 'white' if background == 'black' else 'black'

        draw.text((width + 10, 10), f'Prompt: {prompt_text}', fill=text_color, font=font)
        draw.text((width + 10, 30), f'file={file}', fill=text_color, font=font)
        draw.text((width + 10, 50), f'score={score_name}', fill=text_color, font=font)

        w = pred.w
        e = pred.e
        n = pred.n
        s = pred.s

        also = {score_name: score.values}
        fails = []
        for name in compare:
            try:
                also[name] = pred.scores[name].values
            except KeyError:
                try:
                    also[name] = pred[name].values
                except KeyError:
                    fails.append(name)
        if fails:
            raise ValueError(f"score {fails} not found")

        if len(pred.confidence.columns):
            labels = (
                pred.confidence
                .set_axis(pred.confidence.token, axis=1)
                .assign(**also)
                .iloc[:, [-1, -2, -3, *range(len(pred.confidence.token))]]
                .astype(str)
                .apply(lambda x: x.str[:6])
            )
        else:
            labels = pd.DataFrame(index=pred.index)

        y_offset = 70

        iloc = np.arange(len(pred)) % len(COLORS)
        colors = np.array(COLORS)[iloc]

        boxes = magic.Frame({
            'w': w,
            'e': e,
            'n': n,
            's': s,
            'color': colors[:len(iloc)],
        })

        it = zip(
            *boxes['w e n s color'.split()].itercolumns()
        )
        for wi, ei, ni, si, color in it:
            draw.rectangle([wi, si, ei, ni], outline=color, width=3)

        ilogit = labels.reset_index()['ilogit'].astype(str)
        COLORS = ['white', 'white'] + COLORS[:len(ilogit)]
        if background == 'white':
            COLORS = ['black', 'black'] + COLORS[:len(ilogit)]

        pred: Prediction

        if show_tokens:
            rows = (
                pred.confidence
                .set_axis(pred.confidence.token, axis=1)
                .astype(str)
                .apply(lambda x: x.str[:6])
                .__repr__()
                .split('\n')
                # [1::]
            )

            if top:
                for row, color in zip(rows, COLORS):
                    draw.text((width + 10, y_offset), row, fill=color, font=font)
                    y_offset += 20
        y_offset += 10

        if show_scores:
            rows = (
                pd.DataFrame(also, index=pred.index)
                .astype(str)
                .apply(lambda x: x.str[:6])
                .__repr__()
                .split('\n')
            )

            if top:
                for row, color in zip(rows, COLORS):
                    draw.text((width + 10, y_offset), row, fill=color, font=font)
                    y_offset += 20

        return new_image

    def __call__(
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
            compare: list[str] = tuple(),
            file: str | int = None,
            heat: str = 'blue',
            top: int = 5,
            background: str = 'black',
            buffer=500,
            ilogit: int | list[int] = None,
            show_tokens=True,
            show_scores=True,

    ) -> Image:
        if isinstance(compare, str):
            compare = [compare]
        with pd.option_context('mode.chained_assignment', None):
            if isinstance(score, str):
                return self._string(
                    score=score,
                    compare=compare,
                    file=file,
                    heat=heat,
                    top=top,
                    background=background,
                    buffer=buffer,
                    ilogit=ilogit,
                    show_tokens=show_tokens,
                    show_scores=show_scores
                )
            elif isinstance(score, list):
                return self._list(
                    score=score,
                    compare=compare,
                    file=file,
                    heat=heat,
                    top=top,
                    background=background,
                    buffer=buffer,
                    ilogit=ilogit,
                    show_tokens=show_tokens,
                    show_scores=show_scores
                )
            else:
                raise ValueError(f"score must be a string or a list of strings")

    def _list(
            self,
            score: list[str],
            compare: list[str],
            file: str | int,
            heat: str,
            top: int,
            background: str,
            buffer: int,
            ilogit: int | list[int] = None,
            show_tokens=True,
            show_scores=True,
    ) -> Image:
        _ = self._prediction.scores[score]
        heatmaps = [
            self._string(
                score=s,
                compare=compare,
                file=file,
                heat=heat,
                background=background,
                top=top,
                buffer=buffer,
                ilogit=ilogit,
                show_tokens=show_tokens,
                show_scores=show_scores
            )
            for s in score
        ]
        widths, heights = zip(*(i.size for i in heatmaps))
        max_width = max(widths)
        total_height = sum(heights)
        new_image = Image.new('RGBA', (max_width, total_height), background)

        y_offset = 0
        for idx, img in enumerate(heatmaps):
            new_image.paste(img, (0, y_offset))
            y_offset += img.height

        return new_image
