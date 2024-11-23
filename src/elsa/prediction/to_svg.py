from __future__ import annotations

import base64
import concurrent.futures
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import to_rgb

import elsa.prediction.prediction
import elsa.util as util
import magicpandas as magic

if False:
    from elsa.prediction.prediction import Prediction


class ToSvg(magic.Magic):
    @magic.cached.outer.property
    def prediction(self) -> elsa.prediction.prediction.Prediction:
        ...


    def _svg(
            self,
            outfile,
            file: str | int,
            background: str,
            top: int,
            compare: list[str] = [],
            buffer: int = 500,
    ) -> None:
        pred = self.prediction
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
            file = pred.ifile.unique()[file]
            loc &= pred.ifile.values == file
        elif file is None:
            if pred.file.nunique() != 1:
                raise ValueError("file must be specified if there are multiple files")
        else:
            raise ValueError(f"file must be str or int, not {type(file)}")

        pred = pred.loc[loc]
        path = pred.path.iat[0]
        score = pred.score
        iloc = score.argsort()[::-1]
        pred = pred.iloc[iloc]
        score = score.iloc[iloc]

        # Open the image to get its size
        combined_img = Image.open(path).convert("RGBA")

        pred = pred.iloc[:top]
        # Add extra space for the filename and score
        width, height = combined_img.size
        new_width = width + buffer

        # Create SVG drawing
        from svgwrite import Drawing
        dwg = Drawing(outfile, size=(new_width, height))

        # Set background color
        dwg.add(dwg.rect(insert=(0, 0), size=(new_width, height), fill=background))

        # Embed the combined image
        from io import BytesIO
        output = BytesIO()
        combined_img.save(output, format='PNG')
        image_data = base64.b64encode(output.getvalue()).decode('utf-8')
        image_href = f'data:image/png;base64,{image_data}'
        dwg.add(dwg.image(href=image_href, insert=(0, 0), size=(width, height)))

        # Now, for the top N predictions, draw bounding boxes and labels
        text_x = width + 10

        # Add text for filename and label
        text_color = 'white' if background == 'black' else 'black'
        dwg.add(dwg.text(f'ifile={file}', insert=(text_x, 10), fill=text_color, font_size=14))
        label = pred.prompt.iat[0]
        dwg.add(dwg.text(f'label={label}', insert=(text_x, 30), fill=text_color, font_size=14))

        w = pred['w'].astype(int).tolist()
        e = pred['e'].astype(int).tolist()
        n = pred['n'].astype(int).tolist()
        s = pred['s'].astype(int).tolist()

        labels = pred.confidence.round(2)
        y_offset = 10 + 20 + 20  # Matches the y_offset in _string

        iloc = np.arange(len(pred)) % len(util.colors)
        colors = np.array(util.colors)[iloc]

        boxes = pd.DataFrame({
            'w': w,
            'e': e,
            'n': n,
            's': s,
            'color': colors[:len(iloc)],
        })

        # Draw the bounding boxes
        for wi, ei, ni, si, color in zip(boxes['w'], boxes['e'], boxes['n'], boxes['s'], boxes['color']):
            x = wi
            y = si
            rect_width = ei - wi
            rect_height = ni - si
            dwg.add(dwg.rect(
                insert=(x, y),
                size=(rect_width, rect_height),
                fill='none',
                stroke=color,
                stroke_width=3
            ))

        # Prepare the rows for the text on the right
        labels = pred.confidence.round(2).astype(str)
        # rows = labels.to_string().split('\n')
        # rows = [rows[0]] + rows[5:]
        rows = (
            pred.confidence
            .set_axis(pred.confidence.token, axis=1)
            .astype(str)
            .apply(lambda x: x.str[:6])
            .__repr__()
            .split('\n')
            # [1::]
        )

        # Ensure colors match the number of rows (header + data rows)
        COLORS = [text_color] + list(colors[:len(rows) - 1])

        # Add the text lines to the SVG
        if top:
            for row, color in zip(rows, COLORS):
                text = dwg.text(row, insert=(text_x, y_offset), fill=color, font_size=14, font_family='monospace')
                text['xml:space'] = 'preserve'
                dwg.add(text)
                y_offset += 20
        y_offset += 10


        score_name = score.name
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
        rows = (
            pd.DataFrame(also, index=score.index)
            .iloc[:top]
            .astype(str)
            .apply(lambda x: x.str[:6])
            .__repr__()
            .split('\n')
        )

        if top:
            for row, color in zip(rows, COLORS):
                text = dwg.text(row, insert=(text_x, y_offset), fill=color, font_size=14, font_family='monospace')
                text['xml:space'] = 'preserve'
                dwg.add(text)
                y_offset += 20
        y_offset += 10

        # Save the SVG file
        dwg.save()

    def __call__(
            self,
            outfile,
            file: str | int = None,
            top: int = 10,
            background: str = 'black',
            buffer=500,
    ) -> None:
        with pd.option_context('mode.chained_assignment', None):
            self._svg(
                outfile=outfile,
                file=file,
                top=top,
                background=background,
                buffer=buffer,
            )
