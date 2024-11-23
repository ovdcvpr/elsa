from __future__ import annotations
import concurrent.futures
import itertools
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

import magicpandas as magic

if False:
    # from elsa.prediction.prediction import Prediction
    # import elsa.prediction.prediction
    import elsa.scored.scored


class ToSvgs(magic.Magic):
    @magic.cached.outer.property
    def _scored(self) -> elsa.scored.scored.Scored:
        ...

    def to_svgs(
            self,
            outdir: Path | str,
            threshold: float = None,
            observed=True,
            buffer: int = 500,
            background: str = 'black',
    ):
        """
        Saves each of the predictions to SVG files with bounding boxes and a side panel displaying scores.
        /outdir/filename/prompt.svg
        """
        outdir = Path(outdir).expanduser().resolve()
        scored = self._scored.copy()

        if threshold is not None:
            loc = scored['score'] >= threshold
            scored = scored.loc[loc].copy()

        ifile2path = scored.images.path.to_dict()

        # Define a list of colors for bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'lime', 'pink']
        color_cycle = itertools.cycle(colors)

        def create_svg(group, prompt, ifile, path, outpath):
            # Load image
            image = np.array(Image.open(path))

            # Create figure and axis
            fig, ax = plt.subplots()
            ax.imshow(image)

            # Add bounding boxes
            for _, row in group.iterrows():
                xmin, ymin, xmax, ymax, score = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['score']
                color = next(color_cycle)  # Get the next color from the cycle
                rect = patches.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(xmin, ymin, f'{score:.2f}', color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))

            # Remove axes
            ax.axis('off')

            # Convert figure to image
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            plt.close(fig)

            # Create PIL Image from Matplotlib figure
            overlay_img = Image.fromarray(image_data).convert("RGBA")

            # Add a new image with extra space for text
            new_width = overlay_img.width + buffer
            new_image = Image.new('RGBA', (new_width, overlay_img.height), background)
            new_image.paste(overlay_img, (0, 0))

            # Draw text on the side panel
            draw = ImageDraw.Draw(new_image)
            text_color = 'white' if background == 'black' else 'black'
            y_offset = 10

            # Display file name and score type
            draw.text((overlay_img.width + 10, y_offset), f'file={ifile}', fill=text_color)
            y_offset += 20
            draw.text((overlay_img.width + 10, y_offset), f'prompt={prompt}', fill=text_color)
            y_offset += 20

            # List scores
            for _, row in group.iterrows():
                draw.text((overlay_img.width + 10, y_offset), f"score: {row['score']:.2f}", fill=text_color)
                y_offset += 20

            # Save the final image as an SVG
            path = '/tmp/svgs/BSV_465/a person walking.png'
            new_image.save(path, format='png')
            new_image.save(outpath, format='png')

        # Group data by prompt and file, using multithreading to create SVGs
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for (prompt, ifile), group in scored.groupby('prompt ifile'.split(), sort=False, observed=observed):
                outpath = outdir.joinpath(ifile, f'{prompt}.svg')
                path = ifile2path[ifile]
                outpath.parent.mkdir(parents=True, exist_ok=True)
                futures.append(executor.submit(create_svg, group, prompt, ifile, path, outpath))

            for future in futures:
                future.result()

        return

    def __call__(
            self,
            outdir: Path | str,
            threshold: float = None,
            buffer: int = 500,
            background: str = 'black',
            observed=True,
    ):
        self.to_svgs(outdir, threshold, observed, buffer, background)
