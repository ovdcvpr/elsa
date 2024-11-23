from __future__ import annotations

from typing import Iterator
import numpy as np
from PIL import Image
import os

from typing import Iterator

import numpy as np
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator

if False:
    from elsa.scored.cdba.summary import Summary





def compare_pr_curves(
        *summaries: Summary,
        directory: str = '',
        unique_iclass=False,
        leader: int = 0,
        **kwargs,
) -> Iterator[plt.Figure]:
    leader = summaries[leader]
    ap = leader.ap.sort_values('ap', ascending=False)
    loc = ap.ap.values > 0.
    if unique_iclass:
        loc &= ~ap.iclass.duplicated()
    ap = ap.loc[loc]
    it = zip(ap.iclass.values, ap.iou.values)

    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    for idx, (iclass, iou) in enumerate(it):
        # Generate PR curves and convert to PIL images
        plot_images = [
            summary.plot_pr_curve(cls=iclass, iou=iou, show=False, **kwargs)
            for summary in summaries
        ]

        # Convert each plot to a numpy array
        plot_arrays = [np.array(img) for img in plot_images]

        # Calculate total width and maximum height for the concatenated image
        total_width = sum(arr.shape[1] for arr in plot_arrays)
        max_height = max(arr.shape[0] for arr in plot_arrays)

        # Create a new blank image array
        concatenated_image_array = np.zeros((max_height, total_width, 4), dtype=np.uint8)

        # Paste each image array side by side
        x_offset = 0
        for arr in plot_arrays:
            concatenated_image_array[:arr.shape[0], x_offset:x_offset + arr.shape[1]] = arr
            x_offset += arr.shape[1]

        # Convert the concatenated image array back to a PIL image
        concatenated_image = Image.fromarray(concatenated_image_array)

        # Now use matplotlib to display the final concatenated image
        fig = plt.figure(figsize=(total_width / 100, max_height / 100))
        plt.imshow(concatenated_image)
        plt.axis('off')  # Remove axes for a clean image

        if directory:
            # Save the image to the directory
            filepath = os.path.join(directory, f'pr_curve_iclass_{iclass}_iou_{iou}.png')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        yield fig

