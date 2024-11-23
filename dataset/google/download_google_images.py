
import os
import pandas as pd
from tqdm import tqdm
import streetview

def download_google_images(csv_path: str, output_dir: str):
    """
    Download Google Street View images using metadata from the repository's CSV file.

    Parameters:
        csv_path (str): Path to the CSV file containing metadata.
        output_dir (str): Directory where the downloaded images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load metadata
    metadata = pd.read_csv(csv_path)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Downloading Google images"):
        panoid = row["panoid"]
        heading = row["heading"]
        filename = row["filename"]

        try:
            # Construct the full path for the output image
            image_path = os.path.join(output_dir, filename)

            # Download the image using the streetview library
            streetview.api_download(
                panoid=panoid,
                heading=heading,
                fov=90,  # Field of view (90 degrees is typical for cropped images)
                pitch=0,  # Pitch (tilt of the camera, 0 is level)
                output=image_path,
                width=768,
                height=768
            )
            print(f"Downloaded: {image_path}")
        except Exception as e:
            print(f"Error downloading image for Panoid {panoid}: {e}")


if __name__ == "__main__":
    # Path to the metadata CSV file included in the repository
    csv_path = "google_download.csv"  # Adjust path as needed if file is elsewhere
    output_dir = "google_images"  # Directory to save images

    # Download Google Street View images
    download_google_images(csv_path, output_dir)
