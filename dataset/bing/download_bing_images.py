
import os
import pandas as pd
from tqdm import tqdm
import requests

def download_bing_images_from_csv(
    csv_path: str,
    output_dir: str,
    api_key: str,
    image_size: int = 1024
):
    """
    Download Bing street-level imagery using metadata from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file containing metadata.
        output_dir (str): Directory where the downloaded images will be saved.
        api_key (str): Bing API key.
        image_size (int): Size of the image (both width and height, default is 1024).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load metadata
    metadata = pd.read_csv(csv_path)

    # Bing Maps API endpoint
    base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Metadata/Streetside"

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Downloading Bing images"):
        image_id = row["id"]
        latitude = row["lat"]
        longitude = row["lon"]
        heading = row["heading"]
        filename = row["filename"]

        # Construct API request
        request_url = (
            f"{base_url}/{latitude},{longitude}"
            f"?key={api_key}&heading={heading}&format=png&width={image_size}&height={image_size}"
        )

        try:
            # Fetch metadata and download the image
            response = requests.get(request_url)
            response.raise_for_status()

            # Parse metadata response to get the image URL
            image_metadata = response.json()
            if "resourceSets" in image_metadata and image_metadata["resourceSets"]:
                image_url = image_metadata["resourceSets"][0]["resources"][0]["imageUrl"]
                image_response = requests.get(image_url)
                image_response.raise_for_status()

                # Save the image
                image_path = os.path.join(output_dir, filename)
                with open(image_path, "wb") as f:
                    f.write(image_response.content)
                print(f"Downloaded: {image_path}")
            else:
                print(f"No imagery available for ID {image_id} at {latitude}, {longitude}")

        except Exception as e:
            print(f"Error downloading image for ID {image_id}: {e}")

if __name__ == "__main__":
    # User inputs
    csv_path = "metadata.csv"  # Path to your metadata CSV file
    output_dir = "bing_images"  # Directory to save images
    api_key = "YOUR_BING_API_KEY"  # Replace with your Bing API key

    # Ensure image size is 1024x1024
    image_size = 1024

    # Download images
    download_bing_images_from_csv(csv_path, output_dir, api_key, image_size)
