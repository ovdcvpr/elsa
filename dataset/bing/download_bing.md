
### Download Bing Streetside Images

This section describes how to download Bing Streetview imagery for use in our benchmark dataset. Due to licensing restrictions, the images are not distributed directly. Instead, users must download the imagery using their own Bing Maps API key and the provided metadata file.

---

#### Prerequisites

**Bing Maps API Key**  
   You need an API key to access Bing Maps imagery.  
   [Get an API key here](https://www.microsoft.com/en-us/maps/create-a-bing-maps-key).

---

#### Metadata File

The metadata file (`bing_metadata.csv`) contains information necessary to download the images, such as latitude, longitude, heading, and desired filenames. Below is an example format:

| Column     | Description                                  |
|------------|----------------------------------------------|
| `id`       | Unique identifier for the image             |
| `bubbleId` | Optional identifier for internal tracking   |
| `lon`      | Longitude of the image location             |
| `lat`      | Latitude of the image location              |
| `heading`  | Camera heading angle (0-360 degrees)        |
| `filename` | Desired filename for the downloaded image   |

#### Sample Metadata Row

```csv
id,bubbleId,lon,lat,timestamp,cd,altitude,heading,extract,filename
1338985277,1033303310230331,-73.956662,40.743238,7/7/2021 17:19:00-04,7/7/2021 5:19:00 PM,4.645,105.025,17,103330331023033110_x4_cropped.png
```

---

#### Script Usage

The provided script automates the process of downloading imagery based on the metadata file. Follow these steps to use it:

1. **Save the Metadata File**  
   Ensure the metadata file (`metadata.csv`) is in the root directory of the repository.

2. **Update Your API Key**  
   Replace `"YOUR_BING_API_KEY"` in the script with your actual Bing Maps API key.

3. **Run the Script**  

   Execute the script to download the images:

   ```bash
   python download_bing_images.py
   ```
---

#### Notes

- **Output Directory**: Images are saved to the `bing_images` directory. You can specify a different directory by updating the `output_dir` parameter.
- **Image Size**: Images are downloaded at a resolution of **1024x1024 pixels**.
---