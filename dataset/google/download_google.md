## Download Google Street-View Images

To download Google Street-View imagery for use in our benchmark dataset, follow the steps below. Due to licensing restrictions, the images are not distributed directly. Instead, users must download the imagery using the provided metadata file and the included script.

---

### Prerequisites

1. **Install the `streetview` Library**  
   This script uses the `streetview` library, which you can install directly from its GitHub repository:

   ```bash
   pip install git+https://github.com/robolyst/streetview.git
   ```

2. **Metadata File**

The metadata file (`google_metadata.csv`) contains the necessary information to download the images, such as panorama ID, camera heading, and desired filenames. Below is an example format:

| Column     | Description                                   |
|------------|-----------------------------------------------|
| `panoid`   | Panorama ID (unique for each Google image)    |
| `heading`  | Camera heading angle (0-360 degrees)          |
| `filename` | Desired filename for the downloaded image     |

#### Sample Metadata Row

```csv
panoid,lat,lon,var1,heading,var3,var4,year,month,filename
55t_oREeQKvkCVBuUYsyTQ,40.82831444096637,-73.8502535669643,13.796339,169.55234,89.43216,1.7302337,2018,9,55t_oREeQKvkCVBuUYsyTQ_right_cropped.png
```

3. **Output Directory**  
   By default, images will be saved in the `google_images` directory.


---

### Running the Script

1. **Run the Script**  
   Execute the script with:

   ```bash
   python download_google_images.py
   ```

---

### Output

- The images will be saved in the `google_images` directory.
- Each image will be saved with the filename specified in the `filename` column of `google_download.csv`.

---

### Notes

- **Custom Directory**: To change the output directory, modify the `output_dir` variable in the script.
---