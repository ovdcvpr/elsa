# ELSA: evaluating localization of social activities

## Dataset

Inside the `dataset` folder we include two subfolders `(google | bing)` containing csv files named `matched_rows.csv`.
The `csv` file contains coordinates, ids, datetime and filenames of the images included in ELSA.
There are several libraries available which allows to download from the Google and Bing APIs.
For your convenience we included two scripts which use the streetlevel library. Please follow the instructions in
the respective folders to download the images:

- [Bing](dataset/bing/download_bing.md)
- [Google](dataset/google/download_google.md)

## Benchmark

### Creating a Virtual Environment

(You may choose your own method of creating a virtual environment)
Here, we install mamba and create a virtual environment with python 3.12

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh -b -p ~/mambaforge
~/mambaforge/bin/mamba init
source ~/.bashrc
mamba create -n elsa python=3.12.3
mamba activate elsa
```

### Installing ELSA

Clone the ELSA repository and install

```bash
cd ~
git clone https://github.com/ovdcvpr/SIRiUS.git
cd SIRiUS
pip install -e .
```

### Instantiating ELSA

This toolset encapsulates everything you need to interact with your dataset.:

```python
from elsa import *

bing = 'yourpath/label_1k/bing/images'
google = 'yourpath/label_1k/google/images'
files = bing, google
elsa = Elsa.from_unified(files)
```

## How to Use ELSA and Visualize Results

Our Jupyter Notebooks provide detailed, step-by-step guidance on how to work with ELSA.
These resources are designed to help you:

- **Understand the Dataset**: Explore the structure and annotations in ELSA.
- **Evaluate Models**: Learn how to benchmark open-vocabulary detection models using ELSA.
- **Visualize Results**: Generate professional visualizations of model predictions and ground truth annotations.
- **Adapt Workflows**: Customize the provided workflows to fit your specific research needs.

### Notebooks available:

1. [View Resources and Metadata](notebooks/01-resources.ipynb)
2. [View Bounding Boxes](notebooks/02-combos.ipynb)
3. [Make Predictions](notebooks/03-predict.ipynb)
4. [View Predictions](notebooks/04-prediction.ipynb)
5. [Evaluate Predictions](notebooks/05-evaluate.ipynb)

## Models Supported

For each of the models, the instructions are provided when you attempt to select the respective method.
For example, if GDINO is not installed, `elsa.predict.gdino` will give you the installation instructions.

### NLSE Benchmark Models

- [GDINO](https://github.com/longzw1997/Open-GroundingDino)
    - elsa.predict.gdino(...)
- [MDETR](https://github.com/ashkamath/mdetr)
    - elsa.predict.mdetr(...)

### DBA-AP Benchmark Models

- [DETIC](https://github.com/facebookresearch/Detic)
    - elsa.predict.detic(...)
- [GDINO](https://github.com/longzw1997/Open-GroundingDino)
    - elsa.predict.gdino(...)
- [MDETR](https://github.com/ashkamath/mdetr)
    - elsa.predict.mdetr(...)
- [OVDINO](https://github.com/IDEA-Research/detrex)
    - elsa.predict.ovdino(...)
- [OWL](https://github.com/huggingface/transformers)
    - elsa.predict.owl(...)
- [OWLv2](https://github.com/huggingface/transformers)
    - elsa.predict.owlv2(...)

