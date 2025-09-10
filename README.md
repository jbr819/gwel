# Gwel
The *gwel* Python module provides a framework for handling large image datasets and using neural networks for computer vision tasks in crop science research. The module supports object instance detection and semantic segmentation.

Maintained and created by Jack Rich, Department of Crop Science, School of Agriculture, Policy, and Development; University of Reading as part of my PhD research. 

### 0 Install via PyPI

For a quick installation of the latest stable version ( [conda](https://docs.anaconda.com/miniconda/install/) package manager recomended):

```bash
pip install gwel
```

If installing from source following these directions:

### 1 Clone this repo

```bash
git clone https://gitlab.act.reading.ac.uk/bw832080/gwel.git
cd gwel
```

### 2 Create Virtual Environment

With [conda](https://docs.anaconda.com/miniconda/install/) (recommended):

```bash
conda env create -f environment.yml
conda activate gwel
```

With venv (Linux and macOS):

```bash
python3.10 -m venv gwel 
source gwel/bin/activate
pip install -r requirements.txt
```

With venv (Windows):

```powershell
python3.10 -m venv gwel 
gwel\Scripts\activate 
pip install -r requirements.txt
```

### 3 Install gwel

```bash
pip install -e .
```

## Command Line Interface

Verify gwel installation:

```bash
gwel --version
GWEL CLI version 0.0.1a0
```
To see gwel subcommands: 
```bash
gwel --help
```

Navigate to images directory and view images with:
```bash
gwel view
#to navigate to the next or previous images use the 'n' and 'p' keys respectively.
#press the 'q' key to quit. 
#pressing the 'f' key will flag images.
```

For detailed tutorials, visit [the wiki]().

## Acknowledgments
This research is funded by the [Biotechnology and Biological Sciences Research Council (BBSRC)](https://www.ukri.org/councils/bbsrc/), part of [UK Research and Innovation (UKRI)](https://www.ukri.org/), through the [FoodBioSystems Doctoral Training Partnership (DTP)](https://research.reading.ac.uk/foodbiosystems/) as part of my PhD project at the [University of Reading](https://www.reading.ac.uk/).










