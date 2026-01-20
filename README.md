# Gwel
The *gwel* Python package provides a framework for handling large image datasets and using neural networks for computer vision tasks in scientific research. The module supports object instance detection and semantic segmentation. The flowchart below outlines some of the workflows that can be achieved using this module. 

![flowchart](flowchart.png)

Maintained and created by Jack Rich, Department of Crop Science, School of Agriculture, Policy, and Development; University of Reading as part of my PhD research. 

### 0 Install via PyPI

For a quick installation of the latest stable version ( [conda](https://docs.anaconda.com/miniconda/install/) package manager recomended):

```bash
pip install gwel
```

If installing from source following these directions:

### 1 Clone this repo

```bash
git clone https://github.com/jbr819/gwel.git
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
GWEL CLI version x.y.z
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

For detailed tutorials, visit [the wiki](https://jbr819.github.io/gwel/).

## Acknowledgments
I created this Python package to facilitate the completion of my PhD research. I was funded by the [Biotechnology and Biological Sciences Research Council (BBSRC)](https://www.ukri.org/councils/bbsrc/), part of [UK Research and Innovation (UKRI)](https://www.ukri.org/), through the [FoodBioSystems Doctoral Training Partnership (DTP)](https://research.reading.ac.uk/foodbiosystems/) as a student at the [University of Reading](https://www.reading.ac.uk/).










