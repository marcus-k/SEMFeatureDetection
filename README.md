# SEM Ellipse Finder
Collection of work aiming to detect ellipses in SEM images. This work was done in a summer
research project from May 2022 to August 2022 under the supervision of Dr. Paul Barclay.

This will mainly be an archival tool of the progress I make on this. Commits will mainly
be various updates as I understand and discover different things. 

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Creating Ground Truth Images](#creating-ground-truth-images)
* [Development](#development)

## Installation

Installation is broken up into two section: the Python packages and the AAMED algorithm.
If the AAMED algorithm is not needed, one can skip installing it.

### Installing Python Packages

The Anaconda version of Python was used and tested in this work. Instruction for such 
will be given. If needed, first create a new environment with the desired Python 
version. Versions 3.8-3.10 have been tested.
```
$ conda create -n ellipse python=3.8
``` 
Next, install the required packages
```
$ conda install -n ellipse opencv=4.5.5 numpy pandas matplotlib tifffile ipykernel
```

OpenCV version 4.5.5 is specified since the provided AAMED binaries were compiled with
this version. Other mismatched versions may work, but to be sure, it is best to compile
the AAMED algorithm for that version if the functionality is desired. See the next 
section for details.

### Installing AAMED

In order to use the AAMED algorithm, the AAMED binaries should be downloaded and placed 
into the `./ellipsefinder/methods` folder. The binaries are provided for AAMED in the 
releases section on the right side. Provided are versions for the Anaconda distribution 
of Python 3.8-3.10 for Windows and Linux compiled for OpenCV 4.5.5. The system 
distribution in Linux should also work. For other distributions, one will need to build 
the AAMED binary themselves. 

Cython and a C++ compiler are needed for building. See the 
[AAMED GitHub page](https://github.com/Li-Zhaoxi/AAMED) for building details.

It is not necessary to install the AAMED binaries. If they are not found, an error will
be shown, but the code can be used normally without access to the AAMED algorithm.

## Usage

The main code is in `./ellipsefinder` and it consists of the main `find_ellipses.py`
file and the `methods` folder. In the `find_ellipses.py` file is all the functions
wrapping the functionality of the various ellipse finding algorithms in the `methods`
folder.

Most of the testing of the algorithms I did with the Jupyter notebooks in 
[`./notebooks`](./notebooks/). It is a bit unorganized, but examples of using 
[`find_ellipses.py`](./ellipsefinder/find_ellipses.py) are best shown in
[`ellipses.ipynb`](./notebooks/ellipses.ipynb).

In general, a couple different sections should be written. First, for the images that
we have used, we open then and remove any banners that are present. Second, we set
up any filtering that we want done on the output results. Next, run our selected
detection algorithm in the standard way seen in the examples. Finally, we can calculate
some things with results and save all the output image.

## Creating Ground Truth Images

To compare the accuracy of the ellipse finding methods, and for possible use in future 
machine learning methods, ground truth images/masks were created using 
[ImageJ](https://imagej.net/software/fiji/downloads). For the details in creating the
selection regions of interest (ROIs) and masks, see the 
[ImageJ GT Steps.md](./ImageJ%20GT%20Steps.md) file.

Three files are created from this method: the ground truth mask, the ROI zip file, and 
the CSV file of the selection regions. I tested out both extraction the ellipses from 
the mask and creating the ellipse from the CSV selection region data. Both are okay, 
using `./ellipsefinder/format_roi_csv.py` with the CSV data is probably better.

## Development

I tried to make adding new future methods relatively easy. All the current ellipse
finder algorithms implement the abstract base class (ABC) `finder.py` which has the 
various functions needed to get going. Bare minimum, the two abstract methods 
`preprocess()` and `extract()` should be implemented. `preprocess()`, if needed, should 
return a preprocessed image ready for the extraction process to start. `extract()` 
should actually implement the extraction process and return the DataFrame with the found
ellipses' details. Is this the best way to organize this process? I don't know, but I
thought it worked well for me.

The ABC also has a few convenience functions that can be utilized when creating a new
extraction method, as well as a default function to plot the found ellipses onto the
original image.

A standard that I have used when storing the data in the DataFrame is storing the angle
of the ellipse with the commonplace counterclockwise positive direction starting from
the x-axis. In the OpenCV methods I have used, it adapts a clockwise positive direction
starting from the x-axis as its standard. This is something to keep in mind when
creating the `extract()` function and storing results.