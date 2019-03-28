# Deep Finder

## Contents
- [System requirements](##System requirements)
- [Installation guide](##Installation guide)
- [Instructions for use](##Instructions for use)

## System requirements
__Deep Finder__ has been implemented using __Python 2.7__ and is based on the __Keras__ package. It has been tested on Linux (Debian 8.6), and should also work on Mac OSX as well as Windows.

### Package dependencies
Users should install following packages in order to run Deep Finder. The package versions for which our software has been tested are displayed in brackets:
```
tensorflow-gpu (1.4.0)
keras          (2.1.6)
numpy          (1.14.3)
h5py           (2.7.1)
lxml           (4.3.2)
scikit-learn   (0.19.1)     
scikit-image   (0.14.2)  
matplotlib     (2.2.3)
```

## Installation guide
First install the packages:
```
pip install numpy tensorflow-gpu keras sklearn h5py lxml scikit-learn scikit-image matplotlib
```
For more details about installing Keras, please see [Keras installation instructions](https://keras.io/#installation).

Once the dependencies are installed, the user should be able to run Deep Finder.

## Instructions for use
Instructions for using Deep Finder are contained in folder examples/. The scripts contain comments on how the toolbox should be used. To run a script, first launch ipython:
```
ipython 
```
The launch the script:
```
%run script_file.py
```