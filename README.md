# Deep Finder

The code in this repository is described in [this pre-print](https://hal.inria.fr/hal-01966819/document). This paper has been submitted to Nature Communications and is currently under revision.

__Disclaimer:__ this is a preliminary version of the code, which is subject to improvements. For ease of use, we also plan to create a graphical interface in forthcoming months.

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
Then launch the script:
```
%run script_file.py
```

__Note:__ working examples are contained in examples/analyze/, where Deep Finder processes the test tomogram from the [SHREC'19 challenge](http://www2.projects.science.uu.nl/shrec/cryo-et/). The script in examples/training/ will fail because the training data is not included in this Gitlab. In addition, the evaluation script (examples/analyze/step3_launch_evaluation.py) is the one used in SHREC'19, which needs python3 and additional packages. The performance of Deep Finder has been evaluated by an independent group, and the result of this evaluation has been published in Gubins & al., "SHREC'19 track: Classification in cryo-electron tomograms".