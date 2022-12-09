# Deep Finder

The code in this repository is described in [this pre-print](https://www.biorxiv.org/content/10.1101/2020.04.15.042747v1). This paper has now been [published](https://doi.org/10.1038/s41592-021-01275-4) in Nature Methods.

__Disclaimer:__ DeepFinder is still in its early stages, any feedback is welcome for enhancing the user experience.

__News__: (29/01/20) A first version of the GUI is now available in folder pyqt/. [More information...](###Using the GUI) 

## Contents
- [System requirements](##System requirements)
- [Installation guide](##Installation guide)
- [Instructions for use](##Instructions for use)
- [Documentation](https://cryoet-deepfinder.readthedocs.io/en/latest/)
- [Google group](https://groups.google.com/g/deepfinder)

## System requirements
__Deep Finder__ has been implemented using __Python 3__ and is based on the __Keras__ package. It has been tested on Linux (Debian 10), and should also work on Mac OSX as well as Windows.

The algorithm needs an __Nvidia GPU__ and __CUDA__ to run at reasonable speed (in particular for training). The present code has been tested on Tesla K80 and M40 GPUs. For running on other GPUs, some parameter values (e.g. patch and batch sizes) may need to be changed to adapt to available memory.

### Package dependencies
Deep Finder depends on following packages. The package versions for which our software has been tested are displayed in brackets:
```
tensorflow     (2.6.0)
keras          (2.6.0)
numpy          (1.19.5)
h5py           (3.1.0)
lxml           (4.3.4)
scikit-learn   (0.21.2)     
scikit-image   (0.15.0)  
matplotlib     (3.1.0)
mrcfile        (1.1.2)
PyQt5          (5.13.2)
pyqtgraph      (0.10.0)
```

## Installation guide
Before installation, you need a python environment on your machine. If this is not the case, we advise installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

(Optional) Before installation, we recommend first creating a virtual environment that will contain your DeepFinder installation:
```
conda create --name dfinder python=3.7
conda activate dfinder
```

Now, you can install DeepFinder with pip:
```
pip install cryoet-deepfinder
```

Also, in order for Keras to work with your Nvidia GPU, you need to install CUDA. Once these steps have been achieved, the user should be able to run DeepFinder.

## Instructions for use
### Using the scripts
Instructions for using Deep Finder are contained in folder examples/. The scripts contain comments on how the toolbox should be used. To run a script, first place yourself in its folder. For example, to run the target generation script:
```
cd examples/training/
python step1_generate_target.py
```

### Using the GUI
The GUI (Graphical User Interface) should be more intuitive for those who are not used to work with script. Currently, 5 GUIs are available (tomogram annotation, target generation, training, segmentation, clustering) and allow the same functionalities as the scripts in example/. To run a GUI, first open a terminal. For example, to run the segmentation GUI:
```
segment
```

![Training GUI](./images/gui_segment.png)

For more informations about how to use DeepFinder, please refer to the [documentation](https://cryoet-deepfinder.readthedocs.io/en/latest/).

__Notes:__ 
- working examples are contained in examples/analyze/, where Deep Finder processes the test tomogram from the [SHREC'19 challenge](http://www2.projects.science.uu.nl/shrec/cryo-et/2019/). 
- The script in examples/training/ will fail because the training data is not included in this Gitlab. 
- The evaluation script (examples/analyze/step3_launch_evaluation.py) is the one used in SHREC'19, which needs additional packages (pathlib and pycm, can be installed with pip). The performance of Deep Finder has been evaluated by an independent group, and the result of this evaluation has been published in Gubins & al., "SHREC'19 track: Classification in cryo-electron tomograms".
