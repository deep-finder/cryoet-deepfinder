# ExoDeepFinder

ExoDeepFinder is an exocytosis event detection tool.

This work is based on [DeepFinder](https://github.com/deep-finder/cryoet-deepfinder) which has been customized for TIRF microscopy.

## Installation guide

[ExoDeepFinder binaries are available](https://github.com/deep-finder/tirfm-deepfinder/releases/tag/v0.2.3) for Windows, Linux and Mac, so there is no need to install anything if you just want to use the graphical user interface. The Linux release is big (over 4Gb) because it contains the libraries required for the GPU acceleration. Thus they are split in two parts (`ExoDeepFinder_Linux-x86_64_part1.tar.gz` and `ExoDeepFinder_Linux-x86_64_part2.tar.gz`). To uncompress them, use the following command: `tarcat ExoDeepFinder_Linux-x86_64_part*.tar.gz  | tar -xvzf -`.

> **_Note:_** ExoDeepFinder depends on Tensorflow which is only GPU-accelerated on Linux. There is currently no official GPU support for MacOS and native Windows, so the CPU will be used on those platform, but you can still use it (it will just be slower, yet the training might be very slow and can be buggy). On Windows, WSL2 can be used to run tensorflow code with GPU; see the [install instructions](https://www.tensorflow.org/install/pip?hl=fr#windows-wsl2) for more information.

### Python installation

Alternatively, to install ExoDeepFinder and use it with command lines, create and activate a virtual environment with python 3.11 or later (see the [Virtual environments](#virtual-environments) section for more information), install dependencies (on Linux only, see bellow), and run `pip install exodeepfinder[GUI]`.

On Linux you will need to install [`wxPython` dependencies](https://github.com/wxWidgets/Phoenix/blob/master/README.rst#prerequisites) manually (`sudo apt install libgtk-3-dev`, etc.) or use one [precompiled wxPython version](https://wxpython.org/pages/downloads/index.html) (use `pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04 wxPython` with your Ubuntu version number, or use `conda install wxpython` to install a compiled wxPython from conda). 

Note that on Windows, the `python` command is often replaced by `py` and `pip` by `py -m pip`; so you migth need adapt the commands in this documentation depending on your system settings.

## Usage

Here are all ExoDeepFinder commands (described later):

```
convert_tiff_to_h5              # convert tiff folders to a single h5 file
segment                         # segment a movie
generate_annotation             # generate an annotation file from a segmentation by clustering it
generate_segmentation           # generate a segmentation from an annotation file
detect_spots                    # detect bright spots in movies
merge_detector_expert           # merge the expert annotations with the detector segmentations for training
structure_training_dataset      # structure dataset files for training
train                           # train a new model
exodeepfinder                   # combine all above commands
```

The ExoDeepFinder main GUI enables to execute each of those commands (listed on the Actions panel).

### Command-line usage (Python)

All commands (except `exodeepfinder`) must be prefixed with `edf_` when using the command-line interface.

For more information about an ExoDeepFinder command, use the `--help` option (run `edf_detect_spots --help` to know more about `edf_detect_spots`).

To open a Graphical User Interface (GUI) for a given command, run it without any argument. For example, `edf_segment` opens a GUI which can execute the `edf_segment` command with the arguments specified with the graphical interface.

`exodeepfinder` runs any of the other command as a subcommand (for example `exodeepfinder segment -m movie.h5` is equivalent to `edf_segment -m movie.h5`); and it opens a GUI for all other commands when called without any argument.

If you installed ExoDeepFinder as a developer (see [Development section](## Development)), all commands can either be called directly (`edf_segment -m movie.h5`) or with python and the proper path (`python deepfinder/commands/segment.py -m movie.h5` when in the project root directory).

### Exocytosis events detection

The detection of exocytosis events is formally the segmentation of events in 3D (2D + time) TIRF movies followed by the clustering of the resulting segmentation map.

Detecting exocytosis events in ExoDeepFinder involves executing the following commands:
  1. `convert_tiff_to_h5`  to convert tiff folders to a single h5 file,
  1. `segment` to generate segmentation maps from movies, where 2s will be exocytosis events and 1s will be bright spots,
  1. `generate_annotation` to generate an annotation file from a segmentation by clustering it.

#### 1. Convert movies to h5 format

ExoDeepFinder handles exocytosis movies made from tiff files, where each tiff file is a frame of the movie, and their name ends with the frame number; like in the following structure:

```
exocytosis_data/
├── movie1/
│   ├── frame_1.tiff
│   ├── frame_2.tiff
│   └── ...
```

The frame extensions can be .tif, .tiff, .TIF or .TIFF.

There is no constraint on the file names, but they must contain the frame number (the last number in the file name must be the frame number), and be in the tiff format (it could work with other format like .png since images are read with the `skimage.io.imread()` function of the scikit-image library). For example `frame_1.tiff` could also be named `IMAGE32_1.TIF`. Similarly, there is no constraint on the movie names. In addition, although there is no strict constraint on the file names, be aware that it is much simpler to work with simple file names with no space nor special characters. Lastly, make sure that folders contain only the .tiff frame of your movie and no additional images (e.g. a mask of the cell, etc.).

The movie folders (containing the frames in tiff format) can be converted into a single `.h5` file with the `convert_tiff_to_h5` command.
Most ExoDeepFinder commands take h5 files as input, so the first step is to convert the data to h5 format with the `convert_tiff_to_h5` action in the GUI, or with the following command:
`edf_convert_tiff_to_h5 --tiff path/to/movie/folder/ --output path/to/output/movie.h5`

You can also generate all your movie folders at once using the `--batch` option. 
For example:

`edf_convert_tiff_to_h5 --batch path/to/movies/ --output path/to/outputs/ --make_subfolder`

where `path/to/movies/` contains movies folders (which in turn contains tiff files).
The `--make_subfolder` option enable to put all tiff files in a `tiff/` subfolder; which is useful in batch mode.
The `--batch` option enables to process multiple movie folders at once and work in the same way in all ExoDeepFinder commands.

The above command will turn the following file structure:

```
exocytosis_data/
├── movie1/
│   ├── frame_1.tiff
│   ├── frame_2.tiff
│   └── ...
├── movie2/
│   ├── frame_1.tiff
│   └── ...
└── ...
```

into this one:

```
exocytosis_data/
├── movie1/
│   ├── tiff/
│   │   ├── frame_1.tiff
│   │   ├── frame_2.tiff
│   │   └── ...
│   └── movie.h5
├── movie2/
│   ├── tiff/
│   |   ├── frame_1.tiff
│   │   └── ...
│   └── movie.h5
└── ...
```

#### 2. Segment movies

To generate segmentations, you can either use ExoDeepFinder or [`napari-exodeepfinder`](https://github.com/deep-finder/napari-exodeepfinder).

To segment a movie, use the `segment` action in the GUI, or the following command:
`edf_segment --movie path/to/movie.h5 --model_weights examples/analyze/in/net_weights_FINAL.h5 --patch_size 160 --visualization`

The `--patch-size` argument corresponds to the size of the input patch for the network. The movie is split in cubes of `--patch_size` voxels before being processed. `--patch_size` must be a multiple of 4. Bigger patch sizes will be faster but will take more space on your GPU.

To detect exocytosis events, you can either use the pretrained segmentation model (available in `examples/analyze/in/net_weights_FINAL.h5`), or you can annotate your exocytosis movies and train your own model (see the training section bellow).

You can omit the model weights path (`--model_weights`) if you use the release (downloaded from [here](https://github.com/deep-finder/tirfm-deepfinder/releases/)) or if you cloned the repository since the default example weights will be found automatically. Otherwise (for example if you installed with `pip install exodeepfinder`), the default weights can also be downloaded manually [here](https://github.com/deep-finder/tirfm-deepfinder/raw/master/examples/analyze/in/net_weights_FINAL.h5).

This will generate a segmentation named `path/to/movie_semgmentation.h5` with the pretrained weigths in `examples/analyze/in/net_weights_FINAL.h5` and patches of size 160. It will also generate visualization images.

This should take 10 to 15 minutes for a movie of 1000 frames of size 400 x 300 pixels on a modern CPU (mac M1) and only few dozens of seconds on a GPU.

Use the `--visualization` argument to also generate visualization images and get a quick overview of the segmentation results.

See `edf_segment --help` for more information about the input arguments.

#### 3. Generate annotations

To cluster a segmentation file and create an annotation file from it, use the `generate_annotation` action in the GUI, or the following command:
`edf_generate_annotation --segmentation path/to/movie_segmentation.h5 --cluster_radius 5`

The clustering will convert the segmentation map (here `movie_segmentation.h5`) into an event list. The algorithm groups and labels the voxels so that all voxels of the same event share the same label, and each event gets a different label. The cluster radius is the approximate size in voxel of the objects to cluster.
5 voxels is best for films with a pixel size of 160nm, for exocytosis events of 1 second and of size 300nm.

ExoDeepFinder detects both bright spots (that could be confused with exocytosis events) and genuine exocytosis events. By default, the command will ignore all bright spots (replace label "1" with 0) and will replace exocytosis events (label "2") to ones. Indeed, ExoDeepFinder is an exocytosis event detector, so its output is only composed of exocytosis events labelled with ones. Use the --keep_labels_unchanged option to skip this step and use the raw label map (segmentation) instead. This can be useful if you use a custom detector and want to check the corresponding annotations for example.

#### Using napari-exodeepfinder

The [`napari-exodeepfinder`](https://github.com/deep-finder/napari-exodeepfinder) plugin can be used to compute predictions.
Open the movie you want to segment in napari (it must be in h5 format).
In the menu, choose `Plugins > Napari DeepFinder > Segmentation`  to open the segmentation tools.
Choose the image layer you want to segment.
Select the `examples/analyze/in/net_weights_FINAL.h5` net weights; or the path of the model weights you want to use for the segmentation.
Use 3 for the number of class (0: background, 1: bright spots, 2: exocytosis events), and 160 for the patch size.
Choose an output image name (with the .h5 extension), then launch the segmentation.

### Training

The training is really slow and can generate some bugs when runned on the CPU. We recommend using Linux (or eventually WLS2 on Windows) to take benefit of the GPU (see the "Installaton Guide" section).
To train a model, your data should be organized in the following way:

```
exocytosis_data/
├── movie1/
│   ├── frame_1.tiff
│   ├── frame_2.tiff
│   └── ...
├── movie2/
│   ├── frame_1.tiff
│   └── ...
└── ...
```

#### 1. Convert movies to h5 format

For each movie, tiff files must be converted to a single `.h5` using the `convert_tiff_to_h5` action from the GUI, or the `edf_convert_tiff_to_h5` command, as explained in the [Exocytosis events detection section](#Exocytosis-events-detection):

`edf_convert_tiff_to_h5 --batch path/to/exocytosis_data/ --make_subfolder`

This will change the `exocytosis_data` structure into the following one:

```
exocytosis_data/
├── movie1/
│   ├── tiff/
│   │   ├── frame_1.tiff
│   │   ├── frame_2.tiff
│   │   └── ...
│   └── movie.h5
├── movie2/
│   ├── tiff/
│   |   ├── frame_1.tiff
│   │   └── ...
│   └── movie.h5
└── ...
```

#### 2. Detect bright spots

ExoDeepFinder can generate false positives by confusing bright spots with genuine exocytosis events. The strategy to reduce this type of false positive is to explicitly present these bright spots as counter-examples during the training. Hence,  the training requires bright spots to be annotated. You can use any suitable methods that will accurately detect counter-examples bright spots in your data, or use our spot detector [Atlas](https://gitlab.inria.fr/serpico/atlas). The Atlas installation instructions are detailed in the repository, but the most simple way of installing it is by using conda: `conda install bioimageit::atlas`.

Once atlas (or the detector of your choice) is installed, you can detect spots in each frame using the `detect_spots` action in the GUI, or the `edf_detect_spots` command:

`edf_detect_spots --detector_path path/to/atlas/ --batch path/to/exocytosis_data/`

where `path/to/atlas/` is the root path of atlas (containing the `build/` directory with the binaries inside if you followed the manual installation instructions).

This will generate `detector_segmentation.h5` files (the semgentations of spots) in the movie folders:

```
exocytosis_data/
├── movie1/
│   ├── tiff/
│   │   ├── frame_1.tiff
│   │   ├── frame_2.tiff
│   │   └── ...
│   ├── detector_segmentation.h5
│   └── movie.h5
├── movie2/
└── ...
```

You have two possibilities if you want to use an alternative detector:

1) Call a function with ExoDeepFinder. Make sure your detector generates segmentation maps with 1s where there are bright spots (no matter whether they are exocytosis events or not) and 0s elsewhere. You can specify the command to call the detector with the `--detector_command` and/or the `--detector_path` arguments. For example `edf_detect_spots --detector_path path/to/atlas/ --batch path/to/exocytosis_data/ --detector_path path/to/custom_detector.py --detector_command 'python "{detector}" -i "{input}" -o "{output}"'` will call `custom_detector.py` with each each movies in the dataset like so: `python path/to/custom_detector.py -i path/to/exocytosis_data/movieN/tiff/ -o path/to/exocytosis_data/movieN/detector_segmentation.h5`. The detector will have to handle all `.tiff` frames and generate a segmentation in the `.h5` format.

You can make sure that the detector segmentations are correct by opening them in napari with the corresponding movie. Open both `.h5` files in napari, put the `detector_segmentation.h5` layer on top, then right-click on it and select "Convert to labels". You should see the detections in red on top of the movie.`

2) Use an independent program (e.g. ImageJ). Use the software of your choise to obtain bright spot coordinates (no mattter whether they are exocytosis events or not) and save these annotations in a .csv (or .xml) with the following format:

```
tomo_idx,class_label,x,y,z
0,1,133,257,518
0,1,169,230,519
0,1,184,237,534
0,1,146,260,546
```


Note that one can convert annotations (.xml or .csv files describing bright spots) to segmentation maps (.h5 files) with the `edf_generate_segmentation` command, and segmentation maps to annotations with the `edf_generate_annotation` command. This can be useful if you use your own detector which generates either annotations or segmentations.

#### 3. Annotate exocytosis events

The training requires movies to be annotated with the localizations of exocytosis events and bright spots. The recommended way to annotate exocytosis events is to use the [`napari-exodeepfinder` plugin](https://github.com/deep-finder/napari-exodeepfinder) but it is also possible to use other software (e.g. ImageJ) as long as the output annotations respect the format described below.

Annotate the exocytosis events in the movies with the `napari-exodeepfinder` plugin:

- Follow the install instructions, and open napari.
- In the menu, choose `Plugins > Napari DeepFinder > Annotation`  to open the annotation tools.
- Open a movie (for example `exocytosis_data/movie1/movie.h5`).
- Create a new points layer, and name it `movie_1` (any name with the `_1` suffix, since we want to annotate with the class 1). 
- In the annotation panel, select the layer you just created in the "Points layer" select box (you can skip this step and use the "Add points" and "Delete selected point" buttons from the layer controls).
- You can use the Orthoslice view to easily navigate in the volume, by using the `Plugins > Napari DeepFinder > Orthoslice view` menu.
- Scroll in the movie until you find and exocytosis event.
- If you opened the Orthoslice view, you can click on an exocytosis event to put the red cursor at its location, then click the "Add point" button in the annotation panel to annotate the event.
- You can also use the "Add points" and "Delete selected point" buttons from the layer controls.
- When you annotated all events, save your annotations to xml by choosing the `File > Save selected layer(s)...` menu, or by using ctrl+S (command+S on a mac), **and choose the *Napadi DeepFinder (\*.xml)* format**. Save the file beside the movie, and name it `expert_annotation.xml` (this should result in the `exocytosis_data/movie1/expert_annotation.xml` with the above example).

Annotate all training and validation movies with this procedure; you should end up with the following folder structure:

```
exocytosis_data
├── movie1/
│   ├── tiff/
│   │   ├── frame_1.tiff
│   │   ├── frame_2.tiff
│   │   └── ...
│   ├── detector_segmentation.h5
│   ├── expert_annotation.xml
│   └── movie.h5
├── movie2/
└── ...
```

Make sure that the `expert_annotation.xml` files you just created have the following format:

```
<objlist>
  <object tomo_idx="0" class_label="1" x="71" y="152" z="470"/>
  <object tomo_idx="0" class_label="1" x="76" y="184" z="445"/>
  <object tomo_idx="0" class_label="1" x="141" y="150" z="400"/>
  <object tomo_idx="0" class_label="1" x="200" y="237" z="420"/>
  <object tomo_idx="0" class_label="1" x="95" y="229" z="438"/>
  ...
</objlist>
```

If you used an alternative software (e.g. ImageJ) other than `napari-exodeepfinder` to annotate exocytosis events, make sure your output files follow the same structure. It can be `csv` files, but they must follow the same naming, as in the following `example.csv`:

```
tomo_idx,class_label,x,y,z
0,1,133,257,518
0,1,169,230,519
0,1,184,237,534
0,1,146,260,546
```

The `class_label` must be 1, and `tomo_idx` must be 0.

#### 4. Convert expert annotations to expert segmentations

Convert your manual annotations (named expert annotations) into expert segmentations so that they can be merged with the detected bright spots and used for the training.

Use the `generate_segmentation` action in the GUI, or the following command to convert the annotations to segmentations:

`edf_generate_segmentation --batch path/to/exocytosis_data/`

You will end up with the following structure:

```
exocytosis_data/
├── movie1/
│   ├── tiff/
│   │   ├── frame_1.tiff
│   │   ├── frame_2.tiff
│   │   └── ...
│   ├── detector_segmentation.h5
│   ├── expert_annotation.xml
│   ├── expert_segmentation.h5
│   └── movie.h5
├── movie2/
└── ...
```

Note that the expert annotation can be a `.csv` as long as it respects the correct labeling.

Again, you can check on napari that everything went right by opening all images and checking that `expert_segmentation.h5` corresponds to `expert_annotation.xml` and the movie.

#### 5. Merge detector and expert data

Then, merge detector detections with expert annotations with the `merge_detector_expert` action in the GUI, or the `edf_merge_detector_expert` command:

`edf_merge_detector_expert --batch path/to/exocytosis_data/`

This will create two new files `merged_annotation.xml` (the merged annotations) and `merged_segmentation.h5` (the merged segmentations). The exocytosis events are first removed from the detector segmentation (`detector_segmentation.h5`), then the remaining events (from the detector and the expert) are transferred to the merged segmentation (`merged_segmentation.h5`), with class 2 for exocytosis events and class 1 for others events. The maximum number of other events in the annotation is 9800; meaning that if there are more than 9800 other events, only 9800 events will be picked randomly and the others will be discarded. If you have >>9800 annotations with a vast majority of bright spots, exocytosis events can be under-represented and we recommend to obtain a more specific annotation of the bright spots.

The `exocytosis_data/` folder will then follow this structure:

```
exocytosis_data/
├── movie1/
│   ├── tiff/
│   │   ├── frame_1.tiff
│   │   ├── frame_2.tiff
│   │   └── ...
│   ├── detector_segmentation.h5
│   ├── expert_annotation.xml
│   ├── expert_segmentation.h5
│   ├── merged_annotation.xml
│   ├── merged_segmentation.h5
│   └── movie.h5
├── movie2/
└── ...
```

Again, make sure everything looks right in napari.

#### 6. Organize training files

Finally, the training data should be organized in the following way:

```
dataset/
├── train/
│   ├── movie1.h5
│   ├── movie1_objl.xml
│   ├── movie1_target.h5
│   ├── movie2.h5
│   ├── movie2_objl.xml
│   ├── movie2_target.h5
...
└── valid/
    ├── movie3.h5
    ├── movie3_objl.xml
    ├── movie3_target.h5
...
```

This structure can be obtained with the `structure_training_dataset` action in the GUI, or by using the `edf_structure_training_dataset` command:

`edf_structure_training_dataset --input path/to/exocytosis_data/ --output path/to/dataset/`

This will organize the input folder (which should be structured as in the previous step) with the above final structure, by putting 70% of the movies in the train/ folder, and 30% of them in the valid/ folder.

Make sure the output folder is correct, and that you can open its content in napari.

#### 7. Train your custom model

Finally, launch the training with `train` action in the GUI, or the command `edf_train --dataset path/to/dataset/ --output path/to/model/`.

#### Summary

Here are all the steps you should execute to train a new model:

1. Convert tiff frames to h5 file: `edf_convert_tiff_to_h5 --batch path/to/exocytosis_data/ --make_subfolder`
1. Use [`napari-exodeepfinder`](https://github.com/deep-finder/napari-exodeepfinder) to annotation exocytosis events in the movies
1. Detect all spots: `edf_detect_spots --detector_path path/to/atlas/ --batch path/to/exocytosis_data/`
1. Generate detector segmentations: `edf_generate_segmentation --batch path/to/exocytosis_data/`
1. Merge expert and detector segmentation: `edf_merge_detector_expert --batch path/to/exocytosis_data/`
1. Structure the files: `edf_structure_training_dataset --dataset path/to/exocytosis_data/ --training path/to/dataset/`
1. Train the model: `edf_train --dataset path/to/dataset/ --output path/to/model/`

## Virtual environments & package managers

There are two major ways of creating virtual environments in Python: venv and conda ; and two major ways of installing packages: pip and conda.

### Virtual environment: venv & conda

The simplest way of creating a virtual environment in python is to use [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments). Make sure your Python version greater or equal to 3.10, and simply run `python -m venv ExoDeepFinder/` (`py -m venv ExoDeepFinder/` on Windows) to create your environment (replace `ExoDeepFinder` by the name you want for your environment). Then run `source ExoDeepFinder/bin/activate` to activate it (`ExoDeepFinder\Scripts\activate` on Windows).

Alternatively, you can use [Conda](https://conda.io/projects/conda/en/latest/index.html) (or a nice minimalist alternative like [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), see bellow) to create a Python 3.10 environment, even if your python version is different.

Once conda is installed, run `conda create -n ExoDeepFinder python=3.10` to create the environment with python 3.10, and `conda activate ExoDeepFinder` to activate it.

#### Conda alternatives

The simplest way to install and use Conda is via [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), which a minimalist drop-in replacement. Once you installed it, just use `micromamba` instead of `conda` for all you conda commands (some unusual commands might not be implemented in micromamba, but it is really sufficient for most use cases). 

For example, run `micromamba create -n ExoDeepFinder python=3.10` to create the environment with python 3.10, and `micromamba activate ExoDeepFinder` to activate it.

### Package managers: pip & conda

The [Numpy documentation](https://numpy.org/install/#pip--conda) explains the main differences between pip and conda:

> The two main tools that install Python packages are `pip` and `conda`. Their functionality partially overlaps (e.g. both can install `numpy`), however, they can also work together. We’ll discuss the major differences between pip and conda here - this is important to understand if you want to manage packages effectively.

> The first difference is that conda is cross-language and it can install Python, while pip is installed for a particular Python on your system and installs other packages to that same Python install only. This also means conda can install non-Python libraries and tools you may need (e.g. compilers, CUDA, HDF5), while pip can’t.

> The second difference is that pip installs from the Python Packaging Index (PyPI), while conda installs from its own channels (typically “defaults” or “conda-forge”). PyPI is the largest collection of packages by far, however, all popular packages are available for conda as well.

> The third difference is that conda is an integrated solution for managing packages, dependencies and environments, while with pip you may need another tool (there are many!) for dealing with environments or complex dependencies.

## Development

To install ExoDeepFinder for development, clone the repository (`git clone git@github.com:deep-finder/tirfm-deepfinder.git`), create and activate a virtual environment (see section above), and install it with `pip install -e ./tirfm-deepfinder/[GUI]`.

To generate the release binaries, install PyInstaller with `pip install pyinstaller` in your virtual environment; and package ExoDeepFinder with `pyinstaller exodeepfinder.spec`. You must run this command on the destination platform (run on Windows for a Windows release, on Mac for a Mac release, and Linux for a Linux release).
