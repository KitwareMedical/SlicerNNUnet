# Slicer nnUNet

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/1.png" width="800"/>

## Table of contents

* [Introduction](#introduction)
* [Acknowledgments](#acknowledgments)
* [Using the extension](#using-the-extension)
* [Changelog](#changelog)
* [Contributing](#contributing)

## Introduction

<div style="text-align:center">
<img class="center" src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/SlicerNNUnet/Resources/Icons/SlicerNNUnet.png"/>
</div>

This module allows to install and run nnUNet trained models in 3D Slicer with only the training folder.

It streamlines development and integration of new nnUNet models into the 3D Slicer environment.

## Acknowledgments

This module was originally co-financed by the 
<a href="https://orthodontie-ffo.org/">Fédération Française d\'Orthodontie</a> (FFO) as part of the 
<a href="https://github.com/gaudot/SlicerDentalSegmentator/">Dental Segmentator</a> 
developments and the <a href="https://rhu-cosy.com/en/accueil-english/">Cure Overgrowth Syndromes</a> 
(COSY) RHU Project.

The installation steps are based on the work done in the 
<a href="https://github.com/lassoan/SlicerTotalSegmentator/">Slicer Total Segmentator extension</a>.

This extension interfaces 3D Slicer with the 
<a href="https://github.com/MIC-DKFZ/nnUNet">nnUNet library</a>.

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.


## Using the extension

This extension can be installed directly using Slicer's extension manager.

Once installed, navigate to `Segmentation>nnUNet` in the modules drop down menu or search directly
for `nnUnet`.

Once in the widget, you can install nnUNet's dependencies by clicking on the "nnUNet Install" button.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/2.png"/>

This area will display the current version of nnUNet in 3D Slicer's environment.
If nnUNet is not yet installed, the current version will display None.
It can be installed by clicking on the install Button.

After the install is complete, the current version should display the latest nnUNet version or the version as set in
the "To install" field.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/3.png"/>

Note that the extension may require a 3D Slicer restart before running the inference.

Once the nnUNet is correctly installed, click on the 'nnUNet Run Settings' button to set the path to the Model to use.
This path will be saved for further usage after the first segmentation.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/4.png"/>

Select the volume on which to run the model using the volume input editor.

Then click on the `Apply` button.

The logs console will display all the information regarding running the input model.

Once the model has finished running, the segmentation will be loaded into 3D Slicer with its associated labels.
It can then be viewed and edited in the `Segment Editor` module.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/1.png"/>

## Contributing

This project welcomes contributions. It follows Slicer's rules for contributions. 
If you want more information about how you can contribute, please refer to
the [CONTRIBUTING.md file](https://github.com/Slicer/Slicer/blob/main/CONTRIBUTING.md).
