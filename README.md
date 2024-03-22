# Slicer nnUNet

## Table of contents

* [Introduction](#introduction)
* [Using the extension](#using-the-extension)
* [Changelog](#changelog)
* [Contributing](#contributing)

## Introduction

This module allows to install and run nnUNet trained models in 3D Slicer with only the training folder.

It streamlines development and integration of new nnUNet models into the 3D Slicer environment.

## Using the extension

This extension can be installed directly using Slicer's extension manager.

Once installed, navigate to `Segmentation>Slicer nnUnet` in the modules drop down menu or search directly
for `nnUnet`.

Once in the widget, select the model to use by using the model path selection widget.

Select the volume on which to run the model using the volume input editor.

Then click on the `Apply` button.

The logs console will display all the information regarding running the input model.

During the first launch, the nnUnetV2 module will be downloaded and installed in 3D Slicer.
The UI will ask for confirmation when installing pyTorch and other dependencies.

If you would like to install specific versions of nnUNet, you can do so by accessing
the settings using the `Gear` button in the UI.

Once the model has finished running, the segmentation will be loaded into 3D Slicer.
It can then be viewed and edited in the `Segment Editor` module.
 

## Changelog


## Contributing

This project welcomes contributions. If you want more information about how you can contribute, please refer to
the [CONTRIBUTING.md file](CONTRIBUTING.md).
