cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(NNUNet)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/KitwareMedical/SlicerNNUnet")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Thibault Pelletier (Kitware SAS)")
set(EXTENSION_DESCRIPTION "This extension is meant to streamline the integration of nnUnet based models into 3D Slicer.

It allows for quick and relable nnUNet dependency installation in 3D Slicer environment and provides simple logic to launch nnUNet prediction on given directories.

The installation steps are based on the work done in the SlicerTotalSegmentator exension (https://github.com/lassoan/SlicerTotalSegmentator)")
set(EXTENSION_ICONURL "https://github.com/KitwareMedical/SlicerNNUnet/raw/main/SlicerNNUnet/Resources/Icons/SlicerNNUnet.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/1.png")
set(EXTENSION_DEPENDS "PyTorch SlicerPythonTestRunner") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerNNUnet)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
