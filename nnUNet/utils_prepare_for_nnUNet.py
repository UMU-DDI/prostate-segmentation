import os
import SimpleITK as sitk
import numpy as np

def makeDirectory(pathToDir):

    if not os.path.exists(pathToDir):
        os.makedirs(pathToDir)


def write_image(img, dst):

    sitk.WriteImage(img, dst)


def read_image(dcm_folder):
                
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(dcm_folder)
    reader.SetFileNames(dcm_files)
    img = reader.Execute()

    return img

def resampleToReference(inputImage, referenceImage, interpolator=sitk.sitkLinear, defaultValue=0):

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImage)
    filter.SetInterpolator(interpolator)
    filter.SetDefaultPixelValue(defaultValue)

    outputImage = filter.Execute(inputImage)
    
    return outputImage