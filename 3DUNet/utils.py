import os
import numpy as np
import math
import SimpleITK as sitk


# Function to create a directory if it doesn't exist
def makeDirectory(pathToDir):
    if not os.path.exists(pathToDir):
        os.makedirs(pathToDir)


# Function to normalize pixel values of an image between upper and lower percentiles
def normalize(image, upper_percentile, lower_percentile):
    # Extract pixel values from the image
    image_array = sitk.GetArrayFromImage(image).flatten()

    # Calculate upper and lower percentiles
    upperPerc = np.percentile(image_array, upper_percentile)
    lowerPerc = np.percentile(image_array, lower_percentile)

    # Set up filters for casting and intensity windowing
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    normalizationFilter = sitk.IntensityWindowingImageFilter()
    normalizationFilter.SetOutputMaximum(1.0)
    normalizationFilter.SetOutputMinimum(0.0)
    normalizationFilter.SetWindowMaximum(upperPerc)
    normalizationFilter.SetWindowMinimum(lowerPerc)
    
    # Apply normalization to the image
    img = castImageFilter.Execute(image)
    image_normalized = normalizationFilter.Execute(img)
    
    return image_normalized 
    
    
# Function to pad image and segmentation to required size in z-direction
def padd(image, segmentation, output_size):
    # Caluculate the amount of padding needed in the z-axis
    z_padding = output_size - image.GetSize()[2]
    
    # Calculate lower and upper padding bounds
    pad_lower = math.floor(z_padding / 2)
    pad_upper = math.ceil(z_padding / 2)
    
    # Apply padding if needed
    if z_padding > 0:
    
        image_padding_filter = sitk.ConstantPadImageFilter()
        image_padding_filter.SetPadLowerBound([0,0,pad_lower])
        image_padding_filter.SetPadUpperBound([0,0,pad_upper])
        image_padding_filter.SetConstant(0)

        image = image_padding_filter.Execute(image)
        segmentation = image_padding_filter.Execute(segmentation)
        
    return image, segmentation

# Function to calculate rotation matrix from axis-angle representation
# This function is from https://github.com/rock-learning/pytransform3d/blob/7589e083a50597a75b12d745ebacaa7cc056cfbd/pytransform3d/rotations.py#L302
def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """

    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    return R


# Function to resample an image to a new spacing using a specified interpolator
def  resampleImage(inputImage, newSpacing, interpolator, defaultValue=0):

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing = inputImage.GetSpacing()
    
    oldSpacing_rounded = [ round(oldSpacing[0], 4), round(oldSpacing[1], 4), round(oldSpacing[2], 4) ]
    
    newWidth = oldSpacing_rounded[0] / newSpacing[0] * oldSize[0]
    newHeight = oldSpacing_rounded[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing_rounded[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    # Set up resampling filter
    filter = sitk.ResampleImageFilter()
    inputImage.GetSpacing()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(inputImage.GetOrigin())
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(newSize)
    filter.SetDefaultPixelValue(defaultValue)
    outImage = filter.Execute(inputImage)

    return outImage


# Function to get image and segmentation data from a specified directory
def getData(inputDir):
    sub_folders = os.listdir(inputDir)
    
    for sub_folder in sub_folders:

        # Check for segmentation file
        if ('Seg' and '.nrrd' in sub_folder):
            seg = sitk.ReadImage(os.path.join(inputDir, sub_folder))    
            continue
        
        # Check for DICOM directory with axial image
        if os.path.isdir(os.path.join(inputDir, sub_folder)) and 'tra' in sub_folder:
            dcm_folder = os.path.join(inputDir, sub_folder)
            
            # Read DICOM series and return image
            reader = sitk.ImageSeriesReader()
            dcm_files = reader.GetGDCMSeriesFileNames(dcm_folder)
            reader.SetFileNames(dcm_files)
            img = reader.Execute()
            continue
          
    return  img, seg