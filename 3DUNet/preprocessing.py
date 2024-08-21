import os
import numpy as np
import SimpleITK as sitk

from utils import getData, resampleImage, matrix_from_axis_angle, normalize, padd


# Function for preprocessing an image with corresponding segmentation
def startPreprocess(imgDir, event='Train'):
    
    # Define resolution and output size
    resolution = [0.5, 0.5, 3]
    output_size = [192, 192, 32]

    # Load image and segmentation data. Output is two sitk-images.
    img, seg = getData(imgDir)

    # Resample image and segmentation to the desired resolution
    img = resampleImage(img, newSpacing=resolution, interpolator=sitk.sitkLinear)
    seg = resampleImage(seg, newSpacing=resolution, interpolator=sitk.sitkNearestNeighbor)

    # Threshold and determine the center of the image
    mask = sitk.BinaryThreshold(img, lowerThreshold=0, upperThreshold=1e10, insideValue=1, outsideValue=0)
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(mask)
    center = filter.GetCentroid(1)

    # Define rotation and scaling parameters based on event (Train/Validate)
    if event == 'Train':
        rot = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.925, 1/0.925)
    elif event == 'Validate':
        rot = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.975, 1/0.975)
    else:
        rot = 0
        scale = 1

    # Set up the similarity transform with rotation and scaling
    transform = sitk.Similarity3DTransform()
    transform.SetCenter(center)
    direction = np.array(img.GetDirection()).reshape(3,3)
    axis_angle = (direction[0, 2], direction[1, 2], direction[2, 2], np.deg2rad(rot))
    np_rot_mat = matrix_from_axis_angle(axis_angle)

    transform.SetMatrix(np_rot_mat.flatten().tolist())
    transform.SetScale(scale)

    # Apply transformation
    img_transformed = sitk.Resample(img, transform, sitk.sitkLinear, 0)
    seg_transformed = sitk.Resample(seg, transform, sitk.sitkNearestNeighbor, 0)

    # Define noise parameters
    if event == 'Train':
        noise = np.random.uniform(0, 0.1)
    elif event == 'Validate':
        noise = 0
    else:
        noise = 0

    # Add Gaussian noise
    noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
    noise_filter.SetMean(0)
    noise_filter.SetStandardDeviation(noise)
    img_noise = noise_filter.Execute(img_transformed)


    # Padd image to required output size.
    img_padded, seg_padded = padd(img_noise, seg_transformed, output_size[2])
    
    # Get Translation
        
        # According to PI-QUALS the prostate image FOV (in-plane) should be 12-20cm. Our bounding box is 9.6cm.
        # This means that we have 2.4 - 10.4 cm to move around in. We will restrict movement to be ≤ 2cm in the x- and y-direction.

    if event == 'Train':
        maximum_movement_px = 4
    elif event == 'Validate':
        maximum_movement_px = 2
    else:
        maximum_movement_px = 0


    movement = [i * maximum_movement_px for i in resolution[:2]]
    if maximum_movement_px > 0: 
        rand_x = np.random.randint(-movement[0], movement[0])
        rand_y = np.random.randint(-movement[1], movement[1])
    else:
        rand_x = 0
        rand_y = 0
    center_idx = img_padded.TransformPhysicalPointToIndex(center)
    start_pos = [int(center_idx[0] + rand_x - output_size[0]/2),
                 int(center_idx[1] + rand_y - output_size[1]/2),
                 0]

    # Crop the image and segmentation to the specified output size
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(output_size)
    roi_filter.SetIndex(start_pos)

    img_roi = roi_filter.Execute(img_padded)
    seg_roi = roi_filter.Execute(seg_padded)

    # Normalize between 1st and 99th percentile.
    normalized_image = normalize(img_roi, 99, 1)

    # Convert images to numpy arrays
    arr = sitk.GetArrayFromImage(normalized_image)
    seg = sitk.GetArrayFromImage(seg_roi)

    return arr, seg


# Example usage of preprocessing function
if __name__ == '__main__':

    # Directory to image and segmentation
    dir = r'C:\William\Doktorand\Data\test_output\Test\ProstateX-0015'

    a = []
    s = []

    # Generate and display preprocessed images for visualization
    for i in range(16):
        arr, seg = startPreprocess(dir, event='Test')
        a.append(arr)
        s.append(seg)

    # # Create a 2x10 grid of subplotss for visualization
    # fig, ax = plt.subplots(2,16)

    # for j in range(16):
    #     ax[0, j].imshow(a[j][6+j, :, :], cmap='gray')
    #     ax[1, j].imshow(s[j][6+j, :, :], cmap='gray')

    # plt.show()

    # sitk.WriteImage(sitk.GetImageFromArray(arr), "C:\\William\\Doktorand\\Data\\test_output\\test0015.nrrd")