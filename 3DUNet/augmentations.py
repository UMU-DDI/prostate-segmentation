import os
import numpy as np
import SimpleITK as sitk
import math

from utils import normalize, padd, matrix_from_axis_angle



################ APPLY ####################


def applyTransform(image, segmentation, transform):

    for name in image.keys():
        image.update({name : sitk.Resample(image[name], transform, sitk.sitkLinear, 0)})

    segmentation = sitk.Resample(segmentation, transform, sitk.sitkNearestNeighbor, 0)

    return image, segmentation
    


# Augments the image & segmentation through rotating and scaling as well as normalizes the image and adds random noise.
def applyAugmentations(image, segmentation, output_size, translation, noise):
    
    
    # Translation - moves bounding box within image boundaries
    image, seg = padd(image, segmentation, output_size[2], translation[2])
    
    
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(output_size)
    roi_filter.SetIndex(translation)
    
    for name in image.keys():
        image.update({name : roi_filter.Execute(image[name])})
    segmentation = roi_filter.Execute(seg)

    #
    noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
    noise_filter.SetMean(0)
    noise_filter.SetStandardDeviation(noise)
    for name in image.keys():
        image.update({name : noise_filter.Execute(image[name])})

    # Normalize between 1st and 99th percentile.
    for name in image.keys():
        image.update({name : normalize(image[name], 99, 1)})

    
    return image, segmentation


################ GET ######################


def getTransform(imgs, center, rot, scale):

    transform = sitk.Similarity3DTransform()

    transform.SetCenter(center)

    direction = np.array(imgs['AxT2'].GetDirection()).reshape(3,3)
    axis_angle = (direction[0, 2], direction[1, 2], direction[2, 2], np.deg2rad(rot))
    np_rot_mat = matrix_from_axis_angle(axis_angle)

    transform.SetMatrix(np_rot_mat.flatten().tolist())
    transform.SetScale(scale)

    return transform

def getTranslation(output_size, bounding_box, event):

    # According to PI-QUALS the prostate image FOV (in-plane) should be 12-20cm. Our bounding box is 9.6cm.
    # This means that we have 2.4 - 10.4 cm to move around in. Reasonable to restrict to a maximum movement of 1/2 of what's avaliable? => 1/4 movement in each direction (up/down or left/right).

    delta_max = np.array(output_size) - np.array(bounding_box[3:])

    if event == 'Train':
        if delta_max[0] > 0:
            rand_x = np.random.randint(0.75 * math.floor(delta_max[0]/2), 1/0.75 * math.floor(delta_max[0]/2))
        else:
            rand_x = math.floor(delta_max[0]/2)
            
        if delta_max[1] > 0:
            rand_y = np.random.randint(0.75 * math.floor(delta_max[1]/2), 1/0.75 * math.floor(delta_max[1]/2))
        else:
            rand_y = math.floor(delta_max[1]/2)

        if delta_max[2] > 0:
            rand_z = np.random.randint(0.8 * math.floor(delta_max[2]/2), 1/0.8 * math.floor(delta_max[2]/2))
        else:
            rand_z = 0

    elif event == 'Validate':
        if delta_max[0] > 0:
            rand_x = np.random.randint(0.9 * math.floor(delta_max[0]/2), 1/0.9 * math.floor(delta_max[0]/2))
        else:
            rand_x = math.floor(delta_max[0]/2)

        if delta_max[1] > 0:
            rand_y = np.random.randint(0.9 * math.floor(delta_max[1]/2), 1/0.9 * math.floor(delta_max[1]/2))
        else:
            rand_y = math.floor(delta_max[1]/2)

        if delta_max[2] > 0:
            rand_z = math.floor(delta_max[2]/2)
        else:
            rand_z = math.floor(delta_max[2]/2)

    
    box_pos = [int(np.max([0, bounding_box[0] - rand_x])),
               int(np.max([0, bounding_box[1] - rand_y])),
               int(np.max([0, bounding_box[2] - rand_z]))]

    return box_pos


def getRotation(event):

    if event == 'Train':
        rot = np.random.uniform(-30, 30)
    elif event == 'Validate':
        rot = np.random.uniform(-10, 10)
    else:
        rot = 0

    #print('Rotation at: ' + str(rot) + ' degrees')

    return rot

def getScale(event):

    if event == 'Train':
        scale = np.random.uniform(0.925, 1/0.925)
    elif event == 'Validate':
        scale = np.random.uniform(0.975, 1/0.975)
    else:
        scale = 1

    #print('Scaling factor: ' + str(scale))

    return scale

def getBoundingBox(proseg):

    mask = sitk.BinaryThreshold(proseg, lowerThreshold=1, insideValue=1, outsideValue=0)
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(mask)
    BB = filter.GetBoundingBox(1)

    #print('Bounding box start coordinates: ' + str(BB[:3]))
    #print('Bounding box size: ' + str(BB[3:]))

    return BB

def getCenter(proseg):

    mask = sitk.BinaryThreshold(proseg, lowerThreshold=1, insideValue=1, outsideValue=0)
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(mask)
    center = filter.GetCentroid(1)

    #print('Center point at index: ' + str(mask.TransformPhysicalPointToIndex(center)))

    return center

def getNoise(event):

    # Noise - Random Gaussian
    if event == 'Train':
        noise = np.random.uniform(0, 0.1)
    elif event == 'Validate':
        noise = 0
    else: 
        noise = 0

    #print('Noise level at: ' + str(noise))

    return noise


def getRotationMatrix(rot, imgs):

    direction = np.array(imgs['AxT2'].GetDirection()).reshape(3,3)
    axis_angle = (direction[0, 2], direction[1, 2], direction[2, 2], np.deg2rad(rot))
    rotationMatrix = matrix_from_axis_angle(axis_angle)

    return rotationMatrix


def augmentData(img, seg, event):

    output_size = [192, 192, 32]

    prel_img = img + 1
    center = getCenter(prel_img)
    BB = getBoundingBox(prel_img)

    rot = getRotation(event)
    scale = getScale(event)

    transform = getTransform(img, center, rot, scale)
    imgs_transformed, seg_transformed = applyTransform(img, seg, transform)

    ## Think about range! What is a usual FOV for prostate examinations?
    noise = getNoise(event)
    translation = getTranslation(output_size, BB, event)

    imgs_augmented, seg_augmented = applyAugmentations(imgs_transformed, seg_transformed, output_size, translation, noise)

    return imgs_augmented, seg_augmented