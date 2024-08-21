import os
import SimpleITK as sitk
import numpy as np
import argparse

from utils_postprocess_nnUNet import get_filenames_with_extension, channelSplit, getLargestCC, drawUrethra, findIndices_allSlices, column, fill_empty_voxels

'''
This file will perform the postprocessing for the nnU-Net model based on the probabilities from the model prediction.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--probabilities_folder', type=str, default="C:\\William\\TEST\\nnUNet\\nnUNet_output\\Probabilities", help='Path to folder containing the probabilities from the model prediction.')
parser.add_argument('--file_identifier', type=str, default='nrrd', help='The file identifier. Options are: ...')
parser.add_argument('--simple_postprocess', type=bool, default=False, help='Type of postprocess. Options are: True or False.')
parser.add_argument('--radius', type=int, default=3, help='Defining the radius of the drawn urethra.')
parser.add_argument('--output_folder', type=str, default="C:\\William\\TEST\\nnUNet\\nnUNet_output\\Postprocessed", help='Path to the desired output folder.')
args = parser.parse_args()


filenames = get_filenames_with_extension(args.probabilities_folder, args.file_identifier)

for file in filenames:

    img = sitk.ReadImage(os.path.join(args.probabilities_folder, file))
    spacing = img.GetSpacing()

    probabilities = np.load(os.path.join(args.probabilities_folder, file.replace(args.file_identifier, 'npz')))['probabilities'].transpose(3,2,1,0)

    bg = channelSplit(probabilities, 0)
    pz = channelSplit(probabilities, 1)
    cz = channelSplit(probabilities, 2)
    tz = channelSplit(probabilities, 3)
    afs = channelSplit(probabilities, 4)
    u = channelSplit(probabilities, 5)


    pred_background = np.argmax(probabilities, axis=3) == 0

    lcc_background = getLargestCC(pred_background)

    prostate = getLargestCC((~lcc_background).astype(int))

    background = ~prostate

    urethra_zSlices = np.split(u, u.shape[2], axis=2)

    indices_list = []
    placeholder = [indices_list.append(np.insert(findIndices_allSlices(zSlice), 2, z)) for zSlice, z in zip(urethra_zSlices, range(len(urethra_zSlices)))]


    x = column(np.array(indices_list), 0)
    y = column(np.array(indices_list), 1)
    z = column(np.array(indices_list), 2)
    w = column(np.array(indices_list), 3)

    fitxz = np.polyfit(z, x, deg=2, w=w)
    fityz = np.polyfit(z, y, deg=2, w=w)

    x_pred = np.polyval(fitxz, z)
    y_pred = np.polyval(fityz, z)

    x_p = np.round(x_pred)
    y_p = np.round(y_pred)


    # Draw urethra.
    urethra_list = []
    for zSlice, ii in zip(urethra_zSlices, range(len(indices_list))):
        indices = np.array([x_p[ii], y_p[ii], z[ii]])
        urethra_list.append(drawUrethra(array_shape=zSlice.shape, indicies=indices, radius_mm=args.radius, inplane_image_spacing=spacing[0]))


    urethra_all_slices = np.stack(urethra_list, axis=2).squeeze()

    urethra = urethra_all_slices * prostate.astype(int)

    if args.simple_postprocess:

        updated_probabilities_stacked = np.stack([background.astype(int), pz, cz, tz, afs, urethra], axis=-1)
        segmentation = updated_probabilities_stacked.argmax(axis=3)

    else:
        
        pred_pz = np.argmax(probabilities, axis=3) == 1
        pred_cz = np.argmax(probabilities, axis=3) == 2
        pred_tz = np.argmax(probabilities, axis=3) == 3
        pred_afs = np.argmax(probabilities, axis=3) == 4

        pz_pred_within_prostate = pred_pz * prostate.astype(int)
        cz_pred_within_prostate = pred_cz * prostate.astype(int)
        tz_pred_within_prostate = pred_tz * prostate.astype(int)
        afs_pred_within_prostate = pred_afs * prostate.astype(int)

        prel_segmentation = fill_empty_voxels(background.astype(int), pz_pred_within_prostate, cz_pred_within_prostate, tz_pred_within_prostate, afs_pred_within_prostate, urethra.astype(int), spacing)
        segmentation = prel_segmentation.argmax(axis=3)

    np.save(os.path.join(args.output_folder, file[:-5]), segmentation)