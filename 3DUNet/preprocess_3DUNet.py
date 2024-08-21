import os
import numpy as np
import preprocessing    # Import preprocessing module
import utils            # Import utils module

import argparse

def check_range(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 2:
        raise argparse.ArgumentTypeError(f"Number of flips must be 1 or 2, but got {ivalue}")
    return ivalue

'''
This file will perform the preprocessing for the 3D U-Net model.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default="C:\\William\\TEST", help='Path to folder containing the probabilities from the model prediction.')
parser.add_argument('--output_folder', type=str, default="C:\\William\\TEST\\3DUNet\\Preprocessed", help='Path to the desired output folder.')
parser.add_argument('--n_aug_tr', type=int, default=25, help='...')
parser.add_argument('--n_aug_val', type=int, default=5, help='...')
parser.add_argument('--n_flips_tr', type=check_range, default=2, help='...')
parser.add_argument('--n_flips_val', type=check_range, default=2, help='...')
args = parser.parse_args()

# Define directories for training and validation data.
caseDir_train = os.path.join(args.input_folder, 'Train')
saveDir_train = os.path.join(args.output_folder, 'Train')

caseDir_val = os.path.join(args.input_folder, 'Validate')
saveDir_val = os.path.join(args.output_folder, 'Validate')

caseDir_test = os.path.join(args.input_folder, 'Test')
saveDir_test = os.path.join(args.output_folder, 'Test')

if __name__ == '__main__':
    
    # Configuration for training and validation events.
    # Flips represents LEFT - RIGHT flipping. 1 - No flipping, one sample is saved. 2 - Save two samples, one with and one without flipping.
    train = {   
                'event'         : 'Train',
                'caseDir'       : caseDir_train,
                'save_path'     : saveDir_train,
                'Augmentations' : args.n_aug_tr,
                'Flips'         : args.n_flips_tr
            }
    
    val =   {
                'event'         : 'Validate',
                'caseDir'       : caseDir_val,
                'save_path'     : saveDir_val,
                'Augmentations' : args.n_aug_val,
                'Flips'         : args.n_flips_val
            }
    
    test =   {
                'event'         : 'Test',
                'caseDir'       : caseDir_test,
                'save_path'     : saveDir_test,
                'Augmentations' : 1,
                'Flips'         : 1
            }

    # Loop over training and validation configurations
    for ii in [train, val, test]: #[train, val, test]:

        event = ii['event']
        caseDir = ii['caseDir']
        save_path = ii['save_path']
        augmentations = ii['Augmentations']
        flips = ii['Flips']

        # Create output directory for saving preprocessed data if it doesn't already exist
        utils.makeDirectory(save_path)

    
        cases = os.listdir(caseDir)
        # Loop through cases in specified directory
        for case in cases:
            inputDir = os.path.join(caseDir, case)

            # Perform preprocessing for the specified number of augmentations
            for aug in range(augmentations):
                img, seg = preprocessing.startPreprocess(inputDir, event=event)
                
                for flip in range(flips):
                    # Apply flipping
                    if flip == 1:
                        img = np.flip(img, axis=2)
                        seg = np.flip(seg, axis=2)
                    
                    # Extract segmented regions
                    seg_bg = (seg == 0)     # BG
                    seg_pz = (seg == 1)     # PZ
                    seg_cz = (seg == 2)     # CZ
                    seg_tz = (seg == 3)     # TZ
                    seg_afs = (seg == 4)    # AFS
                    seg_u = (seg == 5)      # Urethra
                
                
                    # Save preprocessed data
                    # Filename will be: CASE_AUG_FLIP
                    # Where AUG and FLIP will be a number within the range specified earlier.
                    if event == 'Train' or event == 'Validate':
                        description = '_' + str(aug) + '_' + str(flip)
                    else:
                        description = ''

                    np.savez_compressed(os.path.join(save_path, str(case) + description),
                                        t2=np.array(img, dtype=np.float32),
                                        seg_bg=np.array(seg_bg, dtype=np.bool_),
                                        seg_pz=np.array(seg_pz, dtype=np.bool_),
                                        seg_cz=np.array(seg_cz, dtype=np.bool_),
                                        seg_tz=np.array(seg_tz, dtype=np.bool_),
                                        seg_afs=np.array(seg_afs, dtype=np.bool_),
                                        seg_u=np.array(seg_u, dtype=np.bool_))