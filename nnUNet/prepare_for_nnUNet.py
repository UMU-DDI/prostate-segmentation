import SimpleITK as sitk
import os
import argparse

from utils_prepare_for_nnUNet import makeDirectory, read_image, write_image, resampleToReference

'''
This file will set up the training and test data in the required structure for the nnU-Net.
The user has the options to include the ADC and HBV sequences in addition to the required T2w axial image.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--imgs', type=str, default=[], nargs='+', help='Additional image sequences you want to include. Options are: ADC and HBV.')
#parser.add_argument('--imgs', type=str, default=['ADC', 'HBV'], help='Additional image sequences you want to include. Options are: ADC or ADC and HBV.')
parser.add_argument('--identifier', type=str, default='nrrd', help='File-type identifier. Options are: nrrd, ...')
parser.add_argument('--data_id', type=str, default="Dataset077_ProstateZones", help='Defining the ID of the dataset.')
parser.add_argument('--input_folder', type=str, default="C:\\William\\Doktorand\\Data\\test_output", help='Path to the folder containing the images.')
parser.add_argument('--output_folder', type=str, default="C:\\William\\TEST\\nnUNet_data\\nnUNet_raw", help='Path to the desired output folder.')
args = parser.parse_args()

image_sequences = args.imgs
image_sequences.insert(0, 'T2')

output_folder = os.path.join(args.output_folder, args.data_id)

############################################
# Setting up Training-data
############################################

for data in ['Train', 'Validate']:
    for patient_folder in os.listdir(os.path.join(args.input_folder, data)):
        
        numbers = ''.join([n for n in patient_folder if n.isdigit()])[-3:]

        images = 'imagesTr'
        labels = 'labelsTr'

        makeDirectory(os.path.join(output_folder, images))
        makeDirectory(os.path.join(output_folder, labels))

        for patient_file in os.listdir(os.path.join(args.input_folder, data, patient_folder)):

            path_to_file = os.path.join(args.input_folder, data, patient_folder, patient_file)

            if 'tra' in patient_file:
                t2w = read_image(path_to_file)
                continue

            elif 'adc' in patient_file:
                adc = read_image(path_to_file)
                continue

            elif 'hbv' in patient_file:       
                hbv = read_image(path_to_file)
                continue

            elif 'Seg' in patient_file:
                seg = sitk.ReadImage(path_to_file)    
                continue

        imgs = dict()

        imgs.update({'Seg': seg})
        imgs.update({'AxT2': t2w})

        dst_t2w = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0000.' + args.identifier)
        dst_label = os.path.join(output_folder, labels, 'PROSTATEx_' + numbers + '.' + args.identifier)

        write_image(imgs['AxT2'], dst_t2w)
        write_image(imgs['Seg'], dst_label)

        if 'ADC' and 'HBV' in image_sequences:
            imgs.update({'ADC': resampleToReference(adc, imgs['AxT2'], interpolator=sitk.sitkLinear)})
            dst_adc = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0001.' + args.identifier)
            imgs.update({'HBV': resampleToReference(hbv,  imgs['AxT2'], interpolator=sitk.sitkLinear)})
            dst_hbv = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0002.' + args.identifier)

            write_image(imgs['ADC'], dst_adc)
            write_image(imgs['HBV'], dst_hbv)

        elif 'ADC' in image_sequences:
            imgs.update({'ADC': resampleToReference(adc, imgs['AxT2'], interpolator=sitk.sitkLinear)})
            dst_adc = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0001.' + args.identifier)

            write_image(imgs['ADC'], dst_adc)


############################################
# Setting up json-file for nnUNet
############################################


from generate_dataset_json import generate_dataset_json

if 'ADC' and 'HBV' in image_sequences:
    ch_names = {
                0: 'T2',
                1: 'ADC',
                2: 'HBV'
                }
elif 'ADC' in image_sequences:
    ch_names = {
                0: 'T2',
                1: 'ADC'
                }
    
elif not 'ADC' in image_sequences:
    ch_names = {
                0: 'T2'
                }
        
seg_labels = {
            'background': 0,
            'PZ': 1,
            'CZ': 2,
            'TZ': 3,
            'AFS': 4,
            'Urethra': 5
        }

samples = len(os.listdir(os.path.join(output_folder, labels)))


generate_dataset_json(output_folder, ch_names, labels=seg_labels, num_training_cases=samples, file_ending=args.identifier)


############################################
# Setting up Test-data
############################################


for patient_folder in os.listdir(os.path.join(args.input_folder, 'Test')):

    numbers = ''.join([n for n in patient_folder if n.isdigit()])[-3:]

    images = 'imagesTs'

    makeDirectory(os.path.join(output_folder, images))

        
    for patient_file in os.listdir(os.path.join(args.input_folder, 'Test', patient_folder)):

        path_to_file = os.path.join(args.input_folder, 'Test', patient_folder, patient_file)

        if 'tra' in patient_file:
            t2w = read_image(path_to_file)
            continue

        elif 'adc' in patient_file:
            adc = read_image(path_to_file)
            continue

        elif 'hbv' in patient_file:
            hbv = read_image(path_to_file)
            continue


    imgs = dict()

    imgs.update({'AxT2': t2w})

    dst_t2w = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0000.' + args.identifier)
    write_image(imgs['AxT2'], dst_t2w)

    if 'ADC' and 'HBV' in image_sequences:
        imgs.update({'ADC': resampleToReference(adc, imgs['AxT2'], interpolator=sitk.sitkLinear)})
        dst_adc = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0001.' + args.identifier)
        imgs.update({'HBV': resampleToReference(hbv,  imgs['AxT2'], interpolator=sitk.sitkLinear)})
        dst_hbv = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0002.' + args.identifier)

        write_image(imgs['ADC'], dst_adc)
        write_image(imgs['HBV'], dst_hbv)

    elif 'ADC' in image_sequences:
        imgs.update({'ADC': resampleToReference(adc, imgs['AxT2'], interpolator=sitk.sitkLinear)})
        dst_adc = os.path.join(output_folder, images, 'PROSTATEx_' + numbers + '_0001.' + args.identifier)

        write_image(imgs['ADC'], dst_adc)