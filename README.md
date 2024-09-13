# Automated Segmentation of Prostatic Zones and Urethra

This repository contains the code for the two models used in the paper **“Evaluation of Deep Learning Models for Automated Segmentation of Prostatic Zones and Urethra”**. The models aim to automate the segmentation of the prostate, prostatic zones, and urethra on MRI scans, reducing the labor and variability associated with manual segmentation.

## Table of Contents

- Introduction
- Data Setup
- Model Setup
    - nnU-Net
    - 3DUNet
- Usage
- Citation

## Introduction

This repository includes the code for two deep learning models, the nnU-Net [1] and an anisotropic 3D U-Net [2]. These models were evaluated for their performance in segmenting the prostate, prostatic zones, and urethra on MRI by using the public PROSTATEx [3-5] dataset together with segmentations from the ProstateZones [6] dataset.

## Data Setup

To set up the data, follow the instructions provided at: https://github.com/UMU-DDI/ProstateZones . These instructions will guide you on how to download the images and segmentations and organize them into the required folder structure for running the models (as described below).

## Model Setup

### nnU-Net

In our work, a 3D full resolution model from version 2.2 of the nnU-Net was used and evaluated with the *checkpoint_best* file.

**Dependencies**:

- Python (v3.10.0)
- PyTorch (v2.1.2)
- CUDA (v12.1.105)
- cuDNN (v8.9.7.29)

Additionally, the `scikit-image` package is necessary in addition to the installed requirements from the data structuring performed above.

**Setup**:

1. Follow the installation instructions on the nnUNet GitHub page (https://github.com/MIC-DKFZ/nnUNet) and create the necessary folders (*nnUNet_raw*, *nnUNet_preprocessed*, *nnUNet_results*). Make sure these folders are added to your paths.
2. To structure the ProstateZones data in the nnU-Net specific folders, run:
    
    ```python
    prepare_for_nnUNet.py -i "PATH_to_images" -o "PATH_to_nnUNet_raw_folder"
    
    ```
    
    The input folder should be the one containing the subfolders *Train*, *Validate*, and *Test* from the Data Setup stage.
    

**Model Training and Evaluation**:

Here, you have two options. Either, you can retrain the model from scratch (by following the steps specified under Usage Instructions on the nnU-Net page) or use our pretrained model weights. These are available at: … Link to weights …
The zip file can be extracted directly to your *nnUNet_results* folder and used for predictions, although you still need to install the nnU-Net locally. The ONNX-files can be used outside of the nnU-Net environment. Your choice.

- **Retrain**:
    - https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
- **Prediction**:
    - To run predictions using the nnU-Net on your desired data, use:
        
        ```python
        nnUNetv2_predict -i "..." -o "..." -d 77 -c 3d_fullres --save_probabilities
        ```
        
        To be able to run the postprocessing, make sure you save the probabilities from the predictions (use `--save_probabilities`). If you use anything other than the test data from ProstateZones, make sure the data is structured in the correct way, otherwise the trained model won’t work.
        

**Postprocessing**:

- Run the postprocessing from the predictions as:
    
    ```bash
    postprocess_nnUNet.py -probabilities_folder "PATH_to_predictions_with_saved_probabilities" -output_folder "PATH_to_output_folder"
    
    ```
    
    The postprocessed output will be saved as an numpy array. This can be easily converted back to an nrrd-file and visualized, for example, using Hero.


### 3D U-Net

**Dependencies**:

- Python (v.3.10.8)
- TensorFlow (v. 2.10.0)
- CUDA (11.2.152)
- cuDNN (v. 8.8.0.121)

From the folder structure created in the Data Setup stage, run the preprocessing as:

```python
preprocess_3DUNet.py -input_folder "PATH_to_input" -output_folder "PATH_to_desired_output_folder"
```

Afterwards, the training is performed with Bayesian Optimization from: https://github.com/UMU-DDI/drs-boost

## Usage

For the simplest usage of each model, the trained weights are available at: https://drive.google.com/drive/folders/1-CboNLS5H_oOwQ7KPLtlVcLTWxeCwbNR?usp=sharing

For the nnU-Net, the each fold of the model is exported as an onnx file or the full model can be accessed within the nnU-Net framework through the ProstateZones_export.zip-file.
The 3D U-Net is available as a .pb file.

### Evaluation

To evaluate the performance of the models on the predicted data, Hero workflows are available.

*OutputSegmentations.ice*: creates images in Hero for each model. What is needed is the original image, the nnU-Net prediction in a folder, and that the 3D U-Net model is loaded as a .pb file. These outputs can then be saved to a Hero database.

*Evaluation.ice*: Takes an image and two segmentations (e.g. a manual delineation and a model prediction) as inputs and saves all metrics in .csv-files in the specified folder.

## Citation

#### nnU-Net

[1] Isensee, F., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2): 203-211.

#### 3D U-Net

[2] Meyer, A., et al. (2019). Towards patient-individual PI-Rads v2 sector map: CNN for automatic segmentation of prostatic zones from T2-weighted MRI. 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), IEEE.

#### PROSTATEx dataset

[3] Geert Litjens, Oscar Debats, Jelle Barentsz, Nico Karssemeijer, and Henkjan Huisman. "ProstateX Challenge data", The Cancer Imaging Archive (2017). DOI: 10.7937/K9TCIA.2017.MURS5CL

[4] Litjens G, Debats O, Barentsz J, Karssemeijer N, Huisman H. "Computer-aided detection of prostate cancer in MRI", IEEE Transactions on Medical Imaging 2014;33:1083-1092. DOI: 10.1109/TMI.2014.2303821

[5] Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7

#### ProstateZones

[6] ...
