# Data Augmentation Pipeline for Hand Gesture Recognition Models

## Overview
This repository implements a robust data augmentation pipeline designed to enhance the performance, robustness, and generalization of three hand gesture recognition (HGR) models. The augmentation techniques are applied to improve the models' ability to generalize across diverse datasets and conditions. The repository is organized into separate folders for each model and includes Kaggle notebooks for training and evaluation.

## Supported Models

1. **[e2eET Skeleton-Based HGR Using Data-Level Fusion](https://github.com/Outsiders17711/e2eET-Skeleton-Based-HGR-Using-Data-Level-Fusion?tab=readme-ov-file)**
   - Implements skeleton-based hand gesture recognition with data-level fusion.
   - Augmentation is applied through a script, and augmented data is uploaded to Kaggle as a dataset.

2. **[FPPR-PCD](https://github.com/multimodallearning/hand-gesture-posture-position/tree/master)**
   - Focuses on hand gesture posture and position recognition.
   - Augmentation is applied through a script, and augmented data is uploaded to Kaggle as a dataset.

3. **[DD-Net](https://github.com/fandulu/DD-Net)**
   - A model designed for dynamic hand gesture recognition.
   - Augmentation is applied directly within the Kaggle notebook.

---

## Data Augmentation Pipeline

The data augmentation pipeline implements the following parameters:

- **Number of Augmentations:** `num_augmentations = 3`
- **Crop Size Ratio:** `crop_size_ratio = 0.95`
- **Brightness Range:** `brightness_range = (0.8, 1.2)`
- **Contrast Range:** `contrast_range = (0.8, 1.2)`
- **Rotation Range:** `rotation_range = (-15, 15)` (degrees)
- **Zoom Range:** `zoom_range = (1.0, 1.2)`

These transformations are applied to increase dataset diversity, making models more robust to variations in input data.

---

## Repository Structure

- **`e2eET/`**
  - Contains scripts to augment the dataset for the e2eET Skeleton-Based HGR model.
  - Augmented data is uploaded to Kaggle for further training and evaluation.

- **`FPPR-PCD/`**
  - Contains scripts to augment the dataset for the FPPR-PCD model.
  - Augmented data is uploaded to Kaggle for further training and evaluation.

- **`DD-Net/`**
  - Augmentation is implemented directly in the Kaggle notebook within this folder.

- **`kaggle_notebook/`**
  - Contains Jupyter notebooks used for model training, evaluation, and results analysis on Kaggle.

---

## Usage Instructions

### 1. Augmenting Data
- For **e2eET**:
  1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) (located in the `image_d` folder).
  2. Follow the model's data processing instructions.
  3. Run the augmentation script: `python data_augmentation.py`.
  4. Upload the augmented dataset to Kaggle.

- For **FPPR-PCD**:
  1. Download the datasets from [SHREC 2017](http://www-rech.telecom-lille.fr/shrec2017-hand/) and [DHG](http://www-rech.telecom-lille.fr/DHGdataset/).
  2. Follow the model's data processing instructions.
  3. Run the augmentation script: `python data_augment.py`.
  4. Upload the augmented dataset to Kaggle.

- For **DD-Net**:
  1. Use the Kaggle notebook in `kaggle_notebook/` to apply the augmentation pipeline directly.

### 2. Training and Evaluation
- Use the Kaggle notebooks in `kaggle_notebook/` to train and evaluate the models.
- Notebooks include visualizations and performance metrics to analyze the results.

---

## Results

The results of the augmented models are documented in the Kaggle notebooks. Metrics such as accuracy, precision, recall, and robustness are provided to showcase the impact of data augmentation on model performance.

### Results Comparison for e2eET and Proposed Framework

| Dataset       | e2eET (%) | Framework (%) |
|---------------|-----------|---------------|
| SHREC'17 14g | 96.6        | 98.21         |
| SHREC'17 28g | 94.05      | 94.52         |
| DHG 28g      | 91.67     | 92.98         |

### Results Comparison for DD-Net and Proposed Framework

| Dataset   | DD-Net (%) | Framework (%) |
|-----------|------------|---------------|
| JHMDB     | 81.82      | 86.36         |
| SHREC'14  | 94.88      | 95.95         |

### Results Comparison for FPPR-PCD and Proposed Framework

| Dataset   | FPPR-PCD (%) | Framework (%) |
|-----------|------------|---------------|
| SHREC'14  | 995.9     | 96.4        |


---

## Acknowledgements

- **[e2eET Skeleton-Based HGR Using Data-Level Fusion](https://github.com/Outsiders17711/e2eET-Skeleton-Based-HGR-Using-Data-Level-Fusion?tab=readme-ov-file)**
- **[FPPR-PCD](https://github.com/multimodallearning/hand-gesture-posture-position/tree/master)**
- **[DD-Net](https://github.com/fandulu/DD-Net)**

