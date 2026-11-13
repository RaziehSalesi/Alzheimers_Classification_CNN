# ADNI-Based Alzheimer's Disease Classification with CNNs

This repository contains the implementation for **Question 1 of Homework
2** in the Neural Networks and Deep Learning course at the University of
Tehran.

The project investigates binary classification of brain MRI images from
the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. Three
convolutional neural network (CNN) architectures are implemented and
evaluated to distinguish between:

-   **AD** --- Alzheimer's disease
-   **MCI** --- Mild cognitive impairment

> **Scope:** This repository currently covers Question 1 (ADNI
> classification). It does not contain the VGG16/ResNet50 experiments
> from Question 2.

## Project overview

The notebook covers the following workflow:

1.  Load the ADNI image dataset and its class labels.
2.  Resize images to `64 × 64`.
3.  Apply offline data augmentation to the training images.
4.  Split the original images into training, validation, and test
    subsets using stratified sampling.
5.  Implement and train three CNN architectures.
6.  Evaluate the trained models using accuracy, loss, ROC curves,
    confusion matrices, precision, recall, F1/DCS, and AUC.

## Dataset

The notebook expects the dataset to be organized into two class folders:

``` text
Alzheimer_s_Disease_Neuroimaging_ADNI_Dataset/
├── AD/
└── MCI/
```

The dataset used in the saved notebook run contains **1,654 images**:

  Class         Number of original images
  ----------- ---------------------------
  AD                                  965
  MCI                                 689
  **Total**                     **1,654**

The dataset is not included in this repository. Users must obtain the
data from an authorized source and configure the notebook's dataset
path.

## Preprocessing and data augmentation

Images are resized from `256 × 256` to `64 × 64` pixels.

The notebook applies offline augmentation to the training subset. Four
augmented images are generated for each original training image, giving
five training images per original image (the original plus four
augmented versions).

The implemented augmentation operations are:

-   Random horizontal flipping
-   Shearing
-   Translation (shifting)
-   Random rotation (up to 45 degrees)
-   Random resized cropping (zooming)

**Important implementation note:** The current preprocessing code
divides pixel values by 255 and then converts the images back to PIL
images. This conversion restores integer pixel values, so the arrays
subsequently passed to the model are not guaranteed to remain in the
`[0, 1]` range. The current notebook should therefore not be described
as performing effective `[0, 1]` normalization.

### Data splitting

The split is performed on the original image paths before augmentation.
Only the training subset is augmented; validation and test images are
not augmented.

The saved run uses:

-   Test set: 5% of the original dataset
-   Validation set: 10% of the remaining 95%
-   Training set: the remaining original images

  Subset         Original images   Images after augmentation
  ------------ ----------------- ---------------------------
  Training                 1,413                       7,065
  Validation                 158                         158
  Test                        83                          83

Stratified splitting is used to preserve class proportions.

## CNN architectures

Three CNN architectures are implemented:

### 1. Proposed Model

-   Four convolutional layers
-   ReLU activation functions
-   Two max-pooling layers
-   Fully connected layers with 128 and 64 units
-   Two output units for binary classification

### 2. Testing Model 1

-   Two convolutional layers
-   Batch normalization and ReLU activations
-   Two max-pooling operations
-   A fully connected layer with 128 units
-   Two output units

### 3. Testing Model 2

-   Two convolutional layers
-   Batch normalization and ReLU activations
-   One max-pooling operation
-   Fully connected layers with 128 and 64 units
-   Two output units

Glorot/Xavier uniform initialization is applied to convolutional and
linear layer weights. Biases are initialized to zero.

## Training configuration

The saved run uses the following configuration:

  Parameter               Value
  ----------------------- ------------------------------------
  Optimizer               Adam
  Loss function           Cross-Entropy Loss
  Learning rate           0.0001
  Epochs                  30
  Batch size              128
  Input image size        64 × 64
  Weight initialization   Xavier/Glorot uniform
  Device                  CUDA when available; otherwise CPU

The notebook uses PyTorch `Dataset` and `DataLoader` classes to prepare
and load batches.

## Evaluation metrics

The models are evaluated on the held-out test subset using:

-   Accuracy
-   Precision
-   Recall
-   DCS (Dice Similarity Coefficient; in this binary classification
    implementation it is calculated as the harmonic mean of precision
    and recall, equivalent to the positive-class F1 score)
-   ROC curve and AUC
-   Confusion matrix
-   Classification report
-   Test loss

The positive class for the manually calculated precision, recall, and
DCS metrics is class `1` (**MCI**), based on the notebook's mapping
`AD = 0` and `MCI = 1`.

## Results

The following values are taken from the saved outputs of the current
notebook run. The test set contains 83 images (48 AD and 35 MCI).

  Metric              Proposed Model   Testing Model 1   Testing Model 2
  ----------------- ---------------- ----------------- -----------------
  Test accuracy           **97.59%**            92.77%            96.39%
  Precision (MCI)             97.14%            87.18%            94.44%
  Recall (MCI)                97.14%            97.14%            97.14%
  DCS / F1 (MCI)              97.14%            91.89%            95.77%
  AUC                     **99.58%**            97.92%            99.17%
  Test loss                   0.0777            0.2186            0.1313

For this saved run, the Proposed Model achieved the highest test
accuracy and AUC among the three architectures.

These results are specific to this run and split. They should not be
interpreted as a clinical performance claim or as evidence of
generalization to independent patients or institutions.

## Accuracy function and its limitation

The notebook retains the following batch-level accuracy function:

``` python
def accuracy_fn(pred, true):
    pred = torch.argmax(pred, dim=1)
    acc = (true == pred).float().mean().item()
    return acc
```

This function correctly calculates the proportion of correctly
classified samples **within one batch**.

However, the training and validation loops accumulate these batch
accuracies and divide by the number of batches. This gives every batch
equal weight, even when the final batch contains fewer samples than the
others. As a result, the reported epoch-level training and validation
accuracy can differ slightly from the sample-weighted accuracy over the
complete subset.

The function is retained to document the implementation used in this
coursework. The test evaluation function uses a sample-weighted
calculation.

## Running the notebook

The notebook was developed for Google Colab and currently mounts Google
Drive.

1.  Open the notebook in Google Colab.

2.  Upload or make the ADNI dataset available in your Google Drive.

3.  Update the dataset path in the data-loading cell:

    ``` python
    path = '/content/drive/MyDrive/Data/Alzheimer_s_Disease_Neuroimaging_ADNI_Dataset'
    ```

4.  Run the notebook cells from top to bottom.

5.  Save the notebook after execution so that the outputs and plots are
    stored in the `.ipynb` file.

The notebook uses Python and the following main packages:

-   PyTorch
-   Torchvision
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   OpenCV
-   Pillow
-   scikit-learn
-   tqdm

## Current limitations and possible improvements

The current notebook is a coursework implementation. The following
points should be considered when interpreting or extending its results:

1.  **Pixel scaling:** The current PIL conversion means that the
    preprocessing pipeline does not reliably preserve `[0, 1]`
    normalized pixel values.
2.  **Epoch accuracy aggregation:** The retained `accuracy_fn` is
    batch-correct, but averaging batch accuracies equally does not
    account for a smaller final batch.
3.  **Validation labels:** The validation loop applies `LabelEncoder`
    separately to each batch even though the dataset already returns
    integer class labels. Using the existing labels directly would avoid
    unnecessary batch-wise re-encoding.
4.  **Patient-level separation:** The current split is image-level. If
    multiple images belong to the same patient, a patient-level split
    should be considered to reduce the risk of data leakage.
5.  **Additional homework experiments:** The current notebook does not
    include the requested comparisons for alternative split ratios (0.3
    and 0.5), dropout, or Glorot versus zero initialization. These
    experiments should not be considered completed based on this
    notebook alone.
6.  **Reproducibility:** The augmentation operations use random
    transformations. For reproducible experiments, random seeds should
    be set for Python, NumPy, and PyTorch, and the data split and
    environment should be documented.

## Repository contents

``` text
.
├── README.md
└── HW2_Q1_Razieh_100__best_0_0001_128_30.ipynb
```

The ADNI dataset is not included.

## Disclaimer

This project was developed for educational purposes as part of a Neural
Networks and Deep Learning course. It is not intended for clinical
diagnosis or medical decision-making.
