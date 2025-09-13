# CT-Scan-Reconstruction-Using-L1-and-L2-Regularization

This project demonstrates the use of **compressed sensing** and **regularization** techniques for CT scan reconstruction from sparse projections. It applies L1 (**LASSO**) and L2 (**Ridge**) regression methods to reconstruct CT images and compares their performance.

-----

## Features

  * Reads CT scan images in **DICOM** format.
  * Preprocesses and masks the CT image with a circular mask to remove the background.
  * Generates **sparse sinogram projections** using the **Radon transform**.
  * Builds a **system matrix** to model the imaging process.
  * Reconstructs the image using **LASSO (L1 regularization)** and **Ridge (L2 regularization)**.
  * Visualizes the original image, sinogram, and both reconstructions.

-----

## Dataset

The dataset used is available on Kaggle: [SIIM Medical Images Dataset](https://www.google.com/search?q=https://www.kaggle.com/c/siim-medical-images/data)

Place the dataset in the `archive/dicom_dir/` directory. The script uses an example file:
`archive/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm`

-----

## Installation

Install the required Python packages:

```bash
pip install scikit-image numpy matplotlib scikit-learn pydicom
```

-----

## Usage

Run the reconstruction script with:

```bash
python ct_reconstruction.py
```

-----

## Workflow

### Load DICOM Image

  * Read a CT scan `.dcm` file with **pydicom**.
  * Normalize pixel values to the range [0,1].
  * Rescale the image for computational efficiency.

### Masking

Apply a **circular mask** to isolate the body region from the background.

### Sparse Projections

Generate sparse sinogram projections using the **Radon transform** with a limited number of angles.

### System Matrix Construction

Build the system matrix that maps pixel values to projection data. Normalize the system matrix for stable computation.

### Reconstruction

  * Apply **LASSO regression (L1 regularization)**, which promotes **sparsity** in the solution.
  * Apply **Ridge regression (L2 regularization)**, which produces **smoother** reconstructions.

### Visualization

Display the original CT image, sparse sinogram, LASSO reconstruction, and Ridge reconstruction.

-----

## Results

  * **LASSO reconstruction** tends to capture **sharper details** but may introduce noise.
  * **Ridge reconstruction** produces **smoother images** but may blur finer structures.

This illustrates the trade-offs between L1 and L2 regularization methods in medical image reconstruction.
