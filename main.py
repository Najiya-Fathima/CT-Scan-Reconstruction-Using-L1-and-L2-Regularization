
# pip install scikit-image numpy matplotlib scikit-learn pydicom

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, rescale
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import normalize
import pydicom
from skimage.draw import disk

# === Step 0: Load CT scan image from the Kaggle dataset ===

# https://www.kaggle.com/datasets/kmader/siim-medical-images

file_path = "archive/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm"

dcm = pydicom.dcmread(file_path)
image = dcm.pixel_array.astype(np.float32)
    
# Normalize to [0, 1]
image -= image.min()
image /= image.max()

# Resize image
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

# === Mask the image (remove background outside the circular region) ===
def mask_circle(img):
    masked = np.zeros_like(img)
    rr, cc = disk((img.shape[0] // 2, img.shape[1] // 2), radius=img.shape[0] // 2)
    masked[rr, cc] = img[rr, cc]
    return masked

image = mask_circle(image)

# === Step 1: Generate full and sparse sinograms ===
n_projections = 60
theta_sparse = np.linspace(0., 180., n_projections, endpoint=False)
sinogram_sparse = radon(image, theta=theta_sparse, circle=False)

# === Step 2: Build system matrix A ===
def build_system_matrix(img_shape, angles):
    output_size = max(img_shape)
    n_pixels = img_shape[0] * img_shape[1]
    A = np.zeros((output_size * len(angles), n_pixels), dtype=np.float32)
    col = 0
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            delta = np.zeros(img_shape, dtype=np.float32)
            delta[i, j] = 1.0
            proj = radon(delta, theta=angles, circle=False)
            proj = proj[:output_size, :].ravel()
            A[:, col] = proj
            col += 1
    return A

print("Building system matrix A (this may take a while)...")
A = build_system_matrix(image.shape, theta_sparse)

# Normalize A
A = normalize(A, axis=0)

# Flatten measurements y
y = sinogram_sparse[:max(image.shape), :].ravel()

# === Step 3: L1 Reconstruction (LASSO) ===
print("Solving LASSO...")
lasso = Lasso(alpha=0.0001, max_iter=10000)
lasso.fit(A, y)
recon_l1 = lasso.coef_.reshape(image.shape)

# === Step 4: L2 Reconstruction (Ridge) ===
print("Solving Ridge...")
ridge = Ridge(alpha=0.01)
ridge.fit(A, y)
recon_l2 = ridge.coef_.reshape(image.shape)

# === Step 5: Visualization ===
titles = ["Original CT",
          f"Sinogram (n={n_projections})",
          "Reconstruction (L1 / LASSO)",
          "Reconstruction (L2 / Ridge)"]
images = [image,
          sinogram_sparse,
          recon_l1,
          recon_l2]

# Each image separately
for img, title in zip(images, titles):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray', aspect='auto' if img.ndim == 2 and img.shape[0] != img.shape[1] else 1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# All images in one row
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray', aspect='auto' if img.ndim == 2 and img.shape[0] != img.shape[1] else 1)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
