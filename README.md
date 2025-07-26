# Age & Gender Prediction using VGG16

This project uses a pre-trained VGG16 model to perform multi-output prediction on facial images from the UTKFace dataset, simultaneously estimating both age (regression) and gender (classification). The workflow includes thorough feature engineering such as image resizing, normalization, and extensive augmentation to improve model robustness and generalization. The VGG16 base is frozen, and custom dense layers are added for both tasks, enabling efficient transfer learning.

---

## 🧠 Description

The model performs:
- **Age Prediction** as a regression task.
- **Gender Prediction** as a binary classification task.

---

## ⚙️ Feature Engineering

Before training, each image undergoes the following preprocessing and augmentation:

- Resizing to **200x200**
- **Normalization** (pixel values scaled to 0–1)
- **Augmentations**:
  - Horizontal flip
  - Brightness adjustment
  - Contrast, hue, and saturation variation
  - Small random rotation (±15°)

These techniques increase dataset variability and help the model generalize better.

---

## 🛠️ Fine-Tuning

- A **pre-trained VGG16** is used as the base (with `include_top=False`).
- The base is **frozen** (non-trainable) during training.
- Two separate branches (heads) are added:
  - One for **age regression**
  - One for **gender classification**
- The model is trained jointly on both tasks with weighted losses.

---

## 🧱 Model Architecture

```bash
               +-----------------------+
               |     VGG16 Base        |
               +-----------------------+
                        |
                      Flatten
                        |
    +-------------------+--------------------+
    |                                        |
Dense Layer                              Dense Layer
    |                                        |
Dense Layer                              Dense Layer
    |                                        |
Output: AGE                            Output: Gender
