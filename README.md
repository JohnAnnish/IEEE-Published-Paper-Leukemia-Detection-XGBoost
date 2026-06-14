# High-Accuracy Acute Lymphoblastic Leukemia Detection via Transfer Learning with EfficientNetB4

## Abstract
Acute Lymphoblastic Leukemia (ALL) is a severe form of blood cancer that progresses rapidly and requires early and accurate detection for effective treatment. In this research, we propose a robust deep learning pipeline for the multi-class categorization of leukemia cells from blood smear images. Utilizing **EfficientNetB4** via transfer learning, our model acts as a powerful feature extractor and classifier. We employ significant data augmentation techniques and class-weighting to combat dataset imbalance. The proposed model achieves an outstanding **98.61% validation accuracy**, demonstrating its viability as an assisting tool for clinical hematological diagnostics.

---

## 1. Introduction
The manual inspection of peripheral blood smears under a microscope is a time-consuming and error-prone process, heavily reliant on the expertise of the hematologist. Automating the detection and classification of ALL sub-types can significantly expedite diagnosis. We present an end-to-end framework leveraging **EfficientNetB4**, an architecture renowned for its optimal scaling of network width, depth, and resolution.

---

## 2. Dataset Description
The dataset comprises 3,242 blood smear images categorized into four distinct classes representing different stages and types of cells:
- `Benign`: Non-cancerous cells.
- `Early Pre-B`: Early-stage pre-B acute lymphoblastic leukemia cells.
- `Pre-B`: Pre-B acute lymphoblastic leukemia cells.
- `Pro-B`: Pro-B acute lymphoblastic leukemia cells.

The dataset was split into training (2,595 images) and validation (647 images) sets using an 80/20 ratio. To mitigate the natural class imbalance, computed **class weights** were applied during the model's training phase (e.g., `Benign` was weighted at ~1.58, and `Early Pre-B` at ~0.82).

---

## 3. Methodology

### 3.1. Data Pre-processing & Augmentation
All images were resized to **380x380 pixels** to match the native input resolution of EfficientNetB4. To enhance the model's generalization capabilities and prevent overfitting, the following online data augmentations were applied using Keras's `ImageDataGenerator`:
- Random Rotations (up to 20 degrees)
- Width and Height Shifts (±10%)
- Shear and Zoom transformations (10%)
- Horizontal and Vertical flips
- EfficientNet specific Input Pre-processing

### 3.2. Model Architecture
The core architecture is based on **EfficientNetB4** pre-trained on the ImageNet dataset.
1. **Base Extraction**: The top classification layers of EfficientNetB4 were removed (`include_top=False`).
2. **Global Average Pooling**: Applied to reduce the spatial dimensions of the extracted feature maps to a 1D vector.
3. **Dropout Regularization**: A dropout layer with a rate of `0.5` was added to limit overfitting.
4. **Classification Head**: A fully connected `Dense` layer with 4 units and a `softmax` activation function outputs the final class probabilities.

### 3.3. Two-Phase Training Strategy
We utilized a rigorous two-phase transfer learning approach:

- **Phase 1 (Feature Head Training):** The entire EfficientNetB4 base model was frozen. Only the new Dense classification head was trained for 10 epochs using the Adam optimizer with a learning rate of `1e-3`. This phase prevents large gradient updates from destroying the pre-trained ImageNet weights.
- **Phase 2 (Fine-Tuning):** The top 50 layers of the EfficientNetB4 base model were unfrozen to allow the network to learn domain-specific hematological features. The network was trained for an additional 30 epochs with a lower learning rate of `1e-4`. 

During both phases, dynamic callbacks were used:
- **ReduceLROnPlateau**: Halves the learning rate if the validation loss plateaus.
- **EarlyStopping**: Halts training if no improvement is seen in validation loss.
- **ModelCheckpoint**: Saves the best-performing model weights.

---

## 4. Results & Performance

The model showcased exceptional learning dynamics throughout the training process. 

- **Phase 1 Validation Accuracy:** Reached **92.43%**
- **Phase 2 Validation Accuracy:** Following the unfreezing of the top 50 layers, the model rapidly converged, achieving a peak validation accuracy of **98.61%** with a validation loss of just `0.0654`.

These results indicate that the fine-tuned EfficientNetB4 architecture is highly capable of distinguishing the subtle morphological differences between Benign cells and the various sub-types of Acute Lymphoblastic Leukemia.

---

## 5. Conclusion
This repository demonstrates a highly effective, automated pipeline for ALL classification. By successfully applying a two-phase fine-tuning strategy to EfficientNetB4 alongside aggressive data augmentation and class balancing, we achieved state-of-the-art results (98.61%). This work validates the usage of scaled convolutional architectures in critical computer-aided medical diagnostics.

---

## 6. How to Run and View the Research
The entire methodology, code, and execution history are encapsulated in a single Jupyter Notebook.

1. Open `efficientnetb4.ipynb` directly in the GitHub repository viewer to see the documented code and the massive execution outputs of the model training.
2. To run locally, ensure you have TensorFlow and Jupyter installed, download the corresponding dataset, and execute the notebook cell-by-cell.
