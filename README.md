# Hybrid Pipeline for Acute Lymphoblastic Leukemia Detection

This repository contains the code for a multi-stage hybrid AI diagnostic pipeline used to detect Acute Lymphoblastic Leukemia from blood smear images. 

## Pipeline Architecture
1. **FCM Segmentation**: Fuzzy C-Means clustering is utilized to segment the distinct cellular regions (nucleus/cytoplasm) from the background.
2. **EfficientNet Feature Extraction**: A pre-trained EfficientNetB4 model (via transfer learning) extracts a high-dimensional feature vector from the segmented images.
3. **XGBoost Classification**: An XGBoost gradient-boosting classifier serves as the final model to classify the extracted features into Benign or Malignant classes.

## Directory Structure
```
.
├── src/
│   ├── config.py           # Configuration variables (paths, img size, clusters)
│   ├── data_loader.py      # Script to load images and create train/test splits
│   ├── segmentation.py     # Fuzzy C-Means implementation and visualization
│   ├── features.py         # EfficientNetB4 feature extraction logic
│   └── evaluation.py       # Metrics, confusion matrix, ROC/PR curves
├── main.py                 # Main execution script to run the full pipeline
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Installation
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

## Usage
1. Place your dataset inside the `./dataset/` directory (organized by classes: `Benign`, `Early Pre-B`, `Pre-B`, `Pro-B`).
2. Run the main pipeline:
```bash
python main.py
```
This will run the segmentation, extract the features, train the XGBoost classifier, and generate the evaluation plots (`confusion_matrix.png`, `roc_curve.png`).
