import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tqdm import tqdm
from src.segmentation import segment_image_fcm

def create_feature_extractor(img_size):
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3),
        pooling='avg'
    )
    return base_model

def extract_features_and_save(image_paths, model, img_size, n_clusters, output_dir_base, set_name):
    features = []
    output_dir = os.path.join(output_dir_base, set_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for path in tqdm(image_paths, desc=f"Extracting Features & Saving Slices for {set_name}"):
        segmented_img = segment_image_fcm(path, img_size, n_clusters)
        
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
        
        img_array = tf.keras.applications.efficientnet.preprocess_input(segmented_img)
        img_array = np.expand_dims(img_array, axis=0)
        current_features = model.predict(img_array, verbose=0)
        features.append(current_features.flatten())
        
    return np.array(features)
