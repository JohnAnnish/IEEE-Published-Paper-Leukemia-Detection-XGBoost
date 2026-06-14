import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def segment_image_fcm(image_path, img_size, n_clusters):
    """Performs Fuzzy C-Means segmentation on a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    
    pixel_values = img.reshape((-1, 3)).astype(np.float32)
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        pixel_values.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
    )
    
    cluster_membership = np.argmax(u, axis=0)
    
    counts = np.bincount(cluster_membership)
    target_cluster = np.argmin(counts[np.nonzero(counts)]) if len(counts) > 1 else 0
    
    segmented_pixels = np.zeros_like(pixel_values, dtype=np.uint8)
    segmented_pixels[cluster_membership == target_cluster] = pixel_values[cluster_membership == target_cluster].astype(np.uint8)
    
    return segmented_pixels.reshape(img.shape)

def visualize_for_paper(image_path, img_size=224, n_clusters=3):
    """Generates the three panels (Original, Segmented, Overlay) for a single sample."""
    original_img = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (img_size, img_size))
    predicted_segmented_img = segment_image_fcm(image_path, img_size, n_clusters)
    
    gray_mask = cv2.cvtColor(predicted_segmented_img, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    overlay = original_img.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    
    return original_img, predicted_segmented_img, overlay
