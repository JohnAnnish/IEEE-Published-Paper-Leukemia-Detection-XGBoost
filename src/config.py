import os

IMG_SIZE = 224
N_CLUSTERS = 3
DATA_PATH = './dataset'
SEG_OUTPUT_DIR = './segmented_images/'

# Create directories
os.makedirs(os.path.join(SEG_OUTPUT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(SEG_OUTPUT_DIR, 'test'), exist_ok=True)
