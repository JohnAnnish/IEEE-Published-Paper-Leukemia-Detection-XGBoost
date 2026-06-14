import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    filepaths = []
    labels = []
    malignant_folders = {'Early Pre-B', 'Pre-B', 'Pro-B'}
    
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        label = -1
        if dir_name == 'Benign':
            label = 0
        elif dir_name in malignant_folders:
            label = 1
        
        if label != -1:
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    filepaths.append(os.path.join(dir_path, filename))
                    labels.append(label)
                    
    return filepaths, labels

def get_train_test_splits(data_path, test_size=0.2, random_state=42):
    filepaths, labels = load_data(data_path)
    X = np.array(filepaths)
    y = np.array(labels)

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train_paths, X_test_paths, y_train, y_test
