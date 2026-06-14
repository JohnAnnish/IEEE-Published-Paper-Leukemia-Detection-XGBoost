import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

def calculate_all_metrics(y_true, y_pred_probs, target_names=['Benign', 'Malignant']):
    y_pred = (y_pred_probs > 0.5).astype(int)
    print("--- Detailed Classification Metrics ---")
    
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1] if cm.shape == (2,2) else 0
    FP = cm[0, 1] if cm.shape == (2,2) else 0
    FN = cm[1, 0] if cm.shape == (2,2) else 0
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=target_names))
    
    dice_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    jaccard_iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    print(f"Dice Score (Malignant): {dice_score:.4f}")
    print(f"IoU (Malignant): {jaccard_iou:.4f}\n")
    
def plot_confusion_matrix(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5), dpi=300)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")
    plt.close()
