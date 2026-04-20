"""
评估指标计算：ACC, MCC, Sn, Sp, AUC
"""

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, f1_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    计算二分类模型性能指标
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测类别
        y_prob (np.ndarray): 预测正类别的概率
        
    Returns:
        dict: 包含各项指标的字典 (ACC, MCC, Sn, Sp, AUC)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 敏感度/召回率 (Sensitivity)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 特异性 (Specificity)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC-AUC
    auc = roc_auc_score(y_true, y_prob)
    
    # Compute F1 Score using sklearn
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred)
    metrics = {
        "Accuracy (ACC)": acc,
        "Matthews Correlation Coefficient (MCC)": mcc,
        "Sensitivity (Sn)": sn,
        "Specificity (Sp)": sp,
        "Area Under Curve (AUC)": auc,
        "F1 Score (F1)": f1
    }
    
    return metrics

def print_metrics(metrics: dict):
    """打印评估指标结果"""
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
