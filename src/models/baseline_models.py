"""
基线机器学习模型训练与评估 (SVM, RF, XGBoost)
实现论文 4.3.2 节的基线对比实验，基于 5 折交叉验证。
"""

import numpy as np
import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import optuna

def calculate_bio_metrics(y_true, y_pred, y_prob):
    """计算生物信息学常用指标：ACC, Sn, Sp, MCC, AUC, F1"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'ACC': acc,
        'Sn': sn,
        'Sp': sp,
        'MCC': mcc,
        'AUC': auc,
        'F1': f1
    }

def run_5fold_cv(model, X: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    """对单个模型执行 5 折交叉验证并返回平均指标"""
    logging.info(f"Starting 5-fold CV for {model_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
        
        # 计算当前折指标
        metrics = calculate_bio_metrics(y_val, y_pred, y_prob)
        fold_metrics.append(metrics)
        
        logging.debug(f"{model_name} Fold {fold}: ACC={metrics['ACC']:.4f}, AUC={metrics['AUC']:.4f}")
        
    # 汇总并计算平均值与标准差
    df_metrics = pd.DataFrame(fold_metrics)
    mean_metrics = df_metrics.mean().to_dict()
    std_metrics = df_metrics.std().to_dict()
    
    result = {}
    for metric in ['ACC', 'Sn', 'Sp', 'MCC', 'AUC', 'F1']:
        result[metric] = mean_metrics[metric]
        result[f"{metric}_std"] = std_metrics[metric]
        
    logging.info(f"{model_name} 5-Fold CV Completed. Avg AUC: {result['AUC']:.4f} ± {result['AUC_std']:.4f}")
    return result

def tune_svm(X: np.ndarray, y: np.ndarray) -> tuple:
    """使用 GridSearchCV 调优 SVM 并返回最佳参数和评估结果 (防止数据泄露)"""
    logging.info("Tuning SVM using GridSearchCV with Pipeline...")
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = {k.replace('svm__', ''): v for k, v in grid_search.best_params_.items()}
    logging.info(f"Best SVM params found: {best_params}")
    
    metrics = run_5fold_cv(best_model, X, y, "SVM (Tuned)")
    return "SVM", best_params, metrics

def tune_rf(X: np.ndarray, y: np.ndarray) -> tuple:
    """使用 GridSearchCV 调优 Random Forest 并返回最佳参数和评估结果 (防止数据泄露)"""
    logging.info("Tuning Random Forest using GridSearchCV with Pipeline...")
    param_grid = {
        'rf__n_estimators': [50, 100, 200, 300],
        'rf__max_depth': [None, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10]
    }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = {k.replace('rf__', ''): v for k, v in grid_search.best_params_.items()}
    logging.info(f"Best Random Forest params found: {best_params}")
    
    metrics = run_5fold_cv(best_model, X, y, "Random Forest (Tuned)")
    return "Random Forest", best_params, metrics

def tune_xgboost(X: np.ndarray, y: np.ndarray) -> tuple:
    """使用 Optuna 进行 XGBoost 的贝叶斯优化 (TPE) - 启用 GPU 加速防崩溃版"""
    logging.info("Tuning XGBoost using Optuna (TPE) on GPU...")
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_estimators': 200,
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',  # 启用直方图加速
            'device': 'cuda'        # 强制启用 GPU 加速
        }
        
        # 忽略训练时的警告
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = XGBClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # 注意：这里坚决不能加 n_jobs=-1，否则 Colab 内存会因为 GPU 并发溢出
            scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        return scores.mean()

    # 隐藏 Optuna 每一步烦人的打印
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    # 使用 50 次智能贝叶斯搜索
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    logging.info(f"Best XGBoost params found: {best_params}")
    
    # 用找到的最佳参数重新实例化模型
    best_model = XGBClassifier(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        n_estimators=200,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',   # 必须同步加上
        device='cuda'         # 必须同步加上
    )
    metrics = run_5fold_cv(best_model, X, y, "XGBoost (Tuned)")
    return "XGBoost", best_params, metrics

def format_results(name: str, best_params: dict, metrics: dict, group_name: str) -> dict:
    """格式化评估结果用于展示和保存"""
    return {
        'Group': group_name,
        'Model': name,
        'Best Params': str(best_params),
        'ACC': f"{metrics['ACC']:.4f} ± {metrics['ACC_std']:.4f}",
        'Sn': f"{metrics['Sn']:.4f} ± {metrics['Sn_std']:.4f}",
        'Sp': f"{metrics['Sp']:.4f} ± {metrics['Sp_std']:.4f}",
        'MCC': f"{metrics['MCC']:.4f} ± {metrics['MCC_std']:.4f}",
        'AUC': f"{metrics['AUC']:.4f} ± {metrics['AUC_std']:.4f}",
        'F1': f"{metrics['F1']:.4f} ± {metrics['F1_std']:.4f}"
    }