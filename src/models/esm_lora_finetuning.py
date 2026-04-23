import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmModel, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, f1_score
from tqdm import tqdm

# ==========================================
# 1. 核心损失函数：Binary Focal Loss
# ==========================================
class BinaryFocalLoss(nn.Module):
    """
    针对二分类任务的 Focal Loss。
    用于强迫模型关注那些长得极像 AVP 的难负样本（AMP）。
    """
    def __init__(self, gamma=2.0, alpha=0.5, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits 维度: (batch_size, 1), targets 维度: (batch_size, 1)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if target=1 else 1-p
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# 2. 自定义网络与分类头：ESM2 LoRA Classifier
# ==========================================
class ESM2LoRAClassifier(nn.Module):
    """
    基于 ESM-2 的端到端微调模型，包含动态均值池化和自定义分类头。
    """
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", lora_r: int = 8):
        super(ESM2LoRAClassifier, self).__init__()
        # 1. 加载预训练基座模型 (不要自带的分类头)
        self.esm = EsmModel.from_pretrained(model_name)
        
        # 2. 配置 LoraConfig
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=["query", "key", "value", "dense"], # 微调注意力层和全连接层以唤醒进化特征
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION" # ESM2 不是标准的序列分类任务，我们手动处理池化
        )
        
        # 3. 使用 peft 库包装基座模型
        self.esm = get_peft_model(self.esm, lora_config)
        
        # ESM-2 150M 的隐藏层维度为 640
        hidden_size = 640
        
        # 4. 构建自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 320),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(320, 1)
        )

    def forward(self, input_ids, attention_mask):
        # 1. 前向传播提取特征
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 2. 动态均值池化 (Dynamic Mean Pooling)
        # 将 attention_mask 扩展到与 hidden_state 相同的维度
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # 将 padding 部分的特征置为 0
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        # 计算真实序列的长度 (避免除以 0)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 计算特征均值，剔除 Padding 干扰
        pooled_output = sum_embeddings / sum_mask  # (batch_size, hidden_size)
        
        # 3. 接入分类网络
        logits = self.classifier(pooled_output)  # (batch_size, 1)
        
        return logits

# ==========================================
# 3. 数据集包装类
# ==========================================
class AVPDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=1024):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# ==========================================
# 4. 生信指标计算函数
# ==========================================
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

# ==========================================
# 5. 训练管线：严谨的 5-Fold 微调流程
# ==========================================
def run_lora_finetuning(dataset_path, output_dir="data/processed/lora_weights", lora_r: int = 8):
    """执行基于 LoRA 的大语言模型端到端微调 (5-Fold CV)"""
    # 自动检测 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据集
    df = pd.read_csv(dataset_path)
    X = df['sequence'].values
    y = df['label'].values
    
    # 配置分词器
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # K 折交叉验证设置
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    # 超参数配置
    epochs = 10
    train_batch_size = 4  # 针对 8GB 显存极限压缩
    val_batch_size = 8
    learning_rate = 2e-4
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"========== Starting Fold {fold}/5 ==========")
        
        # 划分数据集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = AVPDataset(X_train, y_train, tokenizer)
        val_dataset = AVPDataset(X_val, y_val, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        # 初始化模型
        model = ESM2LoRAClassifier(model_name=model_name, lora_r=lora_r).to(device)
        
        # 优化器与损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = BinaryFocalLoss(gamma=2.0, alpha=0.5).to(device)
        
        # 学习率调度器：带有 Warmup 的 Cosine 衰减
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # 初始化混合精度训练的 GradScaler
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        best_auc = 0.0
        best_model_path = os.path.join(output_dir, f"best_model_fold_{fold}.pt")
        
        # 开始训练循环
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                # 使用混合精度上下文管理器
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                    
                    # 缩放损失并反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证循环
            model.eval()
            val_probs = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].numpy()
                    
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            logits = model(input_ids, attention_mask)
                    else:
                        logits = model(input_ids, attention_mask)
                        
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    
                    val_probs.extend(probs)
                    val_targets.extend(labels)
            
            # 计算当前 Epoch 的验证集 AUC
            val_auc = roc_auc_score(val_targets, val_probs)
            logging.info(f"Fold {fold} - Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val AUC: {val_auc:.4f}")
            
            # 实施 Early Stopping（保存每折最高 AUC 的模型权重）
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"-> Saved new best model with AUC: {best_auc:.4f}")
                
            # Epoch 结束时清理显存垃圾
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ================================
        # 使用该折最优模型计算 5 项生信指标
        # ================================
        logging.info(f"Evaluating best model for Fold {fold}...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        final_val_probs = []
        final_val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].numpy()
                
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask)
                else:
                    logits = model(input_ids, attention_mask)
                    
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                final_val_probs.extend(probs)
                final_val_targets.extend(labels)
        
        # 将概率转为二分类预测结果 (阈值 0.5)
        final_val_preds = (np.array(final_val_probs) >= 0.5).astype(int)
        
        metrics = calculate_bio_metrics(final_val_targets, final_val_preds, final_val_probs)
        fold_metrics.append(metrics)
        logging.info(f"Fold {fold} Best Metrics: {metrics}")
        
        # Fold 结束时清理显存垃圾
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总并计算 5 折指标的均值和标准差
    df_metrics = pd.DataFrame(fold_metrics)
    mean_metrics = df_metrics.mean().to_dict()
    std_metrics = df_metrics.std().to_dict()
    
    result_metrics = {}
    for metric in ['ACC', 'Sn', 'Sp', 'MCC', 'AUC', 'F1']:
        result_metrics[metric] = mean_metrics[metric]
        result_metrics[f"{metric}_std"] = std_metrics[metric]
        
    logging.info(f"LoRA Finetuning 5-Fold CV Completed. Avg AUC: {result_metrics['AUC']:.4f} ± {result_metrics['AUC_std']:.4f}")
    
    params = {
        'r': lora_r,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_modules': ["query", "key", "value", "dense"],
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'epochs': epochs
    }
    
    return "ESM-2 LoRA Finetuned", params, result_metrics

# ==========================================
# 6. Rank 消融实验
# ==========================================
def run_rank_ablation(dataset_path):
    """运行 ESM-2 LoRA Rank 参数的消融实验"""
    rank_list = [4, 8, 16, 32]
    all_results = []
    
    for r in rank_list:
        logging.info(f"=== Starting Ablation for LoRA Rank: {r} ===")
        # 调用 run_lora_finetuning
        name, params, metrics = run_lora_finetuning(
            dataset_path=dataset_path, 
            output_dir=f"data/processed/lora_weights_r{r}", 
            lora_r=r
        )
        # 收集结果
        result_dict = {'Rank': r}
        result_dict.update(metrics)
        all_results.append(result_dict)
        
    # 保存至 CSV
    df_ablation = pd.DataFrame(all_results)
    output_csv = "data/processed/ablation_rank_results.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_ablation.to_csv(output_csv, index=False)
    logging.info(f"Rank 消融实验完成，结果已保存至 {output_csv}")
    return output_csv
