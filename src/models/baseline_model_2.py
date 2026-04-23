import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, f1_score
from tqdm import tqdm

# ==========================================
# 1. 核心损失函数：Binary Focal Loss
# ==========================================
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# 2. 数据集包装类
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
# 3. 生信指标计算函数
# ==========================================
def calculate_bio_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
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
# ==========================================
# 4. 模型定义：PeptideNet_Arch (已修复 Mask 泄露漏洞)
# ==========================================
class PeptideNet_Arch(nn.Module):
    def __init__(self, esm_model_name='facebook/esm2_t30_150M_UR50D'):
        super(PeptideNet_Arch, self).__init__()
        # 1. 冻结的特征提取器
        self.esm = EsmModel.from_pretrained(esm_model_name)
        for param in self.esm.parameters():
            param.requires_grad = False
            
        # 2. CNN 块 1
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=640, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 3. CNN 块 2
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 4. 时序层: GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        # --- 步骤 1: 特征提取 ---
        esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        x = esm_outputs.last_hidden_state # [B, 1024, 640]
        
        # --- 步骤 2: CNN 块提取 ---
        x = x.transpose(1, 2) # [B, 640, 1024]
        x = self.cnn_block1(x) # 经过MaxPool, L 变为 512
        x = self.cnn_block2(x) # 经过MaxPool, L 变为 256
        x = x.transpose(1, 2) # [B, 256, 128]
        
        # --- 步骤 3: 时序处理 ---
        gru_out, _ = self.gru(x) # [B, 256, 256]
        
        # --- 步骤 4: 动态 Mask 平均池化 (解决 Padding 泄露) ---
        # 因为经过了两次 MaxPool1d(kernel_size=2)，序列长度缩减到了 1/4
        # 我们需要对 attention_mask 也进行同等比例的缩放
        mask_pooled = attention_mask[:, ::4].float() # 每 4 个取 1 个，形状 [B, 256]
        
        # 将 mask 扩展维度以匹配 gru_out
        mask_expanded = mask_pooled.unsqueeze(-1) # [B, 256, 1]
        
        # 只累加有效位置的特征
        sum_gru = torch.sum(gru_out * mask_expanded, dim=1) 
        
        # 计算每条序列实际的有效长度（防止除以0）
        valid_lens = torch.clamp(mask_pooled.sum(dim=1, keepdim=True), min=1e-9)
        
        # 得到真正代表氨基酸序列的特征
        pooled = sum_gru / valid_lens
        
        # --- 步骤 5: 分类 ---
        out = self.dropout(pooled)
        logits = self.classifier(out)
        
        return logits
# 5. 训练管线
# ==========================================
def run_baseline_2(dataset_path, output_dir="models/baseline_2_weights"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(dataset_path)
    X = df['sequence'].values
    y = df['label'].values
    
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    epochs = 10
    train_batch_size = 128
    val_batch_size = 256
    learning_rate = 2e-4
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"========== Starting Fold {fold}/5 ==========")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = AVPDataset(X_train, y_train, tokenizer)
        val_dataset = AVPDataset(X_val, y_val, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        model = PeptideNet_Arch(esm_model_name=model_name).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = BinaryFocalLoss(gamma=2.0, alpha=0.5).to(device)
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        best_auc = 0.0
        best_model_path = os.path.join(output_dir, f"best_model_fold_{fold}.pt")
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                    
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
            
            val_auc = roc_auc_score(val_targets, val_probs)
            logging.info(f"Fold {fold} - Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"-> Saved new best model with AUC: {best_auc:.4f}")
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
        
        final_val_preds = (np.array(final_val_probs) >= 0.5).astype(int)
        
        metrics = calculate_bio_metrics(final_val_targets, final_val_preds, final_val_probs)
        fold_metrics.append(metrics)
        logging.info(f"Fold {fold} Best Metrics: {metrics}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_metrics = pd.DataFrame(fold_metrics)
    mean_metrics = df_metrics.mean().to_dict()
    std_metrics = df_metrics.std().to_dict()
    
    result_metrics = {}
    for metric in ['ACC', 'Sn', 'Sp', 'MCC', 'AUC', 'F1']:
        result_metrics[metric] = mean_metrics[metric]
        result_metrics[f"{metric}_std"] = std_metrics[metric]
        
    logging.info(f"Baseline 2 (PeptideNet_Arch) 5-Fold CV Completed. Avg AUC: {result_metrics['AUC']:.4f} ± {result_metrics['AUC_std']:.4f}")
    
    params = {
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'epochs': epochs
    }
    
    return "PeptideNet", params, result_metrics
