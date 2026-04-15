import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score
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
    
    return {
        'ACC': acc,
        'Sn': sn,
        'Sp': sp,
        'MCC': mcc,
        'AUC': auc
    }

# ==========================================
# 4. 模型定义：GRUATTNet
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 使用简单的加性注意力 (Additive Attention)
        self.w = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, hidden_dim)
        # mask: (Batch, Seq_Len)
        
        # scores: (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
        scores = self.u(torch.tanh(self.w(x))).squeeze(-1)
        
        if mask is not None:
            # 将 padding 位置的注意力分数设为极小值，忽略其影响
            scores = scores.masked_fill(mask == 0, -1e4)
            
        # weights: (Batch, Seq_Len)
        weights = torch.softmax(scores, dim=-1)
        
        # context: (Batch, 1, Seq_Len) @ (Batch, Seq_Len, hidden_dim) -> (Batch, 1, hidden_dim) -> (Batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        
        return context


class GRUATTNet(nn.Module):
    def __init__(self, esm_model_name='facebook/esm2_t30_150M_UR50D'):
        super(GRUATTNet, self).__init__()
        # 1. 冻结的特征提取器: 加载 ESM-2 基座
        self.esm = EsmModel.from_pretrained(esm_model_name)
        for param in self.esm.parameters():
            param.requires_grad = False
            
        # 2. 时序层: GRU
        self.gru = nn.GRU(
            input_size=640,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. 自注意力层 (Attention): 输入特征维度为 128 * 2 = 256
        self.attention = Attention(hidden_dim=256)
        
        # 4. 分类头
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        # --- 步骤 1: 特征提取 ---
        esm_outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        x = esm_outputs.last_hidden_state
        
        # --- 步骤 2: 时序处理 ---
        gru_out, _ = self.gru(x)
        
        # --- 步骤 3: 自注意力池化 ---
        context = self.attention(gru_out, attention_mask)
        
        # --- 步骤 4: 分类头 ---
        out = self.dropout(context)
        logits = self.classifier(out)
        
        return logits

# ==========================================
# 5. 训练管线
# ==========================================
def run_baseline_1(dataset_path, output_dir="models/baseline_1_weights"):
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
    train_batch_size = 4
    val_batch_size = 8
    learning_rate = 2e-4
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"========== Starting Fold {fold}/5 ==========")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = AVPDataset(X_train, y_train, tokenizer)
        val_dataset = AVPDataset(X_val, y_val, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        model = GRUATTNet(esm_model_name=model_name).to(device)
        
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
    for metric in ['ACC', 'Sn', 'Sp', 'MCC', 'AUC']:
        result_metrics[metric] = mean_metrics[metric]
        result_metrics[f"{metric}_std"] = std_metrics[metric]
        
    logging.info(f"Baseline 1 (GRUATTNet) 5-Fold CV Completed. Avg AUC: {result_metrics['AUC']:.4f} ± {result_metrics['AUC_std']:.4f}")
    
    params = {
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'epochs': epochs
    }
    
    return "GRUATTNet", params, result_metrics
