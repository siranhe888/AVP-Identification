"""
ESM-2 特征提取模块：使用预训练的 ESM-2 模型提取序列的嵌入表示 (Embeddings)
"""

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

def load_esm_model(model_name: str = "facebook/esm2_t30_150M_UR50D"):
    """
    加载 ESM-2 预训练模型和分词器
    
    Args:
        model_name (str): HuggingFace 模型名称
        
    Returns:
        tuple: (分词器, ESM-2模型)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def extract_esm_embeddings(sequences: List[str], model_name: str = "facebook/esm2_t30_150M_UR50D", batch_size: int = 32, device: str = "cuda") -> pd.DataFrame:
    """
    使用 ESM-2 提取全长序列特征表示 (取 CLS token 或是全长均值)
    
    Args:
        sequences (List[str]): 待提取的氨基酸序列列表
        model_name (str): HuggingFace 模型名称
        batch_size (int): 批处理大小
        device (str): 运行设备 (cuda 或 cpu)
        
    Returns:
        pd.DataFrame: 包含序列嵌入特征的 DataFrame
    """
    tokenizer, model = load_esm_model(model_name)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        device = "cpu"
    
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM-2 embeddings"):
            batch_seqs = sequences[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # 使用最后隐藏层状态的均值 (Mean Pooling) 或者直接使用 CLS token
            # 这里示例使用所有 token 隐藏状态的均值
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mask out padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.extend(mean_embeddings.cpu().numpy())
            
    embeddings_df = pd.DataFrame(all_embeddings, columns=[f"esm2_dim_{i}" for i in range(len(all_embeddings[0]))])
    return embeddings_df
