"""
数据清洗和预处理模块：
1. 过滤长度（10-50个氨基酸）
2. 过滤非标准氨基酸
3. 严格执行集合运算与 1:1 数据平衡
"""

import os
import pandas as pd
import logging
from Bio import SeqIO
import random

def filter_sequence(sequence: str, min_len: int = 10, max_len: int = 50) -> bool:
    """严格过滤序列长度和非标准氨基酸"""
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_amino_acids = set("BXZJOU")
    
    sequence = sequence.strip().upper()
    
    if not (min_len <= len(sequence) <= max_len):
        return False
    if not set(sequence).issubset(valid_amino_acids):
        return False
    if any(aa in invalid_amino_acids for aa in sequence):
        return False
        
    return True

def clean_fasta(input_fasta: str, output_fasta: str, min_len: int = 10, max_len: int = 50) -> list:
    """读取原始 FASTA 文件，过滤并保存清洗后的数据"""
    valid_records = []
    valid_seqs = []
    total_count = 0
    
    for record in SeqIO.parse(input_fasta, "fasta"):
        total_count += 1
        seq_str = str(record.seq).strip().upper()
        if filter_sequence(seq_str, min_len, max_len):
            valid_records.append(record)
            valid_seqs.append(seq_str)
            
    SeqIO.write(valid_records, output_fasta, "fasta")
    
    file_name = os.path.basename(input_fasta)
    logging.info(f"{file_name}: 原始序列共 {total_count} 条，经过长度({min_len}-{max_len})和标准氨基酸过滤后，保留 {len(valid_seqs)} 条")
    
    return valid_seqs

def create_hard_negatives(positive_fasta: str, amp_fasta: str, output_csv: str):
    """使用无抗病毒活性的抗菌肽 (AMPs) 作为难负样本构建最终数据集，严格保证 1:1"""
    # 1. 读取正样本序列
    pos_records = list(SeqIO.parse(positive_fasta, "fasta"))
    pos_set = set([str(record.seq).strip().upper() for record in pos_records])
    num_input_pos = len(pos_set)
    
    # 2. 读取难负样本序列并进行集合运算
    amp_records = list(SeqIO.parse(amp_fasta, "fasta"))
    amp_seqs = [str(record.seq).strip().upper() for record in amp_records]
    num_input_neg = len(amp_seqs)
    
    # 保留不在正样本中的负样本序列 (差集运算)
    hard_neg_seqs = list(set(amp_seqs) - pos_set)
    num_hard_neg = len(hard_neg_seqs)
    
    # 3. 极其严谨的 1:1 比例采样
    random.seed(42) # 保证每次跑的结果一致
    
    # 找到正负样本中较小的那个数量，作为最终的采样基数
    final_sample_size = min(num_input_pos, num_hard_neg)
    
    # ⚠️ 修复Bug的关键：正负样本都必须按照 final_sample_size 进行随机抽样
    selected_pos_seqs = random.sample(list(pos_set), final_sample_size)
    selected_neg_seqs = random.sample(hard_neg_seqs, final_sample_size)
    
    # 打印给导师看的详细日志
    logging.info(f"正样本输入: {num_input_pos} 条，负样本初始: {num_input_neg} 条")
    logging.info(f"差集运算后有效难负样本剩: {num_hard_neg} 条")
    logging.info(f"按1:1严谨平衡后最终数据集: 正负各 {final_sample_size} 条，共 {final_sample_size * 2} 条")
    
    # 4. 构建 DataFrame
    data = [{"sequence": seq, "label": 1} for seq in selected_pos_seqs] + \
           [{"sequence": seq, "label": 0} for seq in selected_neg_seqs]
        
    df = pd.DataFrame(data)
    
    # 随机打乱数据集
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. 保存为 CSV
    df.to_csv(output_csv, index=False)
    logging.info(f"Dataset created successfully. Saved to: {output_csv}")