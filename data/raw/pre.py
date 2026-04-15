import pandas as pd
import os

def extract_escape_hard_negatives(input_files, output_file):
    """
    从 ESCAPE 的 CSV 分片中提取难负样本 (Antibacterial=1 & Antiviral=0)
    """
    all_neg_seqs = []
    
    for file_name in input_files:
        if not os.path.exists(file_name):
            print(f"警告: 找不到文件 {file_name}")
            continue
            
        # 读取 CSV
        df = pd.read_csv(file_name)
        
        # 筛选逻辑：具有抗菌活性 (1) 且 明确不具备抗病毒活性 (0)
        # 这对应了论文中“定向挖掘强效抗菌但无抗病毒活性记录”的设定
        condition = (df['Antibacterial'] == 1) & (df['Antiviral'] == 0)
        filtered_df = df[condition]
        
        # 提取 Sequence 列并去除空值
        # 注意：部分行可能只有 Hash 而没有 Sequence，我们只取包含序列的行
        sequences = filtered_df['Sequence'].dropna().tolist()
        all_neg_seqs.extend(sequences)
        print(f"从 {file_name} 中提取了 {len(sequences)} 条符合条件的候选序列。")

    # 去重处理
    unique_seqs = list(set(all_neg_seqs))
    
    # 转换为 FASTA 格式以供后续 clean_fasta 使用
    with open(output_file, "w", encoding="utf-8") as f:
        for i, seq in enumerate(unique_seqs):
            f.write(f">ESCAPE_hard_neg_{i}\n{seq}\n")
            
    print(f"处理完成！最终提取到 {len(unique_seqs)} 条难负样本，已保存为: {output_file}")

if __name__ == "__main__":
    # 指定当前文件夹下的分片文件
    files = ["Fold1.csv", "Fold2.csv"]
    # 输出到你项目定义的原始数据路径
    output_path = "amp_negative.fasta" 
    
    extract_escape_hard_negatives(files, output_path)