"""
同源性去重模块：
使用 MMseqs2 进行序列聚类和去重，减少数据集中的冗余序列。
"""

import os
import subprocess
from Bio import SeqIO
import shutil

def run_mmseqs2_easy_cluster(input_fasta: str, output_prefix: str, tmp_dir: str, seq_id_threshold: float = 0.8, coverage: float = 0.8):
    """
    封装调用 MMseqs2 的 easy-cluster 命令进行同源性去重。
    
    Args:
        input_fasta (str): 待去重的输入 FASTA 文件路径
        output_prefix (str): MMseqs2 输出文件的前缀
        tmp_dir (str): 临时目录路径
        seq_id_threshold (float): 最小序列一致度 (默认 0.8)
        coverage (float): 最小覆盖度 (默认 0.8)
        
    Returns:
        str: 代表序列 (Representative sequences) 的 FASTA 文件路径
    """
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # 构建 easy-cluster 命令
        # mmseqs easy-cluster <i:fasta/fastq/db> <o:clusterPrefix> <tmpDir> [options]
        cmd = [
            "mmseqs", "easy-cluster",
            input_fasta,
            output_prefix,
            tmp_dir,
            "--min-seq-id", str(seq_id_threshold),
            "-c", str(coverage)
        ]
        
        print(f"Running MMseqs2 command: {' '.join(cmd)}")
        
        # 执行命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # MMseqs2 easy-cluster 默认生成 {output_prefix}_rep_seq.fasta
        rep_seq_file = f"{output_prefix}_rep_seq.fasta"
        
        if os.path.exists(rep_seq_file):
            print(f"MMseqs2 clustering completed successfully. Output: {rep_seq_file}")
            return rep_seq_file
        else:
            raise FileNotFoundError(f"Expected output file not found: {rep_seq_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"MMseqs2 clustering failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during MMseqs2 execution: {e}")
        raise

def cleanup_mmseqs2_files(output_prefix: str, tmp_dir: str):
    """
    清理 MMseqs2 运行过程中产生的中间文件和临时目录
    
    Args:
        output_prefix (str): 输出文件前缀
        tmp_dir (str): 临时目录路径
    """
    # 移除 easy-cluster 生成的非必要文件
    files_to_remove = [
        f"{output_prefix}_all_seqs.fasta",
        f"{output_prefix}_cluster.tsv"
    ]
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            
    # 移除临时目录
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
