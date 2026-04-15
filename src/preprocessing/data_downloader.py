"""
数据下载模块：
用于从开源数据库或 GitHub 自动拉取原始的 FASTA 数据。
"""

import os
import requests
from tqdm import tqdm

# 虚拟的 GitHub Raw URL 占位符，后续可替换为真实的下载链接
AVP_URL = "https://raw.githubusercontent.com/username/AVP-ESM2-Comparison/main/data/raw/avp_positive.fasta"
AMP_URL = "https://raw.githubusercontent.com/username/AVP-ESM2-Comparison/main/data/raw/amp_negative.fasta"

def download_file(url: str, save_path: str):
    """
    使用 requests 和 tqdm 库健壮地下载文件，并在终端显示进度条。
    
    Args:
        url (str): 文件的下载链接
        save_path (str): 文件保存的本地路径
    """
    try:
        # 发送 HTTP GET 请求，启用 stream 以便分块读取
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 获取文件总大小 (可能为 None)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        # 确保目标目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=os.path.basename(save_path))
        
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
                
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"WARNING: Something went wrong while downloading {url}")
        else:
            print(f"Successfully downloaded {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        # 如果下载失败，尝试清理未完成的文件
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
