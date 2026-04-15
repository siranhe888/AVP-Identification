import pandas as pd
import numpy as np

print("========== ESM-2 深度特征 (150M) 体检报告 ==========")

try:
    # 1. 加载生成的 ESM-2 特征文件
    df_esm = pd.read_csv("data/processed/features_esm2.csv")
    
    # 2. 检验维度：预期应该是 904 行，642 列 (sequence + label + 640维特征)
    print(f"当前数据形状 (行, 列): {df_esm.shape}")
    
    # 因为有 sequence 和 label 两列，所以总列数应该是 640 + 2 = 642
    if df_esm.shape[1] == 642:
        print("✅ 维度检验通过：完美匹配 640 维特征！(证明你成功用上了 1.5 亿参数的 ESM-2 旗舰模型)")
    else:
        print(f"❌ 维度错误：特征数为 {df_esm.shape[1]-2}，不是 640 维。")
        print("   -> 请检查 src/features/esm_embedding.py 中是否真的改成了 'facebook/esm2_t30_150M_UR50D'")

    # 3. 检验空值 (验证 Average Pooling 是否正常运行)
    # 如果 Pooling 处理不定长序列时除以了 0，就会产生 NaN
    null_counts = df_esm.isnull().sum().sum()
    if null_counts == 0:
        print("✅ 数据完整性检验通过：没有任何缺失值，Average Pooling (平均池化) 完美处理了所有长短不一的多肽！")
    else:
        print(f"❌ 异常警告：发现了 {null_counts} 个缺失值 (NaN)！")

except FileNotFoundError:
    print("❌ 找不到 features_esm2.csv 文件！")
    print("   -> 请确保你在跑这个体检脚本前，已经运行了: python main.py --step feature")

print("====================================================")