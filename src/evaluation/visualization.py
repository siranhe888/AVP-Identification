"""
可视化模块：包含模型性能对比、t-SNE 降维分析与 ESM-2 Attention Map 可视化
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import torch
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel

# 设置全局绘图风格
plt.style.use('default')
sns.set_palette("muted")

def plot_tsne(features: np.ndarray, labels: np.ndarray, save_path: str = "data/processed/tsne_plot.png"):
    """对高维特征进行 t-SNE 降维并可视化分布"""
    print("正在进行 t-SNE 降维计算...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", len(np.unique(labels))),
        alpha=0.7, edgecolor='w', s=60
    )
    plt.title("t-SNE Visualization of Protein Features", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Class", title_fontsize='13', fontsize='11')
    
    # 去除顶部和右侧边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE 图已保存至: {save_path}")

def plot_attention_map(sequence: str, model_name: str = "facebook/esm2_t30_150M_UR50D", layer: int = -1, head: int = 0, save_path: str = "data/processed/attention_map.png"):
    """提取预训练 ESM-2 的注意力权重并绘制热图"""
    print(f"正在生成注意力热图 (Layer {layer}, Head {head})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    inputs = tokenizer(sequence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    attentions = outputs.attentions
    attention_matrix = attentions[layer][0, head].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Attention Map (Layer {layer}, Head {head})", fontsize=16, fontweight='bold')
    plt.xlabel("Tokens", fontsize=12)
    plt.ylabel("Tokens", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention Map 已保存至: {save_path}")

def plot_metrics(csv_path="data/processed/optimized_results.csv", output_dir="data/processed/"):
    """绘制 5 折交叉验证的模型性能对比柱状图和雷达图"""
    if not os.path.exists(csv_path):
        print(f"未找到数据文件 {csv_path}")
        return
        
    print("正在绘制模型性能对比图...")
    df = pd.read_csv(csv_path)
    models = df['Model'].tolist()
    
    # 提取 "均值 ± 标准差"
    metrics_data = {'Model': models}
    metrics = ['ACC', 'Sn', 'Sp', 'MCC', 'AUC']
    for metric in metrics:
        means, stds = [], []
        for val in df[metric]:
            if isinstance(val, str) and ' ± ' in val:
                m, s = val.split(' ± ')
                means.append(float(m))
                stds.append(float(s))
            else:
                means.append(float(val))
                stds.append(0.0)
        metrics_data[metric] = means
        metrics_data[f'{metric}_std'] = stds
        
    plot_df = pd.DataFrame(metrics_data)
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B'] # LoRA用红色

    # --- 1. 绘制分组柱状图 ---
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    x = np.arange(len(metrics))
    width = 0.2

    for i, row in plot_df.iterrows():
        means = [row[m] for m in metrics]
        stds = [row[f'{m}_std'] for m in metrics]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, label=row['Model'], 
               color=colors[i % len(colors)], capsize=5, alpha=0.9, edgecolor='white', linewidth=1)

    ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison of Different Models (5-Fold CV)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=11, frameon=False)
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_bar_chart.png'))
    plt.close()

    # --- 2. 绘制雷达图 ---
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'AUC'], fontsize=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1.0"], color="grey", size=10)
    plt.ylim(0.7, 1.0)

    for i, row in plot_df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        
        is_lora = 'LoRA' in row['Model']
        linewidth = 2.5 if is_lora else 1.5
        linestyle = '-' if is_lora else '--'
        c = colors[i % len(colors)]
        
        ax.plot(angles, values, linewidth=linewidth, linestyle=linestyle, color=c, label=row['Model'])
        if is_lora:
            ax.fill(angles, values, color=c, alpha=0.1)

    ax.set_title("Multi-dimensional Metric Comparison", size=16, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_radar_chart.png'))
    plt.close()
    print("模型对比柱状图和雷达图已生成！")

if __name__ == "__main__":
    # 测试执行性能对比图生成
    plot_metrics()