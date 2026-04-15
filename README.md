# AVP-ESM2-Comparison

## 项目背景
本项目旨在对比传统机器学习（SVM, RF, XGBoost）与预训练大模型（ESM-2）在识别抗病毒肽（Antiviral Peptides, AVPs，长度在 10-50 个氨基酸）上的性能。

核心亮点包括：
1. **难负样本挖掘 (Hard Negative Mining)**：使用无抗病毒活性的抗菌肽 (AMPs) 作为难负样本，以提高模型在实际应用中的鉴别能力。
2. **严格的数据去重**：使用 MMseqs2 进行 80% 相似度的序列去重，减少数据泄露和过拟合。
3. **前沿的大模型微调**：基于 PEFT (Parameter-Efficient Fine-Tuning) 的 LoRA 技术对 ESM-2 大模型进行微调。
4. **可解释性与可视化**：提供 ESM-2 的注意力机制可视化 (Attention Map) 和特征分布的 t-SNE 可视化。

## 模块分工
- **`data/`**：存放原始 FASTA 数据 (`raw/`) 以及清洗和去重后的数据 (`processed/`)。
- **`src/preprocessing/`**：数据清洗代码（长度过滤 10-50aa，标准氨基酸过滤）以及 MMseqs2 去重流程。
- **`src/features/`**：特征提取模块，包括传统机器学习所需的理化特征提取 (`traditional.py`)，以及 ESM-2 特征提取 (`esm_embedding.py`)。
- **`src/models/`**：模型定义与训练脚本。包含基线模型 (`baseline_models.py`) 和基于 LoRA 的大模型微调代码 (`esm_lora_finetuning.py`)。
- **`src/evaluation/`**：模型评估与可视化。包含性能指标计算 (`metrics.py`) 和特征可视化代码 (`visualization.py`)。
- **`configs/`**：存放项目所需的各种路径与超参数配置。

## 快速开始

1. **安装依赖**:
```bash
pip install -r requirements.txt
```

2. **运行主程序 Pipeline**:
```bash
python main.py
```
