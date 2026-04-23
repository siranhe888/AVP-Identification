"""
主程序入口：抗病毒肽（AVP）识别比较框架
"""

import argparse
import logging
import os
import pandas as pd
from configs.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MIN_SEQ_LEN, MAX_SEQ_LEN, MMSEQS_SIMILARITY_THRESHOLD
from src.preprocessing.data_cleaner import clean_fasta, create_hard_negatives
from src.preprocessing.homology import run_mmseqs2_easy_cluster, cleanup_mmseqs2_files
from src.preprocessing.data_downloader import download_file, AVP_URL, AMP_URL

# --- 导入特征提取模块 ---
from src.features.traditional import generate_traditional_features
from src.features.esm_embedding import extract_esm_embeddings

# --- 导入模型训练模块 ---
from src.models.baseline_models import tune_svm, tune_rf, tune_xgboost, format_results
from src.models.esm_lora_finetuning import run_lora_finetuning
from src.models.baseline_model_1 import run_baseline_1
from src.models.baseline_model_2 import run_baseline_2
from src.models.baseline_model_3 import run_baseline_3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AVP Identification Pipeline")
    parser.add_argument('--step', type=str, choices=['all', 'download', 'preprocess', 'feature', 'train', 'lora', 'b1', 'b2', 'b3', 'evaluate', 'ablation'],
                        default='all', help='Pipeline step to run')
    return parser.parse_args()

def main():
    """主调度函数"""
    args = parse_args()
    logging.info(f"Starting pipeline. Step: {args.step}")

    # 定义路径
    raw_avp_fasta = os.path.join(RAW_DATA_DIR, "avp_positive.fasta")
    raw_amp_fasta = os.path.join(RAW_DATA_DIR, "amp_negative.fasta")
    
    cleaned_avp_fasta = os.path.join(PROCESSED_DATA_DIR, "cleaned_avp.fasta")
    cleaned_amp_fasta = os.path.join(PROCESSED_DATA_DIR, "cleaned_amp.fasta")
    
    final_dataset_csv = os.path.join(PROCESSED_DATA_DIR, "final_dataset.csv")

    # --- Step 0: 数据下载 ---
    if args.step in ['all', 'download']:
        logging.info("--- Step 0: Data Downloading ---")
        if not os.path.exists(raw_avp_fasta):
            logging.info(f"Downloading AVP positive data...")
            download_file(AVP_URL, raw_avp_fasta)
        if not os.path.exists(raw_amp_fasta):
            logging.info(f"Downloading AMP negative data...")
            download_file(AMP_URL, raw_amp_fasta)

    # --- Step 1: 数据预处理 (全自动 MMseqs2 模式) ---
    if args.step in ['all', 'preprocess']:
        logging.info("--- Step 1: Data Preprocessing (Automated Mode) ---")
        
        # Phase 1: 长度与残基清洗
        logging.info("--- Phase 1: Cleaning raw FASTA files ---")
        clean_fasta(raw_avp_fasta, cleaned_avp_fasta, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
        clean_fasta(raw_amp_fasta, cleaned_amp_fasta, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
        
        # Phase 2: 自动运行 MMseqs2 去重
        logging.info("--- Phase 2: Homology reduction using MMseqs2 ---")
        
        # AVP 去重
        avp_output_prefix = os.path.join(PROCESSED_DATA_DIR, "avp_cluster")
        avp_tmp_dir = "/tmp/tmp_avp"  # 使用Linux原生tmp，避开Google Drive权限报错
        logging.info(f"Running MMseqs2 for AVP sequences...")
        avp_rep_seq_fasta = run_mmseqs2_easy_cluster(
            cleaned_avp_fasta, avp_output_prefix, avp_tmp_dir, 
            seq_id_threshold=MMSEQS_SIMILARITY_THRESHOLD
        )
        cleanup_mmseqs2_files(avp_output_prefix, avp_tmp_dir)
        
        # AMP 去重
        amp_output_prefix = os.path.join(PROCESSED_DATA_DIR, "amp_cluster")
        amp_tmp_dir = "/tmp/tmp_amp"
        logging.info(f"Running MMseqs2 for AMP sequences...")
        amp_rep_seq_fasta = run_mmseqs2_easy_cluster(
            cleaned_amp_fasta, amp_output_prefix, amp_tmp_dir, 
            seq_id_threshold=MMSEQS_SIMILARITY_THRESHOLD
        )
        cleanup_mmseqs2_files(amp_output_prefix, amp_tmp_dir)

        # Phase 3: 构建最终平衡数据集
        logging.info("--- Phase 3: Creating hard negatives and building final dataset ---")
        create_hard_negatives(
            positive_fasta=avp_rep_seq_fasta,
            amp_fasta=amp_rep_seq_fasta,
            output_csv=final_dataset_csv
        )
        logging.info(f"自动预处理流水线完成。数据集已保存至: {final_dataset_csv}")

    # --- Step 2: 特征提取 ---
    if args.step in ['all', 'feature']:
        logging.info("--- Step 2: Feature Extraction ---")
        if not os.path.exists(final_dataset_csv):
            logging.error(f"错误：找不到数据集 {final_dataset_csv}。请先运行 --step preprocess")
            return
            
        # 加载数据
        df = pd.read_csv(final_dataset_csv)
        
        # 1. 提取传统理化特征 (AAC, DPC, 理化特征 等共 434 维)
        logging.info("Extracting traditional physicochemical features (434 dims)...")
        df_with_traditional = generate_traditional_features(df)
        
        # 直接保存原始特征，Z-score 标准化严格放在交叉验证 Pipeline 内部进行，防泄露
        traditional_out = os.path.join(PROCESSED_DATA_DIR, "features_traditional.csv")
        df_with_traditional.to_csv(traditional_out, index=False)
        logging.info(f"传统特征(原始未标准化)已保存至 {traditional_out}")
        
        # 2. 提取 ESM-2 嵌入特征
        logging.info("Extracting ESM-2 embeddings (this may take a while)...")
        sequences = df['sequence'].tolist()
        # 使用 150M 参数模型 (隐藏层 640 维) 对齐论文描述
        df_esm_embeddings = extract_esm_embeddings(sequences, model_name="facebook/esm2_t30_150M_UR50D")
        
        # 合并标签并保存
        df_esm_final = pd.concat([df[['sequence', 'label']], df_esm_embeddings], axis=1)
        esm_out = os.path.join(PROCESSED_DATA_DIR, "features_esm2.csv")
        df_esm_final.to_csv(esm_out, index=False)
        logging.info(f"ESM-2 特征已保存至 {esm_out}")

    # --- Step 3: 模型训练 ---
    if args.step in ['all', 'train']:
        logging.info("--- Step 3: Model Tuning & Evaluation ---")
        traditional_out = os.path.join(PROCESSED_DATA_DIR, "features_traditional.csv")
        esm_out = os.path.join(PROCESSED_DATA_DIR, "features_esm2.csv")
        
        results_list = []
        
        # === 传统组: SVM 和 Random Forest ===
        if os.path.exists(traditional_out):
            logging.info("Loading traditional features dataset (434 dims)...")
            df_trad = pd.read_csv(traditional_out)
            y_trad = df_trad['label'].values
            
            # 排除非特征列
            feature_cols_trad = [c for c in df_trad.columns if c not in ['sequence', 'label']]
            X_trad = df_trad[feature_cols_trad].values
            
            logging.info(f"Traditional Group shape for training: X={X_trad.shape}, y={y_trad.shape}")
            
            # 调优并评估 SVM 和 RF
            name, params, metrics = tune_svm(X_trad, y_trad)
            results_list.append(format_results(name, params, metrics, "Traditional (434D)"))
            
            name, params, metrics = tune_rf(X_trad, y_trad)
            results_list.append(format_results(name, params, metrics, "Traditional (434D)"))
        else:
            logging.error(f"错误：找不到特征文件 {traditional_out}。请先运行 --step feature")
            
        # === 深度组: XGBoost ===
        if os.path.exists(esm_out):
            logging.info("Loading ESM-2 features dataset (640 dims)...")
            df_esm = pd.read_csv(esm_out)
            y_esm = df_esm['label'].values
            
            # 排除非特征列
            feature_cols_esm = [c for c in df_esm.columns if c not in ['sequence', 'label']]
            X_esm = df_esm[feature_cols_esm].values
            
            logging.info(f"Deep Group shape for training: X={X_esm.shape}, y={y_esm.shape}")
            
            # 深度组仅训练 XGBoost，不使用 Z-score 标准化
            name, params, metrics = tune_xgboost(X_esm, y_esm)
            results_list.append(format_results(name, params, metrics, "Deep (ESM-2 640D)"))
        else:
            logging.error(f"错误：找不到特征文件 {esm_out}。请先运行 --step feature")
            
        if results_list:
            results_df = pd.DataFrame(results_list)
            
            # 打印对比结果表
            print("\n" + "="*100)
            print("              论文 3.3 节：模型调优与对比实验结果 (5-Fold CV)")
            print("="*100)
            print(results_df.to_string(index=False))
            print("="*100 + "\n")
            
            # 保存结果为 CSV
            results_csv = os.path.join(PROCESSED_DATA_DIR, "optimized_results.csv")
            results_df.to_csv(results_csv, index=False)
            logging.info(f"模型调优与评估结果已汇总保存至: {results_csv}")

    # --- Step 3.5: ESM-2 LoRA 端到端微调 ---
    if args.step in ['all', 'lora']:
        logging.info("--- Step 3.5: ESM-2 LoRA End-to-End Finetuning ---")
        if not os.path.exists(final_dataset_csv):
            logging.error(f"错误：找不到数据集 {final_dataset_csv}。请先运行 --step preprocess")
            return
            
        logging.info("Starting ESM-2 LoRA 5-Fold Cross Validation...")
        # 传入 final_dataset.csv
        name, params, metrics = run_lora_finetuning(
            dataset_path=final_dataset_csv,
            output_dir=os.path.join(PROCESSED_DATA_DIR, "lora_weights")
        )
        
        # 格式化结果
        lora_result = format_results(name, params, metrics, "Deep (ESM-2 640D)")
        lora_df = pd.DataFrame([lora_result])
        
        results_csv = os.path.join(PROCESSED_DATA_DIR, "optimized_results.csv")
        
        # 追加保存，不要覆盖传统跑分
        if os.path.exists(results_csv):
            existing_df = pd.read_csv(results_csv)
            # 检查是否已经跑过该模型，如果跑过可以替换或直接 append，这里采用直接 append
            updated_df = pd.concat([existing_df, lora_df], ignore_index=True)
        else:
            updated_df = lora_df
            
        updated_df.to_csv(results_csv, index=False)
        logging.info(f"LoRA 微调结果已追加保存至: {results_csv}")
        
        # 打印最新的汇总表格
        print("\n" + "="*100)
        print("              论文 3.3.3 节：LoRA 微调与基线模型对比结果 (5-Fold CV)")
        print("="*100)
        print(updated_df.to_string(index=False))
        print("="*100 + "\n")

    # --- Step 3.6: Baseline 1 (GRUATTNet) ---
    if args.step in ['all', 'b1']:
        logging.info("--- Step 3.6: Baseline 1 (GRUATTNet) Finetuning ---")
        if not os.path.exists(final_dataset_csv):
            logging.error(f"错误：找不到数据集 {final_dataset_csv}。请先运行 --step preprocess")
            return
            
        logging.info("Starting Baseline 1 (GRUATTNet) 5-Fold Cross Validation...")
        name, params, metrics = run_baseline_1(
            dataset_path=final_dataset_csv,
            output_dir=os.path.join(PROCESSED_DATA_DIR, "baseline_1_weights")
        )
        
        b1_result = format_results(name, params, metrics, "Deep (GRUATTNet)")
        b1_df = pd.DataFrame([b1_result])
        
        results_csv = os.path.join(PROCESSED_DATA_DIR, "optimized_results.csv")
        if os.path.exists(results_csv):
            updated_df = pd.concat([pd.read_csv(results_csv), b1_df], ignore_index=True)
        else:
            updated_df = b1_df
            
        updated_df.to_csv(results_csv, index=False)
        logging.info(f"Baseline 1 结果已追加保存至: {results_csv}")
        
        print("\n" + "="*100)
        print("              Baseline 1 (GRUATTNet) 对比结果 (5-Fold CV)")
        print("="*100)
        print(updated_df.to_string(index=False))
        print("="*100 + "\n")

    # --- Step 3.7: Baseline 2 (PeptideNet) ---
    if args.step in ['all', 'b2']:
        logging.info("--- Step 3.7: Baseline 2 (PeptideNet) Finetuning ---")
        if not os.path.exists(final_dataset_csv):
            logging.error(f"错误：找不到数据集 {final_dataset_csv}。请先运行 --step preprocess")
            return
            
        logging.info("Starting Baseline 2 (PeptideNet) 5-Fold Cross Validation...")
        name, params, metrics = run_baseline_2(
            dataset_path=final_dataset_csv,
            output_dir=os.path.join(PROCESSED_DATA_DIR, "baseline_2_weights")
        )
        
        b2_result = format_results(name, params, metrics, "Deep (PeptideNet)")
        b2_df = pd.DataFrame([b2_result])
        
        results_csv = os.path.join(PROCESSED_DATA_DIR, "optimized_results.csv")
        if os.path.exists(results_csv):
            updated_df = pd.concat([pd.read_csv(results_csv), b2_df], ignore_index=True)
        else:
            updated_df = b2_df
            
        updated_df.to_csv(results_csv, index=False)
        logging.info(f"Baseline 2 结果已追加保存至: {results_csv}")
        
        print("\n" + "="*100)
        print("              Baseline 2 (PeptideNet) 对比结果 (5-Fold CV)")
        print("="*100)
        print(updated_df.to_string(index=False))
        print("="*100 + "\n")

    # --- Step 3.8: Baseline 3 (UniDL4BioPep) ---
    if args.step in ['all', 'b3']:
        logging.info("--- Step 3.8: Baseline 3 (UniDL4BioPep) Finetuning ---")
        if not os.path.exists(final_dataset_csv):
            logging.error(f"错误：找不到数据集 {final_dataset_csv}。请先运行 --step preprocess")
            return
            
        logging.info("Starting Baseline 3 (UniDL4BioPep) 5-Fold Cross Validation...")
        name, params, metrics = run_baseline_3(
            dataset_path=final_dataset_csv,
            output_dir=os.path.join(PROCESSED_DATA_DIR, "baseline_3_weights")
        )
        
        b3_result = format_results(name, params, metrics, "Deep (UniDL4BioPep)")
        b3_df = pd.DataFrame([b3_result])
        
        results_csv = os.path.join(PROCESSED_DATA_DIR, "optimized_results.csv")
        if os.path.exists(results_csv):
            updated_df = pd.concat([pd.read_csv(results_csv), b3_df], ignore_index=True)
        else:
            updated_df = b3_df
            
        updated_df.to_csv(results_csv, index=False)
        logging.info(f"Baseline 3 结果已追加保存至: {results_csv}")
        
        print("\n" + "="*100)
        print("              Baseline 3 (UniDL4BioPep) 对比结果 (5-Fold CV)")
        print("="*100)
        print(updated_df.to_string(index=False))
        print("="*100 + "\n")

    # --- Step 3.9: ESM-2 LoRA Rank 参数消融实验 ---
    if args.step in ['all', 'ablation']:
        logging.info("--- Step 3.9: ESM-2 LoRA Rank Ablation ---")
        if not os.path.exists(final_dataset_csv):
            logging.error(f"错误：找不到数据集 {final_dataset_csv}。请先运行 --step preprocess")
            return
            
        from src.models.esm_lora_finetuning import run_rank_ablation
        from src.evaluation.visualization import plot_ablation_results
        
        logging.info("Starting ESM-2 LoRA Rank Ablation Study...")
        run_rank_ablation(final_dataset_csv)
        plot_ablation_results()

    # --- Step 4: 评估与可视化 (待后续完善) ---
    if args.step in ['all', 'evaluate']:
        logging.info("--- Step 4: Evaluation & Visualization ---")
        from src.evaluation.visualization import plot_metrics
        csv_path = os.path.join(PROCESSED_DATA_DIR, "optimized_results.csv")
        plot_metrics(csv_path=csv_path, output_dir=PROCESSED_DATA_DIR)
        logging.info("图表已成功生成")

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()