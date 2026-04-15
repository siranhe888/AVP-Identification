"""
提取传统机器学习所需的理化特征 (AAC, Dipeptide, etc.)
"""

import pandas as pd
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def extract_aac(sequence: str) -> dict:
    """
    提取氨基酸组成 (Amino Acid Composition, AAC) 特征
    
    Args:
        sequence (str): 氨基酸序列
        
    Returns:
        dict: 20 种标准氨基酸的组成比例
    """
    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_len = len(sequence)
    aac = {aa: sequence.count(aa) / seq_len for aa in valid_amino_acids}
    return aac

def extract_dpc(sequence: str) -> dict:
    """
    提取二肽组成 (Dipeptide Composition, DPC) 特征 (400维)
    
    Args:
        sequence (str): 氨基酸序列
        
    Returns:
        dict: 400 种二肽组合的组成比例
    """
    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    # 初始化 400 维字典
    dpc = {f"DPC_{aa1}{aa2}": 0.0 for aa1 in valid_amino_acids for aa2 in valid_amino_acids}
    
    seq_len = len(sequence)
    if seq_len < 2:
        return dpc
        
    # 计算二肽频率
    dipeptides = [sequence[i:i+2] for i in range(seq_len - 1)]
    total_dipeptides = len(dipeptides)
    
    counts = Counter(dipeptides)
    for dp, count in counts.items():
        key = f"DPC_{dp}"
        if key in dpc:
            dpc[key] = count / total_dipeptides
            
    return dpc

def extract_physicochemical_properties(sequence: str) -> dict:
    """
    提取蛋白质的 14 项核心理化性质
    
    Args:
        sequence (str): 氨基酸序列
        
    Returns:
        dict: 理化性质特征字典
    """
    analyzer = ProteinAnalysis(sequence)
    
    # 1. 脂肪族指数 (Aliphatic Index) 的计算
    # 公式: Aliphatic index = X(Ala) + a * X(Val) + b * ( X(Ile) + X(Leu) )
    # 其中 a = 2.9, b = 3.9
    seq_len = len(sequence)
    if seq_len > 0:
        mole_fraction_Ala = sequence.count('A') / seq_len * 100
        mole_fraction_Val = sequence.count('V') / seq_len * 100
        mole_fraction_Ile = sequence.count('I') / seq_len * 100
        mole_fraction_Leu = sequence.count('L') / seq_len * 100
        aliphatic_index = mole_fraction_Ala + 2.9 * mole_fraction_Val + 3.9 * (mole_fraction_Ile + mole_fraction_Leu)
    else:
        aliphatic_index = 0.0

    # 2. 极性 (Polarity) 和 极性相关氨基酸比例
    # Polar amino acids: D, E, H, K, N, Q, R, S, T, Y, C, W (根据一些文献定义，这里以亲水性评估补充)
    # 此处取极性和非极性比例，或者直接使用其他常见极性特征。这里简单实现带电氨基酸与无电荷氨基酸比例
    
    try:
        molecular_weight = analyzer.molecular_weight()
    except ValueError:
        molecular_weight = 0.0
        
    try:
        isoelectric_point = analyzer.isoelectric_point()
    except ValueError:
        isoelectric_point = 0.0

    try:
        gravy = analyzer.gravy()
    except ValueError:
        gravy = 0.0
        
    try:
        instability_index = analyzer.instability_index()
    except ValueError:
        instability_index = 0.0
        
    try:
        aromaticity = analyzer.aromaticity()
    except ValueError:
        aromaticity = 0.0

    # 提取二级结构比例 (螺旋, 折叠, 转角) - 算作 3 项
    try:
        sec_struct = analyzer.secondary_structure_fraction() # (helix, turn, sheet)
        helix, turn, sheet = sec_struct
    except ValueError:
        helix, turn, sheet = 0.0, 0.0, 0.0
        
    # 计算带电荷氨基酸比例 (正电荷和负电荷)
    positive_charge_aa = sum(sequence.count(aa) for aa in ['R', 'K', 'H']) / seq_len if seq_len > 0 else 0
    negative_charge_aa = sum(sequence.count(aa) for aa in ['D', 'E']) / seq_len if seq_len > 0 else 0
    
    # 净电荷数 (Net Charge) 在 pH 7.0 下的简单估算 (正电荷氨基酸数量 - 负电荷氨基酸数量)
    net_charge = sum(sequence.count(aa) for aa in ['R', 'K']) - sum(sequence.count(aa) for aa in ['D', 'E'])

    # 提取摩尔消光系数 (Molar Extinction Coefficient) - 返回两个值（还原型和氧化型），取平均
    try:
        extinction_coef = analyzer.molar_extinction_coefficient()
        extinction = (extinction_coef[0] + extinction_coef[1]) / 2.0
    except ValueError:
        extinction = 0.0
        
    # 极性氨基酸比例 (Polarity approximation)
    polar_aa = set("DERKQNSTHYCW")
    polarity = sum(1 for aa in sequence if aa in polar_aa) / seq_len if seq_len > 0 else 0.0

    features = {
        "molecular_weight": molecular_weight,              # 1. 分子量
        "isoelectric_point": isoelectric_point,            # 2. 等电点
        "aromaticity": aromaticity,                        # 3. 芳香性
        "instability_index": instability_index,            # 4. 不稳定性指数
        "gravy": gravy,                                    # 5. 总体疏水性 (GRAVY)
        "aliphatic_index": aliphatic_index,                # 6. 脂肪族指数
        "net_charge": net_charge,                          # 7. 净电荷数
        "positive_charge_freq": positive_charge_aa,        # 8. 正电荷氨基酸比例
        "negative_charge_freq": negative_charge_aa,        # 9. 负电荷氨基酸比例
        "polarity": polarity,                              # 10. 极性比例
        "extinction_coefficient": extinction,              # 11. 摩尔消光系数
        "helix_fraction": helix,                           # 12. 螺旋比例
        "turn_fraction": turn,                             # 13. 转角比例
        "sheet_fraction": sheet                            # 14. 折叠比例
    }
    return features

def generate_traditional_features(data_df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    """
    为数据集生成所有传统特征 (AAC + DPC + 理化特征)
    
    Args:
        data_df (pd.DataFrame): 包含序列数据的 DataFrame
        sequence_col (str): 序列所在的列名
        
    Returns:
        pd.DataFrame: 包含提取特征的数据集
    """
    features_list = []
    for seq in data_df[sequence_col]:
        aac = extract_aac(seq)
        dpc = extract_dpc(seq)
        physicochemical = extract_physicochemical_properties(seq)
        # 合并所有特征 (20 + 400 + 14 = 434 维)
        combined = {**aac, **dpc, **physicochemical}
        features_list.append(combined)
        
    features_df = pd.DataFrame(features_list)
    return pd.concat([data_df, features_df], axis=1)
