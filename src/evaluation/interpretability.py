import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from transformers import AutoModel, AutoTokenizer

def plot_tsne():
    """
    Reads ESM-2 features, applies t-SNE for dimensionality reduction,
    and plots the result.
    """
    file_path = "data/processed/features_esm2.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Ensure features have been generated.")
        return
    
    print("Loading features...")
    df = pd.read_csv(file_path)
    
    if 'label' not in df.columns:
        print(f"'label' column not found in {file_path}")
        return
        
    labels = df['label']
    
    # Extract feature columns (usually all numeric columns except label and typical identifiers)
    features = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    
    for non_feature_col in ['ID', 'id', 'Sequence', 'sequence']:
        if non_feature_col in features.columns:
            features = features.drop(columns=[non_feature_col])
            
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Prepare DataFrame for plotting
    plot_df = pd.DataFrame({
        't-SNE 1': features_2d[:, 0],
        't-SNE 2': features_2d[:, 1],
        'label': labels.map({1: 'AVP', 0: 'AMP'})
    })
    
    # Define colors
    palette = {'AVP': 'darkred', 'AMP': 'darkblue'}
    
    print("Plotting t-SNE...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x='t-SNE 1',
        y='t-SNE 2',
        hue='label',
        palette=palette,
        alpha=0.7,
        s=50
    )
    
    plt.title("t-SNE Visualization of ESM-2 Representations", fontsize=16)
    plt.legend(title='Peptide Type')
    plt.tight_layout()
    
    out_path = "data/processed/tsne_plot.png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"t-SNE plot successfully saved to {out_path}\n")


def plot_attention():
    """
    Loads ESM-2 model, performs forward pass on a sample sequence,
    and visualizes the attention weights of the last layer's first head.
    """
    model_name = "facebook/esm2_t30_150M_UR50D"
    print(f"Loading tokenizer and model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # output_attentions=True is required to get attention weights
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    sequence = "GLFDIVKKIAGHIAGSI"
    print(f"Performing inference on sequence: {sequence}")
    inputs = tokenizer(sequence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get attentions: tuple of length num_layers
    # Each layer has shape (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    # Extract last layer (-1), first attention head (0), first batch item (0)
    last_layer_attn = attentions[-1]
    first_head_attn = last_layer_attn[0, 0, :, :].numpy()
    
    # Convert token IDs back to string tokens to use as labels
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Remove special tokens <cls> (index 0) and <eos> (index -1)
    attn_matrix = first_head_attn[1:-1, 1:-1]
    amino_acids = tokens[1:-1]
    
    print("Plotting attention heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix,
        xticklabels=amino_acids,
        yticklabels=amino_acids,
        cmap="Blues",
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title("Attention Heatmap (Last Layer, First Head)", fontsize=16)
    plt.xlabel("Key", fontsize=12)
    plt.ylabel("Query", fontsize=12)
    
    # Fix potential layout issues with ticks
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_path = "data/processed/attention_heatmap.png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Attention heatmap successfully saved to {out_path}\n")


if __name__ == "__main__":
    plot_tsne()
    plot_attention()
