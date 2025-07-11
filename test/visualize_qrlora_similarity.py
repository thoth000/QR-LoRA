import torch
import safetensors.torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from torch.nn.functional import cosine_similarity
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the similarity between two DeltaR-LoRA weight files')
    parser.add_argument('--lora1_path', type=str, required=True,
                        help='Path to the first LoRA weight file')
    parser.add_argument('--lora2_path', type=str, required=True,
                        help='Path to the second LoRA weight file')
    parser.add_argument('--lora1_name', type=str, default="style",
                        help='Name of the first LoRA')
    parser.add_argument('--lora2_name', type=str, default="content",
                        help='Name of the second LoRA')
    parser.add_argument('--output_dir', type=str, default="deltaR_similarity_analysis",
                        help='Output directory')
    parser.add_argument('--fixed_scale', action='store_true',
                        help='Whether to fix the similarity axis range to 0-1')
    return parser.parse_args()

def load_lora_weights(path: str) -> dict:
    """Load LoRA weight file"""
    if path.endswith('.safetensors'):
        return safetensors.torch.load_file(path)
    return torch.load(path)

def get_matrices_by_type(state_dict: dict, matrix_type: str) -> dict:
    """Extract matrices of the specified type
    matrix_type can be:
    - 'Q': Q matrix (lora.q.weight)
    - 'deltaR': deltaR matrix (lora.delta_r.weight)
    - 'baseR': base R matrix (lora.base_r.weight)
    """
    matrices = {}
    type_patterns = {
        'Q': 'lora.q.weight',
        'deltaR': 'lora.delta_r.weight',
        'baseR': 'lora.base_r.weight'
    }
    pattern = type_patterns[matrix_type]
    
    for key in state_dict.keys():
        if pattern in key:
            matrices[key] = state_dict[key]
    return matrices

def calculate_cosine_similarities(m1_matrices: dict, m2_matrices: dict) -> dict:
    """Calculate cosine similarity between two sets of matrices"""
    similarities = {}
    for key in m1_matrices.keys():
        if key in m2_matrices:
            m1_flat = m1_matrices[key].view(-1)
            m2_flat = m2_matrices[key].view(-1)
            sim = cosine_similarity(m1_flat, m2_flat, dim=0)
            similarities[key] = sim.item()
    return similarities

def plot_similarities(similarities: dict, output_dir: str, lora1_name: str, 
                     lora2_name: str, matrix_type: str, fixed_scale: bool = False):
    """Plot similarity visualization"""
    plt.rcParams['font.size'] = 12
    
    layer_names = list(similarities.keys())
    sim_values = list(similarities.values())
    layer_indices = list(range(len(sim_values)))
    
    # 1. Line plot
    plt.figure(figsize=(24, 6))
    plt.plot(layer_indices, sim_values, marker='o', linewidth=1.5, markersize=3)
    plt.title(f'Cosine Similarity of {matrix_type} matrices\n({lora1_name} vs {lora2_name})', 
             fontsize=14, pad=15)
    
    step = max(len(layer_indices) // 20, 1)
    plt.xticks(layer_indices[::step], layer_indices[::step], rotation=45)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    
    if fixed_scale:
        plt.ylim(-0.1, 1.1)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    scale_type = "fixed" if fixed_scale else "auto"
    plt.savefig(os.path.join(output_dir, f'{matrix_type}_similarity_{scale_type}_scale_line.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(sim_values, bins=20)
    plt.title(f'Distribution of {matrix_type} Similarity\n({lora1_name} vs {lora2_name})', 
             fontsize=14, pad=15)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    if fixed_scale:
        plt.xlim(-0.1, 1.1)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{matrix_type}_similarity_{scale_type}_scale_hist.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    stats = {
        "mean": float(np.mean(sim_values)),
        "std": float(np.std(sim_values)),
        "min": float(np.min(sim_values)),
        "max": float(np.max(sim_values)),
        "similarities": {k: float(v) for k, v in similarities.items()}
    }
    
    with open(os.path.join(output_dir, f'{matrix_type}_similarities.json'), 'w') as f:
        json.dump(stats, f, indent=4)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load weights
    lora1_weights = load_lora_weights(args.lora1_path)
    lora2_weights = load_lora_weights(args.lora2_path)
    
    # Analyze all three types of matrices
    matrix_types = ['Q', 'deltaR', 'baseR']
    
    all_stats = {}
    for matrix_type in matrix_types:
        # Extract matrices
        m1_matrices = get_matrices_by_type(lora1_weights, matrix_type)
        m2_matrices = get_matrices_by_type(lora2_weights, matrix_type)
        
        # Calculate similarity
        similarities = calculate_cosine_similarities(m1_matrices, m2_matrices)
        
        # Generate visualization
        plot_similarities(similarities, args.output_dir, args.lora1_name, args.lora2_name, 
                         matrix_type, args.fixed_scale)
        
        # Calculate statistics
        sim_values = list(similarities.values())
        stats = {
            "mean": np.mean(sim_values),
            "std": np.std(sim_values),
            "min": np.min(sim_values),
            "max": np.max(sim_values)
        }
        all_stats[matrix_type] = stats
        
        print(f"\n{matrix_type} matrix analysis result ({args.lora1_name} vs {args.lora2_name}):")
        print(f"Mean similarity: {stats['mean']:.4f}")
        print(f"Max similarity: {stats['max']:.4f}")
        print(f"Min similarity: {stats['min']:.4f}")
        print(f"Similarity standard deviation: {stats['std']:.4f}")
    
    # Save overall statistics
    with open(os.path.join(args.output_dir, 'all_statistics.json'), 'w') as f:
        json.dump({k: {m: float(v) for m, v in s.items()} 
                  for k, s in all_stats.items()}, f, indent=4)

if __name__ == "__main__":
    main() 