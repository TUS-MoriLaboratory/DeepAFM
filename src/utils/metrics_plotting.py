import matplotlib.pyplot as plt
import numpy as np
import csv
import os

from utils.plot_style import set_plot_style

def plot_metric_histograms(save_dir, mse, rmse, mae, cc):
    # Set plot style
    set_plot_style(font_size=14)

    os.makedirs(save_dir, exist_ok=True)
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "CC": cc
    }

    for name, arr in metrics.items():
        plt.figure(figsize=(5,4))
        plt.hist(arr, bins=30, alpha=0.7)
        plt.title(f"{name} distribution")
        plt.xlabel(name)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{name}_hist.pdf"), 
            format='pdf', 
            dpi=300
        )
        plt.close()

def plot_parameter_sensitivity(csv_path, save_dir):
    """
    CSVを読み込み、各パラメータがCCに与える影響を棒グラフで可視化する。
    Pandas不使用。
    """
    print(f"Analyzing parameter sensitivity from: {csv_path}")

    # 1. データの読み込み
    data_rows = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 数値型に変換して格納
                converted = {k: float(v) for k, v in row.items()}
                data_rows.append(converted)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        return

    if not data_rows:
        print("Error: No data in CSV.")
        return

    target_params = [
        "probe_radius", 
        "half_angle", 
        "scale", 
        "rotation_x", 
        "rotation_y", 
        "rotation_z"
    ]
    
    available_keys = data_rows[0].keys()
    params_to_plot = [p for p in target_params if p in available_keys]

    analysis_results = {}

    for param in params_to_plot:
        grouped_cc = {}
        for row in data_rows:
            val = row[param]
            cc = row['cc']
            
            if val not in grouped_cc:
                grouped_cc[val] = []
            grouped_cc[val].append(cc)
        
        sorted_vals = sorted(grouped_cc.keys())
        means = []
        stds = []
        
        for v in sorted_vals:
            cc_list = np.array(grouped_cc[v])
            means.append(np.mean(cc_list))
            stds.append(np.std(cc_list))
            
        analysis_results[param] = {
            "x": sorted_vals,
            "mean": means,
            "std": stds,
            "sensitivity": max(means) - min(means)
        }

    num_params = len(params_to_plot)
    cols = 3
    rows = (num_params + cols - 1) // cols 

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        res = analysis_results[param]
        
        x_vals = res["x"]
        means = res["mean"]
        stds = res["std"]
        
        x_pos = np.arange(len(x_vals))
        
        ax.bar(x_pos, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.set_title(f"Effect of {param}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Correlation Coefficient (CC)")
        ax.set_xlabel(param)
        
        x_labels = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in x_vals]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45)
        
        min_cc = min(np.array(means) - np.array(stds))
        ax.set_ylim(max(0, min_cc - 0.1), 1.05)
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.text(0.95, 0.95, f"Range: {res['sensitivity']:.3f}", 
                transform=ax.transAxes, ha='right', va='top', 
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "parameter_sensitivity_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved sensitivity plot to: {save_path}")

    print("\n=== Parameter Sensitivity Ranking (Impact on CC) ===")
    ranking = sorted(params_to_plot, key=lambda p: analysis_results[p]["sensitivity"], reverse=True)
    for rank, p in enumerate(ranking, 1):
        print(f"{rank}. {p}: Range = {analysis_results[p]['sensitivity']:.4f}")