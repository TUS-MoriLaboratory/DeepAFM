import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

from utils.plot_style import set_plot_style

def plot_accuracy_vs_tolerance(
        results, 
        save_path=None,
        as_percentage=False,
        ):
    """
    Plot Accuracy vs Tolerance (+-n) graph.

    Args:
        results (list of dict): Return value of calc_accuracy_vs_tolerance
        save_path (str, optional): Path to save the plot image
    """
    set_plot_style() 

    plt.figure(figsize=(8, 6))

    for res in results:

        if as_percentage:
            # trans to %
            res["accuracies"] = [
                acc * 100 
                for acc in res["accuracies"]
                ]

        plt.plot(
            res["tolerances"], 
            res["accuracies"], 
            marker='o', 
            label=res["name"],
            linewidth=2,
            markersize=6
        )

    plt.xlabel('Tolerance ($\pm n$)', fontsize=14)
    if as_percentage:
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.ylim(0, 105) 
    else:
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1.05) 
    #plt.title('Accuracy vs. Tolerance Level', fontsize=16)
    
    max_n = max(max(r["tolerances"]) for r in results)
    plt.xticks(range(max_n + 1))
    plt.grid()
    plt.legend(fontsize=12)    
    plt.legend(ncol=2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Visualizer] Saved graph to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_accuracy_vs_s_by_w(results, save_path=None):
    """
    Plot Accuracy vs Scale Parameter s for each Range Parameter w.
    X-axis: s (Variance Scale) - placed categorically
    Y-axis: Accuracy (at tolerance n=0)
    Series: w (Range Scale) - displayed as grouped bar charts

    Args:
        results: calc_accuracy_vs_tolerance の戻り値のリスト
        save_path: 保存先パス
    """
    set_plot_style()

    data_map = {} # (s, w) -> acc
    
    s_values = set()
    w_values = set()

    for res in results:
        name_parts = res["name"].split(", ")
        s_val = float(name_parts[0].split("=")[1])
        w_val = float(name_parts[1].split("=")[1])
        
        # Accuracy at tolerance n=0
        acc = res["accuracies"][0]
        
        data_map[(s_val, w_val)] = acc
        s_values.add(s_val)
        w_values.add(w_val)

    unique_s = sorted(list(s_values))
    unique_w = sorted(list(w_values))

    # --- 2. Create colormap (DarkRed -> Red -> Pink) ---
    colors = ['darkred', "red", 'tomato']
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_reds", colors, N=len(unique_w))
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_w)-1)

    # --- 3. Prepare plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar graph settings
    num_s = len(unique_s)
    num_w = len(unique_w)
    
    # x-axis indices (0, 1, 2, ...) -> fixed equal spacing
    x_indices = np.arange(num_s)
    
    # Bar width settings
    total_width = 0.6
    bar_width = total_width / num_w

    # --- 4. Draw bar graphs ---
    for i, w in enumerate(unique_w):

        accs = [data_map.get((s, w), 0.0) for s in unique_s]
        
        offset = (i - (num_w - 1) / 2) * bar_width
        
        # Get color
        color = cmap(norm(i))
        
        bars = ax.bar(
            x_indices + offset, 
            accs, 
            width=bar_width, 
            color=color,
            linewidth=0.3,
            zorder=3
        )

    # --- 5. Decoration ---
    ax.set_xlabel("Scale Parameter s", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)

    # Replace x-axis labels with s values
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(int(s)) for s in unique_s], fontsize=22)
    
    ax.set_ylim(0, 1.05)

    # Legend
    #ax.legend(title="Range ($w$)", fontsize=10, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Visualizer] Saved bar chart to {save_path}")
    else:
        plt.show()
    
    plt.close()