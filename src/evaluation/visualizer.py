"""
visualizer.py
-----------------------------------
General visualization utilities

Main features:
 - Simultaneous visualization of input images and attention rollout
 - Supports both overlay and side-by-side display
-----------------------------------
"""

import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

from visualization.colormap import array_to_rgb, create_afmhot_cmap
from utils.image_processing import center_crop_or_pad
from utils.plot_style import set_plot_style
from utils.metric_analysis import calc_accuracy_vs_tolerance

from visualization.plot import plot_heightmap, plot_three_heightmaps, plot_three_heightmap_with_lineprofile, plot_multitask_comprehensive
from visualization.metric_analysis import plot_accuracy_vs_tolerance, plot_accuracy_vs_s_by_w

def visualize_confusion_matrix(cm, pdf_path=None):
    """
    Visualize confusion matrix using seaborn heatmap.
    cm: Confusion matrix (2D array-like)
    """
    set_plot_style()

    fig, ax = plt.subplots()
    # confusion matrix heatmap
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(0, cm.shape[1]))
    ax.set_yticks(np.arange(0, cm.shape[0]))
    ax.set_xticklabels(np.arange(1, cm.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, cm.shape[0] + 1))

    plt.xlabel('Predicted Label', fontname='Times New Roman', fontsize=12)
    plt.ylabel('True Label', fontname='Times New Roman', fontsize=12)
    
    norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())
    cmap = plt.get_cmap('Blues')

    # text annotations
    def fmt_abbrev(v):
        v = int(v)
        if v >= 1_000_000:
            return f"{v // 1_000_000}M"
        if v >= 1_000:
            return f"{v // 1_000}K"
        return str(v)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            value_str = fmt_abbrev(value)
            # get color for the cell background
            color = cmap(norm(value))
            
            # Determine text color based on brightness (using RGB values)
            text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'

            # Draw text
            ax.text(j, i, value_str, ha='center', va='center',
                    fontsize=6, color=text_color,
                    fontname='Times New Roman')
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax_cb = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cax, cax=cax_cb)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: fmt_abbrev(x)))
    cbar.ax.tick_params(labelsize=12)

    if pdf_path is not None:
        plt.savefig(pdf_path, dpi=800)
        plt.close(fig)
    else:
        plt.show()

def visualize_denoising(
        input_img, 
        denoised_img, 
        target_img, 
        texts=None, 
        start=None, 
        angle_deg=None,
        num_points=None,
        save_path=None,
        scale=None,
        overlay_line=False,
        pt_path=None
        ):
    """
    Visualize denoising results: input, denoised, and target images side by
    side.
    Args:
        input_img (Tensor): Input AFM image tensor (C, H, W).
        denoised_img (Tensor): Denoised AFM image tensor (C, H, W).
        target_img (Tensor): Target AFM image tensor (C, H, W).
        texts (list of str, optional): Text annotations for each subplot.
        start (tuple, optional): Start coordinates for line profile (x, y).
        angle_deg (float, optional): Angle in degrees for line profile.
        num_points (int, optional): Number of points for line profile.
        save_path (str, optional): Path to save the visualization image.
        scale (float, optional): Scale factor for heightmap color mapping.
        pt_path (str, optional): Path to save the tensor image. If provided, data will be loaded from this .pt file.
    """
    # Visualize input, denoised, and target images side by side.
    if pt_path is not None:
        data = torch.load(pt_path, map_location='cpu')
        
        # load data from .pt file
        input_img    = data["input"]
        denoised_img = data["denoised"]
        target_img   = data["ideal"]
        
        # metadata extraction 
        meta = data.get("meta", {})
        pred_label = meta.get("pred_state", None) # 1-based
        true_label = meta.get("true_state", None) # 1-based

        # prepare texts (states are 1-based)
        texts = [
            None, 
            f'state:{int(pred_label)}' if pred_label is not None else None, 
            f'state:{int(true_label)}' if true_label is not None else None
            ]

    # === data processing ===
    def to_numpy(img):
        if img is None: return None
        if isinstance(img, torch.Tensor):
            if img.ndim == 3: img = img.squeeze(0)
            return img.detach().cpu().numpy()
        return img

    arr_list = [
        to_numpy(input_img),
        to_numpy(denoised_img),
        to_numpy(target_img)
    ]

    if start is not None and angle_deg is not None and num_points is not None:
        plot_three_heightmap_with_lineprofile(
            arr_list=arr_list,
            start=start,
            angle_deg=angle_deg,
            scale=scale,
            colorbar=True,
            texts=texts,
            num_points=num_points,
            overlay_line=overlay_line,
            save_path=save_path,
        )

    else:
        plot_three_heightmaps(
            arr_list=arr_list,
            texts=texts,
            save_path=save_path,
            colorbar=True
        )

def denormalize(img_tensor, mean=0.5, std=0.5):
    """
    Convert normalized tensor from (-1, 1) back to [0, 1].
    img_tensor: (C, H, W)
    """
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    img_tensor = img_tensor * std + mean
    return img_tensor.clamp(0, 1)


def visualize_image_with_attention(
    img_tensor,
    attn_map,
    save_path=None,
    alpha=0.7,
    cmap_attn='viridis'
):
    """
    Display or save a single input image alongside its attention rollout.

    Args:
        img_tensor (Tensor): Input image tensor already descaled with shape (C, H, W).
        attn_map (Tensor): Attention rollout map (grid, grid).
        label (Optional[int]): Optional label for title display.
        save_path (Optional[str]): Path to save the visualization image.
        overlay (bool): If True, overlay the heatmap on the image; otherwise show side-by-side.
        cmap (str): Colormap for attention visualization.
        alpha (float): Transparency used when overlaying the heatmap.
    """

    img = img_tensor.cpu().numpy()
    if img.shape[0] == 1:
        img = img.squeeze(0)  # (H, W)

    # Resize attention map to the image size
    attn_resized = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=img.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu()

    vmin, vmax = img.min(), img.max()
    cmap_afm, norm_afm = create_afmhot_cmap(vmin, vmax)
    rgb_img = array_to_rgb(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    # === Original Image ===
    im1 = ax1.imshow(rgb_img, origin="upper")
    ax1.axis("off")

    # === Attention Overlay ===
    ax2.imshow(rgb_img, origin="upper")
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    im2 = ax2.imshow(attn_norm, cmap=cmap_attn, alpha=alpha, origin="upper")
    ax2.axis("off")

    # --- Colorbar (Optional) ---
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    im_dummy = ax1.imshow(img, cmap=cmap_afm, norm=norm_afm, alpha=0)
    
    cbar = fig.colorbar(im_dummy, cax=cax, orientation='vertical')
    cbar.set_label('Height (nm)', fontsize=22)    
    cbar.solids.set_alpha(1)
    cbar.ax.tick_params(labelsize=20)

    # --- Attention Colorbar ---
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    
    cbar2.set_label('Attention Intensity', fontsize=22)
    cbar2.ax.tick_params(labelsize=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def visualize_multitask_comprehensive(
        input_img, 
        denoised_img, 
        target_img,
        attn_map, 
        class_logits,
        true_label,
        start,
        angle_deg,
        texts=None, 
        save_path=None,
        num_points=None,
        overlay_line=True,
        pt_path=None
        ):
    """
    Comprehensive visualization for Multi-Task AutoEncoder:

    Args:
        input_img (Tensor): Input AFM image tensor (C, H, W).
        denoised_img (Tensor): Denoised AFM image tensor (C, H, W).
        target_img (Tensor): Target AFM image tensor (C, H, W).
        attn_map (Tensor): Attention rollout map (grid, grid).
        class_logits (Tensor): Classification logits tensor (num_classes,).
        true_label (int): True class label.
        start (tuple): Start coordinates for line profile (x, y).
        angle_deg (float): Angle in degrees for line profile.
        texts (list of str, optional): Text annotations for each subplot.
        save_path (str, optional): Path to save the visualization image.
        num_points (int, optional): Number of points for line profile.
        overlay_line (bool): Whether to overlay the line profile on the images.
        pt_path (str, optional): Path to a .pt file containing the above data, which will be prioritized if provided.

    """
    
    # Visualize input, denoised, and target images side by side.
    if pt_path is not None:
        data = torch.load(pt_path, map_location='cpu')
        
        # load data from .pt file
        input_img    = data["input"]
        denoised_img = data["denoised"]
        target_img   = data["ideal"]
        attn_map     = data["attn"]
        class_logits = data["logits"]
        
        # metadata extraction 
        meta = data.get("meta", {})
        pred_label = meta.get("pred_state", pred_label) # 1-based
        true_label = meta.get("true_state", true_label) # 1-based

        # prepare texts (states are 1-based)
        texts = [
            None, 
            f'state:{int(pred_label.item())}' if pred_label is not None else None, 
            f'state:{int(true_label.item())}' if true_label is not None else None
            ]

        if true_label is not None:
            true_label = true_label - 1 # 1-based -> 0-based (for function input)

        print(f"[Visualizer] Data loaded from {pt_path}. Prioritizing .pt content.")

    # === data processing ===
    def to_numpy(img):
        if img is None: return None
        if isinstance(img, torch.Tensor):
            if img.ndim == 3: img = img.squeeze(0)
            return img.detach().cpu().numpy()
        return img

    arr_list = [
        to_numpy(input_img),
        to_numpy(denoised_img),
        to_numpy(target_img)
    ]

    plot_multitask_comprehensive(
        arr_list=arr_list,
        attn_map=attn_map,
        class_logits=class_logits,
        true_label=true_label,
        start=start,
        angle_deg=angle_deg,
        texts=texts,
        save_path=save_path,
        colorbar=True,
        num_points=num_points,
        overlay_line=overlay_line,
    )

# =================================
# Distorted Dependency Evaluation
# =================================
def plot_distorted_dependency_accuracy(
    save_dir,
    variance_scale_param_list,      # Distortion noise std list
    range_scale_param_list          # Distortion blur width list
):
    """
    Plot distorted dependency accuracy graph.

    Args:
        results (list of dict): Return value of calc_accuracy_vs_tolerance
        save_dir (str): Directory to save the plot image
        save_filename (str): Filename for the saved plot image
    """
    # Prepare confusion matrix paths and names
    name_list = []
    cm_path_list = []
    for s in variance_scale_param_list:
        for w in range_scale_param_list:
            cm_path = os.path.join(
                save_dir,
                f's{s}_w{w}_results',
                'confusion_matrix.npy'
            )
            cm_path_list.append(cm_path)
            name_list.append(f's={s}, w={w}')

    # Calculate accuracy vs tolerance
    results = calc_accuracy_vs_tolerance(
        cm_path_list=cm_path_list,
        names=name_list
    )
    save_path = os.path.join(
        save_dir, 'distorted_dependency_accuracy_vs_tolerance.pdf'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ACC vs Tolerance plot
    plot_accuracy_vs_tolerance(
        results=results,
        save_path=save_path
    )

    # ACC vs s (for each w) plot
    save_path_dependency = os.path.join(
        save_dir, 'distorted_dependency_s_w.pdf'
    )
    plot_accuracy_vs_s_by_w(
        results=results,
        save_path=save_path_dependency
    )

# =================================
# Structure Dependency Evaluation
# =================================
def plot_entropy_trajectory(csv_path, save_path=None):
    """
    Load CSV and plot graph with structure_id on x-axis and entropy on y-axis.
    """

    set_plot_style()

    print(f"[Visualizer] Plotting entropy from: {csv_path}")

    ids = []
    entropies = []

    # --- Load CSV ---
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            raw_data = []
            
            for row in reader:
                s_id = row['structure_id']
                ent = float(row['entropy'])
                raw_data.append((s_id, ent))
            
            if not raw_data:
                print("Warning: CSV contains no data.")
                return

            def extract_num(text_tuple):
                """
                Extract numeric part from structure_id for sorting.
                """
                s_id = text_tuple[0]
                match = re.search(r'(\d+)', s_id)
                if match:
                    return int(match.group(1))
                return 0 

            # sort (md1 -> md2 -> ... -> md10)
            raw_data.sort(key=extract_num)

            ids = [x[0] for x in raw_data]
            entropies = [x[1] for x in raw_data]

    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return

    plt.figure(figsize=(8, 6)) 
    x_positions = range(1, len(ids) + 1) # x positions for bars

    # --- bar plot ---
    plt.bar(x_positions, entropies, color="#0f52bd", linewidth=0.5, alpha=0.8)
    
    plt.xlabel("Morphing Step", fontsize=20)
    plt.ylabel("Entropy", fontsize=20)
    plt.xticks(x_positions, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # --- save ---
    if save_path is None:
        base, _ = os.path.splitext(csv_path)
        save_path = f"{base}_bar_plot.png"
    
    plt.savefig(save_path, dpi=800)
    plt.close()
    
    print(f"[Visualizer] Saved entropy bar plot to: {save_path}")

def plot_mean_probability_distribution(csv_path, target_ids=None, save_path=None, color_list=None):
    """
    Plot mean prediction probability distribution for specified structure IDs.
    
    Args:
        csv_path (str): Path to the aggregated CSV file
        target_ids (list[str], optional): List of IDs to visualize. Example: ["md0", "md10", "md23"]
                                         If None, all data will be displayed (which may be cluttered if there are many).
        save_path (str, optional): Path to save the plot image.
        color_list (list, optional): List of colors to use for the bars. If None, default colormap will be used.
    """
    print(f"[Visualizer] Plotting probability distribution from: {csv_path}")

    # Data storage: { "structure_id": [prob_0, prob_1, ...] }
    plot_data = {}
    class_indices = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Identify mean_prob_ columns
            mean_cols = [c for c in fieldnames if c.startswith("mean_prob_")]
            mean_cols.sort(key=lambda x: int(x.split('_')[-1]))
        
            std_cols = [c for c in fieldnames if c.startswith("std_prob_")]
            std_cols.sort(key=lambda x: int(x.split('_')[-1]))

            if not mean_cols:
                print("Error: No 'mean_prob_*' columns found in CSV.")
                return
            
            num_classes = len(mean_cols)
            class_indices = range(num_classes)

            # Load data
            for row in reader:
                s_id = row['structure_id']
                
                # If target_ids is specified, extract only those included
                # If not specified, use all data
                if target_ids is not None and s_id not in target_ids:
                    continue
                
                # Retrieve probability values
                try:
                    means = [float(row[c]) for c in mean_cols]
                    stds = [float(row[c]) for c in std_cols] if std_cols else [0.0]*len(means)
                    
                    plot_data[s_id] = {"mean": means, "std": stds}
                except ValueError:
                    continue

    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return

    if not plot_data:
        print(f"Warning: No matching data found for IDs: {target_ids}")
        return

    # --- Prepare plot ---
    # To maintain the order of target_ids, retrieve data in the order of target_ids instead of keys()
    # (If None, use the order of reading)
    if target_ids is None:
        sorted_ids = list(plot_data.keys())
    else:
        # Use only those that exist in the data
        sorted_ids = [tid for tid in target_ids if tid in plot_data]

    num_groups = len(sorted_ids)
    if num_groups == 0:
        return

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate bar positions
    total_width = 0.8  # Total width
    bar_width = total_width / num_groups
    
    # Color map (change color for each ID)
    # Get colors from viridis, coolwarm, etc.
    if color_list is not None:
        colors = color_list
    else:
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_groups)) 

    # Loop through IDs and plot
    for i, s_id in enumerate(sorted_ids):
        data = plot_data[s_id]
        means = data["mean"]
        stds = data["std"]
        
        # Calculate X positions: center on class ID and shift left/right
        # Offset from center: (i - (N-1)/2) * width
        offset = (i - (num_groups - 1) / 2) * bar_width
        x_positions = [x + offset for x in class_indices]
        
        # Adjust structure_id for display (e.g., md0 -> md1)
        s_id_num = re.findall(r'\d+', s_id)
        if s_id_num:
            s_id = f"md{int(s_id_num[0])+1}"

        ax.bar(
            x_positions, 
            means, 
            yerr=stds,       
            capsize=4,       
            width=bar_width, 
            label=f"ID: {s_id}", 
            color=colors[i], 
            linewidth=0.5,
            alpha=0.9
        )

    # --- Decoration ---
    ax.set_xlabel("Protein State", fontsize=22)
    ax.set_ylabel("Probability", fontsize=22)
    ax.set_xticks(class_indices)
    ax.set_xticklabels([str(c+1) for c in class_indices], fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, 1.05)
    
    # Legend
    ax.legend(title="Structure ID", fontsize=12)

    plt.tight_layout()

    # --- Save ---
    if save_path is None:
        base, _ = os.path.splitext(csv_path)
        # If IDs are specified, reflect them in the filename for clarity
        suffix = "_dist_plot.png"
        if target_ids and len(target_ids) <= 3:
            suffix = f"_dist_{'_'.join(target_ids)}.png"
        save_path = f"{base}{suffix}"
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[Visualizer] Saved probability distribution plot to: {save_path}")