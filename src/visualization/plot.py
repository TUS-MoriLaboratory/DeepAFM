import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .colormap import array_to_rgb, create_afmhot_cmap
from utils.plot_style import set_plot_style
from utils.image_processing import center_crop_or_pad
from utils.image_analysis import get_line_profile

# default colorlist for heightmap with line profile
colorlist=["blue", "green", "orange"]

def plot_heightmap(
        arr, 
        ref=None, 
        image_size=None, 
        save_path=None, 
        colorbar=True):

    set_plot_style()

    if image_size is not None:
        arr = center_crop_or_pad(arr, image_size)
        if ref is not None:
            ref = center_crop_or_pad(ref, image_size)

    # ---- colormap ----
    if ref is not None:
        assert arr.shape == ref.shape, "arr and ref must have the same shape."
        vmin = min(arr.min(), ref.min())
        vmax = max(arr.max(), ref.max())
    
    else:
        vmin, vmax = arr.min(), arr.max() 
    
    cmap, norm = create_afmhot_cmap(vmin, vmax)
    rgb = array_to_rgb(arr, vmin, vmax)
    cax = None
    
    if ref is not None:
            
        fig = plt.figure(figsize=(12, 6))

        grid = ImageGrid(fig, 111,  
                    nrows_ncols=(1, 2),
                    axes_pad=0.2,
                    cbar_mode='single' if colorbar else None,
                    cbar_location='right',
                    cbar_pad=0.15,
                    cbar_size="5%" 
                    )
        
        ax1 = grid[0] # reference image
        ax2 = grid[1] # arr image

        rgb_ref = array_to_rgb(ref, vmin, vmax)
        ax1.imshow(rgb_ref, origin="upper")
        ax1.axis("off")

        ax2.imshow(rgb, origin="upper")
        ax2.axis("off")

        # ---- transparent height map for colorbar with correct cmap/norm ----
        im_for_cbar = ax2.imshow(arr, cmap=cmap, norm=norm, alpha=0)

        if colorbar:
            cax = grid.cbar_axes[0]

    else:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis("off")

        im_for_cbar = ax.imshow(arr, cmap=cmap, norm=norm, origin="upper")

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

    # ---- colorbar ----
    if colorbar and cax is not None:
        cbar = fig.colorbar(
            im_for_cbar, 
            cax=cax, 
            orientation='vertical', 
        )
        cbar.set_label('Height (nm)', fontsize=22)
        cbar.solids.set_alpha(1)
        cbar.ax.tick_params(labelsize=20)

    # ---- save ---- 
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_three_heightmaps(
        arr_list, 
        texts=[None, "state: 1", "state: 1"], 
        image_size=None, 
        save_path=None, 
        colorbar=True):
    """
    plot three heightmaps side by side.

    Args:
        arr_list (list of ndarray): 3 arrays(2D) [arr1, arr2, arr3]
        texts (list of str, optional): 3 titles [title1, title2, title3]
        image_size (int, optional): size to crop/pad
        save_path (str, optional): path to save the figure
        colorbar (bool): whether to display colorbar
    """
    
    # Input check
    if len(arr_list) != 3:
        raise ValueError("arr_list must contain exactly 3 arrays.")
    
    set_plot_style()

    # --- Preprocessing ---
    # Adjust image sizes
    if image_size is not None:
        arr_list = [center_crop_or_pad(arr, image_size) for arr in arr_list]

    # --- Colormap Setup ---
    # Get the min and max across all three images to unify the scale
    vmin = min(arr.min() for arr in arr_list)
    vmax = max(arr.max() for arr in arr_list)
    cmap, norm = create_afmhot_cmap(vmin, vmax)

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 6))

    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(1, 3),
                     axes_pad=0.2,
                     cbar_mode='single' if colorbar else None,
                     cbar_location='right',
                     cbar_pad=0.15,
                     cbar_size="5%" 
                     )
    
    # setting texts 
    if texts is None:
        texts = [None, None, None]

    elif len(texts) != 3:
        raise ValueError("texts list must contain exactly 3 strings.")

    # Plot three images in a loop
    for i, ax in enumerate(grid):
        arr = arr_list[i]
        
        # Convert array to AFM-style RGB
        rgb = array_to_rgb(arr, vmin, vmax) # transform to RGB using unified vmin/vmax
        
        # Display
        ax.imshow(rgb, origin="upper")
        ax.axis("off")
        
        # Set text (optional)
        if texts[i]:
            ax.text(
                0.33, 0.12, texts[i], 
                transform=ax.transAxes, 
                color='white',
                fontsize=36,
                fontweight='bold',
                ha='left', va='top', 
            )

    # --- Colorbar ---
    if colorbar:
        im_for_cbar = grid[-1].imshow(arr_list[-1], cmap=cmap, norm=norm, alpha=0)
        
        cbar =fig.colorbar(
            im_for_cbar, 
            cax=grid.cbar_axes[0],
            orientation='vertical',
            )
        cbar.set_label('Height (nm)', fontsize=22)
        cbar.solids.set_alpha(1)
        cbar.ax.tick_params(labelsize=20)

    # --- Save ---
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_heightmap_with_lineprofile(
        img_tensor, 
        start,
        angle_deg,
        image_size=None,
        save_path=None,
        num_points=None
        ):
    """
    Visualize a line profile on an AFM image.

    Args:
        img_tensor (Tensor): AFM image tensor (1, H, W) or (H, W)
        start (tuple): Starting coordinates (y, x)
        angle_deg (float): Angle in degrees
        num_points (int, optional): Number of sampling points along the line
    """

    set_plot_style()
    img = img_tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)  # (H, W) 

    if image_size is not None:
        img = center_crop_or_pad(img, image_size)

    profile_data = get_line_profile(img, start, angle_deg, num_points)

    distances = profile_data["distances"]
    values = profile_data["values"]
    xs, ys = profile_data["coords"]

    # create colormap and rgb image
    vmin, vmax = img.min(), img.max()
    cmap, norm = create_afmhot_cmap(vmin, vmax)
    rgb = array_to_rgb(img, vmin, vmax) 

    wr = [1.0, 1.5]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   gridspec_kw={'width_ratios': wr},
                                   layout='constrained')

    # --- height map and line profile ---
    ax1.imshow(rgb, origin="upper") 

    # plot line
    ax1.plot(xs, ys, linestyle='--', color='white', linewidth=3, alpha=0.8)
    
    # start point marker
    ax1.plot(xs[0], ys[0], marker='o', color='white', markersize=7)
    ax1.axis("off")

    # colorbar
    im_dummy = ax1.imshow(img, cmap=cmap, norm=norm, alpha=0)
    cbar = fig.colorbar(im_dummy, ax=ax1, location='right', shrink=1.0)
    cbar.set_label('Height (nm)', fontsize=24)
    cbar.solids.set_alpha(1)
    cbar.ax.tick_params(labelsize=22)

    # --- line profile ---
    ax2.plot(distances, values, color="orange", linewidth=2.0)
    ax2.set_xlabel("Pixel", fontsize=24, labelpad=10) 
    ax2.set_ylabel("Height (nm)", fontsize=24, labelpad=10)
    ax2.tick_params(axis='both', labelsize=22) 
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # aspect ratio
    img_h, img_w = img.shape
    img_aspect = img_h / img_w

    target_aspect = img_aspect / (wr[1] / wr[0])
    ax2.set_box_aspect(target_aspect)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_three_heightmap_with_lineprofile(
        arr_list, 
        start,
        angle_deg,
        colorlist=colorlist,
        colorbar=True,
        texts=None,
        image_size=None,
        save_path=None,
        num_points=None,
        scale=None, 
        overlay_line=True,
        ):
    """
    Visualize a line profile on an AFM image.

    Args:
        arr_list (list of ndarray): 3 arrays(2D) [arr1, arr2, arr3]
        start (tuple): Starting coordinates (y, x)
        angle_deg (float): Angle in degrees
        num_points (int, optional): Number of sampling points along the line
        scale (float, optional): Scale factor (nm/pixel) for converting the x-axis to nanometers
    """
    # Input check
    if len(arr_list) != 3:
        raise ValueError("arr_list must contain exactly 3 arrays.")

    set_plot_style()

    # --- Preprocessing ---
    # Adjust image sizes
    if image_size is not None:
        arr_list = [center_crop_or_pad(arr, image_size) for arr in arr_list]

    for i in range(len(arr_list)):
        img = arr_list[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)  # (H, W) 
        arr_list[i] = img

    # --- Setting texts ---
    if texts is None:
        texts = [None, None, None]

    elif len(texts) != 3:
        raise ValueError("texts list must contain exactly 3 strings.")

    # --- Colormap Setup ---
    # Get the min and max across all three images to unify the scale
    vmin = min(arr.min() for arr in arr_list)
    vmax = max(arr.max() for arr in arr_list)
    cmap, norm = create_afmhot_cmap(vmin, vmax)

    # --- Line Profile Data ---
    profile_data_list = []
    for arr in arr_list:
        profile_data = get_line_profile(arr, start, angle_deg, num_points)
        profile_data_list.append(profile_data)

    wr = [1.0, 1.0, 1.0, 1.5]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(26, 6), 
                                   gridspec_kw={'width_ratios': wr},
                                   layout='constrained')

    for i, ax in enumerate([ax1, ax2, ax3]):
        # get array
        arr = arr_list[i] 

        # get profile data
        profile_data = profile_data_list[i]
        distances = profile_data["distances"]
        if scale is not None:
            distances = distances * scale  # pixel -> nm
        values = profile_data["values"]
        xs, ys = profile_data["coords"]

        # create colormap and rgb image
        rgb = array_to_rgb(arr, vmin, vmax) 

        # --- height map and line profile ---
        ax.imshow(rgb, origin="upper") 

        # plot line
        if overlay_line:
            ax.plot(xs, ys, linestyle='--', color='white', linewidth=3, alpha=0.8)            
            # start point marker
            ax.plot(xs[0], ys[0], marker='o', color='white', markersize=7)
        ax.axis("off")

        # Set text (optional)
        if texts[i]:
            ax.text(
                0.33, 0.12, texts[i], 
                transform=ax.transAxes, 
                color='white',
                fontsize=36,
                fontweight='bold',
                ha='left', va='top', 
            )

        # colorbar
        if colorbar and i == 2:
            im_dummy = ax.imshow(arr, cmap=cmap, norm=norm, alpha=0)
            cbar = fig.colorbar(im_dummy, ax=ax, location='right', shrink=1.0)
            cbar.set_label('Height (nm)', fontsize=24)
            cbar.solids.set_alpha(1)
            cbar.ax.tick_params(labelsize=22)

        # --- line profile ---
        ax4.plot(distances, values, color=colorlist[i], linewidth=3.0)
        ax4.set_xlabel("Distance (nm)" if scale is not None else "Pixel", fontsize=24, labelpad=10) 
        ax4.set_ylabel("Height (nm)", fontsize=24, labelpad=12)
        ax4.tick_params(axis='both', labelsize=22) 
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.tick_right()

    # aspect ratio
    img_h, img_w = img.shape
    img_aspect = img_h / img_w

    target_aspect = img_aspect / (wr[3] / wr[0])
    ax4.set_box_aspect(target_aspect)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved heightmaps with line profile to {save_path}")
    else:
        plt.show()

def plot_multitask_comprehensive(
        arr_list, 
        attn_map, 
        class_logits,       
        true_label=None,    
        start=(0,0),
        angle_deg=0,
        colorlist=colorlist,
        colorbar=True,
        texts=None,
        image_size=None,
        save_path=None,
        num_points=None,
        alpha=0.7,
        cmap_attn='viridis',
        overlay_line=True,
        ):
    """
    Layout: [Img1] [Img2].. [Img N] [Profile Graph] [Img1+Attn] [Prob Bar]
    """
    
    set_plot_style()

    arr_count = len(arr_list)

    # --- Preprocessing (Images) ---
    if image_size is not None:
        arr_list = [center_crop_or_pad(arr, image_size) for arr in arr_list]

    for i in range(len(arr_list)):
        img = arr_list[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        arr_list[i] = img

    input_img = arr_list[0] # For Attention Overlay

    # --- Preprocessing (Attention) ---
    if isinstance(attn_map, np.ndarray):
        attn_tensor = torch.from_numpy(attn_map)
    else:
        attn_tensor = attn_map.cpu()

    if attn_tensor.ndim == 2: attn_tensor = attn_tensor.unsqueeze(0).unsqueeze(0)
    elif attn_tensor.ndim == 3: attn_tensor = attn_tensor.unsqueeze(0)

    attn_resized = F.interpolate(
        attn_tensor,
        size=input_img.shape[-2:], 
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    # --- Preprocessing (Classification) ---
    if torch.is_tensor(class_logits):
        class_logits = class_logits.cpu()
    else:
        class_logits = torch.tensor(class_logits)
        
    # Softmax to get probabilities
    probs = F.softmax(class_logits.float(), dim=-1).numpy()
    num_classes = len(probs)

    # --- Settings ---
    if texts is None: texts = [None, None, None]
    vmin = min(arr.min() for arr in arr_list)
    vmax = max(arr.max() for arr in arr_list)
    cmap, norm = create_afmhot_cmap(vmin, vmax)

    # --- Line Profile Calculation ---
    profile_data_list = []
    for arr in arr_list:
        profile_data = get_line_profile(arr, start, angle_deg, num_points)
        profile_data_list.append(profile_data)

    # --- Plotting Setup ---
    wr = [1.0] * arr_count + [1.0, 1.5, 1.5]
    fig, axes = plt.subplots(1, arr_count + 3, figsize=(int(6*arr_count+24), 6), 
                             gridspec_kw={'width_ratios': wr},
                             layout='constrained')
    
    array_axes = axes[:arr_count]
    ax_attn = axes[arr_count]  # Attention Overlay
    ax_line = axes[arr_count + 1]      # Line Profile Graph
    ax_prob = axes[arr_count + 2]  # Probability Bar Chart

    # === 1. Images with Line ===
    for i, ax in enumerate(array_axes):
        arr = arr_list[i]
        profile_data = profile_data_list[i]
        xs, ys = profile_data["coords"]

        rgb = array_to_rgb(arr, vmin, vmax) 
        ax.imshow(rgb, origin="upper") 
        if overlay_line:
            ax.plot(xs, ys, linestyle='--', color='white', linewidth=3, alpha=0.8)
            ax.plot(xs[0], ys[0], marker='o', color='white', markersize=7)
        ax.axis("off")

        if texts[i]:
            ax.text(0.33, 0.12, texts[i], transform=ax.transAxes, color='white',
                    fontsize=36, fontweight='bold', ha='left', va='top')

        if colorbar and i == arr_count - 1:
            im_dummy = ax.imshow(arr, cmap=cmap, norm=norm, alpha=0)
            cbar = fig.colorbar(im_dummy, ax=ax, location='right', shrink=1.0)
            cbar.set_label('Height (nm)', fontsize=24)
            cbar.solids.set_alpha(1)
            cbar.ax.tick_params(labelsize=22)

    # === 2. Attention Overlay (ax_attn) ===
    rgb_input = array_to_rgb(input_img, vmin, vmax)
    ax_attn.imshow(rgb_input, origin="upper")
    im_attn = ax_attn.imshow(attn_norm, cmap=cmap_attn, alpha=alpha, origin="upper")
    ax_attn.axis("off")

    if colorbar:
        cbar2 = fig.colorbar(im_attn, ax=ax_attn, location='right', shrink=1.0)
        cbar2.set_label('Attention Rollout Score', fontsize=24)
        cbar2.ax.tick_params(labelsize=22)
        cbar2.solids.set_alpha(1)

    # === 3. Line Profile Graph (ax_line) ===
    distances = profile_data_list[0]["distances"]
    for i, profile_data in enumerate(profile_data_list):
        values = profile_data["values"]
        ax_line.plot(distances, values, color=colorlist[i], linewidth=3.0, label=f"Img {i+1}")
    ax_line.set_xlabel("Pixel", fontsize=24, labelpad=10) 
    ax_line.set_ylabel("Height (nm)", fontsize=24, labelpad=12)
    ax_line.tick_params(axis='both', labelsize=22) 
    ax_line.yaxis.set_label_position("right")
    ax_line.yaxis.tick_right()
    ax_line.grid(True, linestyle="--", alpha=0.5)

    img_h, img_w = arr_list[0].shape
    img_aspect = img_h / img_w
    target_aspect = img_aspect / (wr[arr_count + 1] / wr[0])
    ax_line.set_box_aspect(target_aspect)

    # === 4. Probability Bar Chart (ax_prob) ===
    x_pos = np.arange(1, num_classes+1) # 1-based index for x-axis
    
    bar_colors = ["#e29a9a"] * num_classes
    
    bars = ax_prob.bar(
        x_pos, 
        probs, 
        color=bar_colors, 
        linewidth=1.5, 
        width=0.6,
        alpha=0.9 
    )

    if true_label is not None and 0 <= true_label < num_classes:
        ax_prob.axvline(
            x=int(true_label+1), # 0-based to 1-based index
            color='black',      
            linestyle='--',     
            linewidth=3,        
            label='Ground Truth', 
            zorder=10           
        )
        
    ax_prob.set_ylabel("Probability", fontsize=24, labelpad=10)
    ax_prob.set_ylim(0, 1.1) 
    ax_prob.set_xticks(x_pos)
    ax_prob.tick_params(axis='both', labelsize=22)
    ax_prob.yaxis.set_label_position("left")
    ax_prob.yaxis.tick_left()

    target_aspect_bar = img_aspect / (wr[arr_count + 2] / wr[0])
    ax_prob.set_box_aspect(target_aspect_bar)

    # === Save ===
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# =============================
# Plot for Rigid Body Fitting
# =============================
# RB vs AE vs Ground Truth and predicted State
def plot_three_heightmaps_with_state_attn_rollout(
        arr_list, 
        attn_map,
        class_logits,
        rb_state, # RB predicted state (int) 0-indexed
        image_size=None, 
        save_path=None, 
        colorbar=True,
        alpha=0.7,  
        cmap_attn='viridis',
        ):
    """
    plot three heightmaps side by side.

    Args:
        arr_list (list of ndarray): 3 arrays(2D) [arr1, arr2, arr3]
        texts (list of str, optional): 3 titles [title1, title2, title3]
        image_size (int, optional): size to crop/pad
        save_path (str, optional): path to save the figure
        colorbar (bool): whether to display colorbar
    """
    
    # Input check
    if len(arr_list) != 3:
        raise ValueError("arr_list must contain exactly 3 arrays.")
    
    set_plot_style()

    # --- Preprocessing ---
    # Adjust image sizes
    if image_size is not None:
        arr_list = [center_crop_or_pad(arr, image_size) for arr in arr_list]

    input_img = arr_list[0] # For Attention Overlay

    # --- Preprocessing (Classification) ---
    if torch.is_tensor(class_logits):
        class_logits = class_logits.cpu()
    else:
        class_logits = torch.tensor(class_logits)

    # --- Preprocessing (Attention) ---
    if isinstance(attn_map, np.ndarray):
        attn_tensor = torch.from_numpy(attn_map)
    else:
        attn_tensor = attn_map.cpu()

    if attn_tensor.ndim == 2: attn_tensor = attn_tensor.unsqueeze(0).unsqueeze(0)
    elif attn_tensor.ndim == 3: attn_tensor = attn_tensor.unsqueeze(0)

    attn_resized = F.interpolate(
        attn_tensor,
        size=input_img.shape[-2:], 
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)


    # Softmax to get probabilities
    probs = F.softmax(class_logits.float(), dim=-1).numpy()
    num_classes = len(probs)

    # --- Colormap Setup ---
    # Get the min and max across all three images to unify the scale
    vmin = min(arr.min() for arr in arr_list)
    vmax = max(arr.max() for arr in arr_list)
    cmap, norm = create_afmhot_cmap(vmin, vmax)

    # --- Plotting ---
    fig = plt.figure(figsize=(20, 6))

    arr_count = len(arr_list)
    wr = [1.0] * (arr_count+1) + [0.1] + [1.5]
    fig, axes = plt.subplots(1, arr_count + 3, figsize=(int(6*arr_count+20), 6), 
                             gridspec_kw={'width_ratios': wr},
                             layout='constrained')
    
    array_axes = [axes[0], axes[2], axes[3]] # Heightmaps
    ax_attn    = axes[1]      # Attention Overlay
    ax_space  = axes[arr_count+1]  # Spacer
    ax_prob = axes[arr_count + 2]  # Probability Bar Chart

    # Plot three images in a loop
    for i, ax in enumerate(array_axes):
        arr = arr_list[i]
        
        # Convert array to AFM-style RGB
        rgb = array_to_rgb(arr, vmin, vmax) # transform to RGB using unified vmin/vmax
        
        # Display
        ax.imshow(rgb, origin="upper")
        ax.axis("off")

    # --- Colorbar ---
    if colorbar and i == arr_count - 1:
        im_dummy = ax.imshow(arr, cmap=cmap, norm=norm, alpha=0)
        cbar = fig.colorbar(im_dummy, ax=ax, location='right', shrink=1.0)
        cbar.set_label('Height (nm)', fontsize=26)
        cbar.solids.set_alpha(1)
        cbar.ax.tick_params(labelsize=22)

    # === 2. Attention Overlay (ax_attn) ===
    rgb_input = array_to_rgb(input_img, vmin, vmax)
    ax_attn.imshow(rgb_input, origin="upper")
    im_attn = ax_attn.imshow(attn_norm, cmap=cmap_attn, alpha=alpha, origin="upper")
    ax_attn.axis("off")

    if colorbar:
        cbar2 = fig.colorbar(im_attn, ax=ax_attn, location='right', shrink=1.0)
        cbar2.set_label('Attention Rollout Score', fontsize=26, labelpad=20)
        cbar2.ax.tick_params(labelsize=22)
        cbar2.solids.set_alpha(1)

    # === 3. Spacer (ax_space) ===
    ax_space.axis("off")

    # === 4. Probability Bar Chart (ax_prob) ===
    img_h, img_w = arr_list[0].shape
    img_aspect = img_h / img_w

    x_pos = np.arange(1, num_classes+1) # 1-based index for x-axis
    
    bar_colors = ["#e29a9a"] * num_classes
    
    bars = ax_prob.bar(
        x_pos, 
        probs, 
        color=bar_colors, 
        linewidth=1.5, 
        width=0.6,
        alpha=0.9 
    )

    if rb_state is not None and 0 <= rb_state < num_classes:
        ax_prob.axvline(
            x=int(rb_state+1), # 0-based to 1-based index
            color='black',      
            linestyle='--',     
            linewidth=3,        
            label='RB State', 
            zorder=10           
        )

    ax_prob.set_ylabel("Probability", fontsize=26, labelpad=8)
    ax_prob.set_ylim(0, 1.1) 
    ax_prob.set_xticks(x_pos)
    ax_prob.tick_params(axis='both', labelsize=22)
    ax_prob.yaxis.set_label_position("left")
    ax_prob.yaxis.tick_left()

    target_aspect_bar = img_aspect / (wr[arr_count+2] / wr[0])
    ax_prob.set_box_aspect(target_aspect_bar)
    
    # --- Save ---
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# RB vs AE with Diff
def plot_three_heightmaps_with_diff(
        arr_list, 
        texts=[None, None, None], 
        image_size=None, 
        save_path=None, 
        colorbar=True):
    """
    Plot 5 images in a row: [Ref, Img1, Diff1, Img2, Diff2]
    
    Args:
        arr_list (list): 3 arrays [Ref, Img1, Img2]
        texts (list): 3 titles corresponding to arr_list
    """

    set_plot_style()
    
    # Input check
    if len(arr_list) != 3:
        raise ValueError("arr_list must contain exactly 3 arrays.")
    
    # --- Preprocessing ---
    if image_size is not None:
        arr_list = [center_crop_or_pad(arr, image_size) for arr in arr_list]

    ref_img = arr_list[0]
    img1 = arr_list[1]
    img2 = arr_list[2]

    # --- 1. Calculate Ranges (Scale) ---
    # Common scale for height maps (AFM images)
    vmin_h = min(arr.min() for arr in arr_list)
    vmax_h = max(arr.max() for arr in arr_list)

    cmap, norm = create_afmhot_cmap(vmin_h, vmax_h)

    # Common scale for difference maps (difference images)
    diff1 = np.absolute(img1 - ref_img)
    diff2 = np.absolute(img2 - ref_img)
    
    max_diff = max(np.abs(diff1).max(), np.abs(diff2).max())
    if max_diff < 1e-6: max_diff = 1.0
    
    vmin_d = -max_diff
    vmax_d = max_diff

    # --- 2. Prepare Data for Plotting ---
    plot_data = [
        {"data": ref_img, "type": "height", "text": texts[0]},
        {"data": img1,    "type": "height", "text": texts[1]},
        {"data": diff1,   "type": "diff",   "text": None},
        {"data": img2,    "type": "height", "text": texts[2]},
        {"data": diff2,   "type": "diff",   "text": None},
    ]

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(30, 6))
    
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 5),
                     axes_pad=0.3,
                     cbar_mode='each',     
                     cbar_location='bottom',
                     cbar_pad=0.05,
                     cbar_size="5%",
                     share_all=True        
                     )

    for i, ax in enumerate(grid):
        item = plot_data[i]
        data = item["data"]
        item_type = item["type"] 
        cax = grid.cbar_axes[i]
        
        if item_type == "height":
            # --- Height Image ---
            vmin, vmax = vmin_h, vmax_h
            cbar_label = "Height (nm)"
            
            rgb = array_to_rgb(data, vmin, vmax)
            im = ax.imshow(rgb, origin="upper")
            
            mappable = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            mappable.set_array([])
            
        else:
            # --- Difference Image ---
            vmin, vmax = 0, vmax_d 
            current_cmap = 'viridis'  
            cbar_label = "Abs Diff (nm)"
            
            im = ax.imshow(data, cmap=current_cmap, vmin=vmin, vmax=vmax, origin="upper") 
            
            mappable = im

        ax.axis("off")
        
        # Set text (optional)
        if item["text"] is not None:
            ax.text(
                0.33, 0.12, item["text"], 
                transform=ax.transAxes, 
                color='white',
                fontsize=36,
                fontweight='bold',
                ha='left', va='top', 
            )

        # --- Colorbar ---
        if colorbar:            
            is_diff = (item_type == "diff")

            if i ==3 or (is_diff and i==4): 
                cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
                cbar.set_label(cbar_label, fontsize=22)
                cbar.ax.tick_params(labelsize=20)
            else:
                cax.axis("off")
        else:
            cax.axis("off")

    # --- Save ---
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
        plt.close(fig)
    else:
        plt.show()

# RB vs AE with AttnRollout
def plot_rb_vs_ae_with_attn_rollout(
        arr_list, 
        attn_map, 
        colorbar=True,
        texts=None,
        image_size=None,
        save_path=None,
        alpha=0.7,
        cmap_attn='viridis',
        ):
    """
    Layout: [Img1] [Img2] [Attention Overlay] empty [Img3] empty
    """
    
    set_plot_style()

    arr_count = len(arr_list)

    # --- Preprocessing (Images) ---
    if image_size is not None:
        arr_list = [center_crop_or_pad(arr, image_size) for arr in arr_list]

    for i in range(len(arr_list)):
        img = arr_list[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        arr_list[i] = img

    input_img = arr_list[0] # For Attention Overlay

    # --- Preprocessing (Attention) ---
    if isinstance(attn_map, np.ndarray):
        attn_tensor = torch.from_numpy(attn_map)
    else:
        attn_tensor = attn_map.cpu()

    if attn_tensor.ndim == 2: attn_tensor = attn_tensor.unsqueeze(0).unsqueeze(0)
    elif attn_tensor.ndim == 3: attn_tensor = attn_tensor.unsqueeze(0)

    attn_resized = F.interpolate(
        attn_tensor,
        size=input_img.shape[-2:], 
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    # --- Settings ---
    if texts is None: texts = [None, None, None]
    vmin = min(arr.min() for arr in arr_list)
    vmax = max(arr.max() for arr in arr_list)
    cmap, norm = create_afmhot_cmap(vmin, vmax)

    # --- Plotting Setup ---
    wr = [1.0] * int(arr_count+3)
    fig, axes = plt.subplots(1, int(arr_count + 3), figsize=(int(6*arr_count+24), 6), 
                             gridspec_kw={'width_ratios': wr},
                             layout='constrained')
    
    array_axes = [axes[0], axes[2], axes[4]]
    ax_attn    = axes[1]
    ax_empty   = [axes[3], axes[5]]

    # === 1. Images with Line ===
    for i, ax in enumerate(array_axes):
        arr = arr_list[i]
        rgb = array_to_rgb(arr, vmin, vmax) 
        ax.imshow(rgb, origin="upper") 
        ax.axis("off")

        if texts[i]:
            ax.text(0.33, 0.11, texts[i], transform=ax.transAxes, color='white',
                    fontsize=36, fontweight='bold', ha='left', va='top')

        if colorbar and i == arr_count - 1:
            im_dummy = ax.imshow(arr, cmap=cmap, norm=norm, alpha=0)
            cbar = fig.colorbar(im_dummy, ax=ax, location='right', shrink=1.0)
            cbar.set_label('Height (nm)', fontsize=28)
            cbar.solids.set_alpha(1)
            cbar.ax.tick_params(labelsize=22)

    # === 2. Attention Overlay (ax_attn) ===
    rgb_input = array_to_rgb(input_img, vmin, vmax)
    ax_attn.imshow(rgb_input, origin="upper")
    im_attn = ax_attn.imshow(attn_norm, cmap=cmap_attn, alpha=alpha, origin="upper")
    ax_attn.axis("off")

    if colorbar:
        cbar2 = fig.colorbar(im_attn, ax=ax_attn, location='right', shrink=1.0)
        cbar2.set_label('Attention Rollout Score', fontsize=28, labelpad=20)
        cbar2.ax.tick_params(labelsize=22)
        cbar2.solids.set_alpha(1)

    # === 3. Empty Axes ===
    for ax in ax_empty:
        ax.axis("off")

    # === Save ===
    if save_path:
        fig.savefig(save_path, dpi=800, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# =============================
# Plot for ViT
# =============================
def plot_heightmap_with_grid(
        arr,
        patch_size, 
        colorbar=True,
        save_path=None,
        ):
    """
    Visualize heightmap with grid lines indicating patches. 
    """
    set_plot_style()

    vmin, vmax = arr.min(), arr.max() 
    cmap, norm = create_afmhot_cmap(vmin, vmax)
    cax = None

    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis("off")

    im_for_cbar = ax.imshow(arr, cmap=cmap, norm=norm, origin="upper")

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

    # ---- colorbar ----
    if colorbar and cax is not None:
        cbar = fig.colorbar(
            im_for_cbar, 
            cax=cax, 
            orientation='vertical', 
        )
        cbar.set_label('Height (nm)', fontsize=22)
        cbar.solids.set_alpha(1)
        cbar.ax.tick_params(labelsize=20)

    # ---- grid lines (Corrected) ----
    grid_color = "#BBB7B4"  
    img_h, img_w = arr.shape
    
    y_start, y_end = -0.5, img_h - 0.5
    x_start, x_end = -0.5, img_w - 0.5

    grid_style = {'colors': grid_color, 'linewidth': 1.5, 'linestyles': '-'}

    # --- Horizontal lines ---
    h_inner = np.arange(patch_size, img_h, patch_size) - 0.5
    h_positions = np.sort(np.concatenate(([y_start, y_end], h_inner)))
    ax.hlines(h_positions, xmin=x_start, xmax=x_end, **grid_style)

    # --- Vertical lines ---
    v_inner = np.arange(patch_size, img_w, patch_size) - 0.5
    v_positions = np.sort(np.concatenate(([x_start, x_end], v_inner)))
    
    ax.vlines(v_positions, ymin=y_start, ymax=y_end, **grid_style)
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_end, y_start) 

    # ---- save ----
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_selected_patches_vertical(
        patches_tensor, 
        indices, 
        patch_size=3, 
        save_path=None
    ):
    """
    Visualize selected patches vertically with consistent coloring.
    
    Args:
        patches_tensor (Tensor or ndarray): Shape [1, NumPatches, FlattenedDim] e.g., [1, 144, 9]
        indices (list): List of patch indices to show e.g., [0, 12, 143]
        patch_size (int): Height/Width of the patch (assuming square).
    """
    
    # Tensor -> Numpy
    if hasattr(patches_tensor, 'cpu'):
        data = patches_tensor.detach().cpu().numpy()
    else:
        data = patches_tensor
        
    assert data.shape[0] == 1, "Expected batch size of 1."

    # (Batch, NumPatches, FlattenedDim) -> (NumPatches, FlattenedDim)
    data = data.squeeze(0)
    
    # For consistent coloring across patches
    vmin = data.min()
    vmax = data.max()
    
    cmap, norm = create_afmhot_cmap(vmin, vmax)

    num_patches = len(indices)
    
    fig, axes = plt.subplots(num_patches, 1, figsize=(2, 2.1 * num_patches))
    
    if num_patches == 1:
        axes = [axes]

    grid_color = "#BBB7B4" 
    grid_style = {'colors': grid_color, 'linewidth': 1.0, 'linestyles': '-'}

    for ax, idx in zip(axes, indices):
        # Flattened -> 2D
        if idx is None:
            ax.axis('off')
            continue

        patch_flat = data[idx]
        patch_img = patch_flat.reshape(patch_size, patch_size)
        
        ax.imshow(patch_img, cmap=cmap, norm=norm, interpolation='nearest')
        ax.axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()