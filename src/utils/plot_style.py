# src/utils/plot_style.py

import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os

__PLOT_STYLE_ALREADY_SET__ = False


def set_plot_style(font_size: int = 14, font_path: str = None, verbose=True):
    """
    Set unified matplotlib style for publication-quality figures.

    - Uses Times New Roman if available
    - Falls back to DejaVu Serif
    - Idempotent: only applies once
    """

    global __PLOT_STYLE_ALREADY_SET__

    if __PLOT_STYLE_ALREADY_SET__:
        return

    __PLOT_STYLE_ALREADY_SET__ = True

    # --------------------------------------
    # 1. Try to load Custom Font (.ttf)
    # --------------------------------------
    if font_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_font_path = os.path.join(current_dir, "times.ttf") # bundled font
        
        if os.path.exists(default_font_path):
            font_path = default_font_path

    if font_path and os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
            if verbose:
                pass
                #print(f"[PlotStyle] Loaded custom font from: {font_path}")
        except Exception as e:
            print(f"[PlotStyle] Error loading font: {e}")

    # --------------------------------------
    # 2. Check font availability (Times New Roman)
    # --------------------------------------
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in available_fonts:
        font_family = "Times New Roman"
    else:
        font_family = "DejaVu Serif"  # fallback
        if verbose:
            #print("[PlotStyle] Warning: 'Times New Roman' not found. Using fallback font 'DejaVu Serif'.")
            pass

    # --------------------------------------
    # 3. Apply style
    # --------------------------------------
    rcParams.update({
        "font.family": font_family,
        "font.size": font_size,

        "mathtext.fontset": "cm", 
        "mathtext.rm": font_family, 
        "mathtext.it": f"{font_family}:italic",
        "mathtext.bf": f"{font_family}:bold",

        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,

        "lines.linewidth": 1.5,
        "axes.linewidth": 1.2,

        "figure.dpi": 120,
        "pdf.fonttype": 42, 
        "ps.fonttype": 42
    })

    if verbose:
        #print(f"[PlotStyle] Matplotlib style set to '{font_family}'.")
        pass

def new_fig(figsize=(6, 6)):
    """Convenient helper to create figures with unified style."""
    return plt.subplots(figsize=figsize)
