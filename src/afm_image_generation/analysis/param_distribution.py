# afm_image_generation/analysis/param_distribution.py

import os
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

from utils.plot_style import set_plot_style


def collect_param_distributions(reader):
    """
    collect parameter distributions from AFMWebdatasetReader and return as dict
    
    Args:
        reader: AFMWebdatasetReader or similar iterable object
    """
    param_dict = defaultdict(list)

    for sample in reader:
        cfg = sample["config"]  # dict

        for key, value in cfg.items():
            # only collect int/float or list of int/float
            if isinstance(value, (int, float)):
                param_dict[key].append(value)

            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                # tip_shape = [18,18] → key="tip_shape_0", "tip_shape_1"
                for i, v in enumerate(value):
                    param_dict[f"{key}_{i}"].append(v)

    return param_dict

def plot_param_histograms(param_dict, save_dir):
    """
    Plot and save histograms of parameters.
    Args:
        param_dict: dict of parameter name to list of values
        save_dir: directory to save histogram PNGs
    """

    os.makedirs(save_dir, exist_ok=True)
    set_plot_style(16)

    for key, values in param_dict.items():
        if len(values) == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=40, color="gray", edgecolor="black")
        plt.title(key)
        plt.xlabel("value")
        plt.ylabel("count")

        out_path = os.path.join(save_dir, f"{key}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[ParamDist] Saved → {out_path}")


def plot_param_histograms_unique(param_dict, save_dir):
    """
    Histogram where bins = unique values with auto bar width.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plot_style(16)

    for key, values in param_dict.items():

        # flatten (for list or list-of-list)
        flat = []
        for v in values:
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)

        if len(flat) == 0:
            continue

        # ============================================================
        # Special handling for pdb_num (too many unique values)
        # ============================================================
        if key == "pdb_num":
            print(f"[ParamDist] Using bucketing for {key} (bin=100).")

            bucket = 100
            # bucketize
            bucketed = [(v // bucket) * bucket for v in flat]

            uniq_vals, counts = np.unique(bucketed, return_counts=True)

            plt.figure(figsize=(7, 4))
            plt.bar(uniq_vals, counts, width=bucket * 0.8,
                    color="gray", edgecolor="black")

            plt.xlabel(f"{key} (bucket={bucket})")
            plt.ylabel("count")
            plt.title(f"{key} distribution (bucketed)")

            # xticks too dense → thin out
            if len(uniq_vals) > 20:
                plt.xticks(uniq_vals[::max(1, len(uniq_vals) // 20)])

            out_path = os.path.join(save_dir, f"{key}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"[ParamDist] Saved → {out_path}")
            continue  # skip normal logic

        uniq_vals, counts = np.unique(flat, return_counts=True)
        n_unique = len(uniq_vals)

        # ----------------------------------------------------------------
        # 1) fallback for too many unique values
        # ----------------------------------------------------------------
        if n_unique > 1000:
            print(f"[Warning] {key}: unique={n_unique} → fallback to 100 bins histogram")
            plt.figure(figsize=(7, 4))
            plt.hist(flat, bins=100, color="gray", edgecolor="black")
            plt.xlabel(key)
            plt.ylabel("count")

        else:
            # ----------------------------------------------------------------
            # 2) compute auto bar width
            # ----------------------------------------------------------------
            if n_unique > 1:
                diffs = np.diff(uniq_vals)
                diff_min = np.min(diffs)

                # avoid near-zero diff
                if diff_min < 1e-6:
                    diff_min = 1.0

                width = diff_min * 0.8  # shrink a bit for visibility
            else:
                width = 1.0

            plt.figure(figsize=(7, 4))
            plt.bar(uniq_vals, counts, width=width, color="gray", edgecolor="black")

            plt.xlabel(key)
            plt.ylabel("count")
            plt.title(f"{key} (unique bins: {n_unique})")

            # x-tick density control
            if n_unique > 20:
                plt.xticks(uniq_vals[::max(1, n_unique // 20)])

        out_path = os.path.join(save_dir, f"{key}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[ParamDist] Saved → {out_path}")


def analyze_param_distribution(reader, save_dir):
    """
    Analyze and plot parameter distributions from AFMWebdatasetReader.
    
    Args:
        reader: AFMWebdatasetReader or similar iterable object
        save_dir: directory to save histogram PNGs
    """
    param_dict = collect_param_distributions(reader)
    plot_param_histograms_unique(param_dict, save_dir)
