# trainer/logger.py
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, List
from collections import defaultdict

from utils.plot_style import set_plot_style

class CSVLogger:
    def __init__(self, save_dir, filename="metrics.csv"):
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        self.headers = None

    def log(self, stats: Dict[str, Any]):
        """
        Log a dictionary of statistics.
        stats example: {"epoch": 1, "train_loss": 0.5, "val_acc": 0.8, ...}
        """
        # Round float values for cleaner CSV
        clean_stats = {}
        for k, v in stats.items():
            if isinstance(v, float):
                clean_stats[k] = round(v, 6)
            else:
                clean_stats[k] = v

        # Determine headers on first log
        if self.headers is None:
            self.headers = list(clean_stats.keys())
            
            # If file doesn't exist, write header
            if not os.path.exists(self.filepath):
                with open(self.filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.headers)
                    writer.writeheader()

        # Write row
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            # Handle cases where keys might be missing or new keys added (safe get)
            row = {k: clean_stats.get(k, "") for k in self.headers}
            writer.writerow(row)

    def plot(self, csv_path=None, extension=".png"):
        """
        Automatically load CSV and plot all metrics grouped by type.
        """
        set_plot_style()

        if csv_path is None:
            csv_path = self.filepath
        if not os.path.exists(csv_path):
            print(f"[Logger] CSV file not found: {csv_path}")
            return

        assert extension in [".png", ".pdf"], "Unsupported file extension for plots."

        # 1. Read Data
        data = defaultdict(list)
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    try:
                        data[k].append(float(v))
                    except ValueError:
                        pass # Skip non-numeric

        if "epoch" not in data:
            return
        
        epochs = data["epoch"]

        # 2. Group Metrics dynamically
        # Groups: "Loss", "LR", and others like "acc", "mse"
        metric_groups = defaultdict(list)
        
        for key in data.keys():
            if key == "epoch": continue
            
            elif "lr" in key:
                metric_groups["Learning Rate"].append(key)
            else:
                # Extract metric name (remove train_ or val_ prefix)
                # e.g. train_acc -> acc, val_state_acc -> state_acc
                clean_name = key.replace("train_", "").replace("val_", "")
                metric_groups[clean_name].append(key)

        # 3. Plot each group
        for group_name, keys in metric_groups.items():
            if not keys: continue
            
            plt.figure(figsize=(8, 5))
            
            for key in keys:
                # Determine style based on train/val
                label = key
                style = "-" if "train" in key else "--"
                marker = "o" if "train" in key else "s"
                
                if group_name == "state_acc" or group_name == "acc":
                    # Convert to percentage
                    data[key] = [v * 100 for v in data[key]]

                plt.plot(epochs, data[key], label=label, linestyle=style, marker=marker, markersize=4)

            plt.xlabel("Epoch")
            
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

            if group_name == "state_acc" or group_name == "acc":
                plt.ylabel("Accuracy (%)")
                plt.ylim(0, 100)
            else:
                plt.ylabel(group_name) 
                plt.legend()
            
            #plt.title(f"{group_name} Curve")
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            
            # Save filename: "curve_Loss.png", "curve_acc.png", etc.
            safe_name = group_name.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(self.save_dir, f"curve_{safe_name}{extension}"))
            print(f"[Logger] Saved plot: curve_{safe_name}{extension}")
            plt.close()