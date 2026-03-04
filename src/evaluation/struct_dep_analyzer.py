import csv
import numpy as np
import os
import math
from pathlib import Path
from configs.experiment_config import ExperimentConfig

class StructDepAnalyzer:
    def __init__(self, exp_cfg: ExperimentConfig):
        
        # Set save directory for analysis results
        self.morph_dir = os.path.join(
            exp_cfg.system.project_root, 
            'runs', 
            exp_cfg.system.run_name, 
            'evaluation_of_structure_dependency',
            f'{exp_cfg.struct_dep.morphing_start}_to_{exp_cfg.struct_dep.morphing_end}',
        )

        self.num_classes = exp_cfg.model.num_classes

    def _find_result_dirs(self) -> list:
        """Find directories under self.morph_dir whose name ends with '_results'.

        Returns a list of Path objects (may be empty).
        """
        base = Path(self.morph_dir)
        if not base.exists():
            return []

        # Recursively search for directories ending with '_results'
        result_dirs = [p for p in base.rglob('*_results') if p.is_dir()]
        return result_dirs

    def calculate_entropy_for_all_csvs(self):
        """
        Search all `*_results` directories under `self.morph_dir`, find CSV files
        and run entropy calculation on each CSV.
        """
        base = Path(self.morph_dir)
        if not base.exists():
            print(f"[Analyzer] Morph directory does not exist: {self.morph_dir}")
            return

        result_dirs = self._find_result_dirs()
        if not result_dirs:
            print(f"[Analyzer] No '*_results' directories found under: {self.morph_dir}")
            return

        all_results = []
        for d in result_dirs:
            print(f"[Analyzer] Scanning directory: {d}")
            for csv_path in sorted(d.glob('*_results.csv')):
                out = self.calculate_entropy_from_csv(str(csv_path))
                if out:
                    all_results.extend(out)

        if all_results:
            output_path = os.path.join(self.morph_dir, 'structure_dependency_entropy_summary.csv')
            
            fieldnames = ["structure_id", "entropy", "sample_count"] + [f"mean_prob_{i}" for i in range(self.num_classes)] + [f"std_prob_{i}" for i in range(self.num_classes)]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"[Analyzer] Saved analysis to: {output_path}")
            return output_path
    
    def calculate_entropy_from_csv(self, input_csv_path):
        """
        Load a CSV file and calculate uncertainty (entropy) per structure.
        """
        print(f"[Analyzer] Processing: {input_csv_path}")

        grouped_data = {}
        prob_indices = []

        # --- load csv ---
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print("Warning: Empty CSV file.")
                return None

            try:
                possible_ids = ["structure_id"]
                id_idx = -1
                for pid in possible_ids:
                    if pid in header:
                        id_idx = header.index(pid)
                        break
                if id_idx == -1: raise ValueError("ID column not found")

                prob_indices = [i for i, col in enumerate(header) if col.startswith("prob_")]
                if not prob_indices: raise ValueError("Probability columns not found")

            except ValueError as e:
                print(f"Error parsing header: {e}")
                return None

            for row in reader:
                if not row: continue
                s_id = row[id_idx]
                try:
                    probs = np.array([float(row[i]) for i in prob_indices], dtype=np.float64)
                    
                    if s_id not in grouped_data:
                        grouped_data[s_id] = {
                            "sum_probs": np.zeros_like(probs), 
                            "sum_sq_probs": np.zeros_like(probs), 
                            "count": 0
                            }

                    grouped_data[s_id]["sum_probs"] += probs
                    grouped_data[s_id]["sum_sq_probs"] += probs ** 2 
                    grouped_data[s_id]["count"] += 1
                except ValueError:
                    continue

        # --- calculate entropy ---
        results = []
        
        for s_id, data in grouped_data.items():
            mean_probs = data["sum_probs"] / data["count"]
            # normalize
            mean_probs /= (np.sum(mean_probs) + 1e-9)
            

            variance = (data["sum_sq_probs"] / data["count"]) - (mean_probs ** 2)
            std_probs = np.sqrt(np.maximum(variance, 0))

            # entropy: -sum(p log p)
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9))
            
            res_row = {
                "structure_id": s_id,
                "entropy": entropy,
                "sample_count": data["count"]
            }
            # save mean probabilities as well
            for i, (m, s) in enumerate(zip(mean_probs, std_probs)):
                res_row[f"mean_prob_{i}"] = m
                res_row[f"std_prob_{i}"] = s
            
            results.append(res_row)
        
        return results