# ./src/utils/metric_analysis.py

import numpy as np
import os

def calc_accuracy_vs_tolerance(cm_path_list, names=None):
    """
    Takes multiple paths to Confusion Matrices and calculates the Accuracy for each
    tolerance level (+-n). Returns the results.

    Args:
        cm_path_list (list): List of paths to .npy files containing confusion matrices

    Returns:
        list of dict: [
            {
                "name": "experiment_1", 
                "tolerances": [0, 1, 2, ...], 
                "accuracies": [0.85, 0.95, 0.99, ...]
            }, 
            ...
        ]
    """
    results = []

    for path in cm_path_list:
        # Load confusion matrix(.npy)
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        cm = np.load(path)
        num_classes = cm.shape[0]
        total_samples = np.sum(cm)
        
        tolerances = []
        accuracies = []

        for n in range(num_classes):
            correct_count = 0

            for i in range(num_classes):     # True Class
                for j in range(num_classes): # Predicted Class
                    if abs(i - j) <= n:
                        correct_count += cm[i, j]
            
            acc = correct_count / total_samples if total_samples > 0 else 0
            
            tolerances.append(n)
            accuracies.append(acc)

            # 100% accuracy reached            
            #if acc >= 1.0:
            #    break

        # set result name
        if names is not None and len(names) == len(cm_path_list):
            name = names[cm_path_list.index(path)]
        else:
            name = os.path.splitext(os.path.basename(path))[0]
        
        results.append({
            "name": name,
            "tolerances": tolerances,
            "accuracies": accuracies
        })

    return results

def count_errors_beyond_tolerance(cm_path, tolerance):
    """
    Counts the number of incorrect predictions with error >= tolerance.

    Args:
        cm_path : Confusion matrix
        tolerance (int): Tolerance level (predictions with |true - pred| >= tolerance are counted as errors)

    Returns:
        int: Number of incorrect predictions with absolute error >= tolerance
    """
    error_count = 0
    cm = np.load(cm_path)
    num_classes = cm.shape[0]

    for i in range(num_classes):     # True Class
        for j in range(num_classes): # Predicted Class
            if abs(i - j) > tolerance:
                error_count += cm[i, j]

    return error_count