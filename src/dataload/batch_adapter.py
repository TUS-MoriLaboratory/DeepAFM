# src/dataload/batch_adapter.py

def prepare_batch_for_model(batch, task_mode):
    """
    Convert a flexible AFM batch into strict (x, y) required by model.
    
    Args:
        batch: dict from Dataset
        task_mode: string ("denoise", "classification", "regression", "multitask")

    Returns:
        x, y  (both Tensor or dict)
    """

    # ----------------------------
    # DENOISE (distorted → ideal)
    # ----------------------------
    if task_mode == "denoise":
        x = batch["distorted"]
        y = batch["ideal"]

    # ----------------------------
    # CLASSIFICATION (config→state)
    # ----------------------------
    elif task_mode == "classification":
        x = batch["distorted"]
        y = batch["state"]  # in preprocess 

    # ----------------------------
    # CLASSIFICATION (config→state)
    # ----------------------------
    elif task_mode == "ideal_classification":
        x = batch["ideal"]
        y = batch["state"]  # in preprocess 

    # ----------------------------
    # MULTITASK (denoise + classification + regression)
    # ----------------------------
    elif task_mode == "multitask":
        x = batch["distorted"]
        y = {
            "ideal": batch.get("ideal"),
            "state": batch.get("state"),
        }
    
    elif task_mode == "multitask_ideal":
        x = batch["ideal"]
        y = {
            "ideal": batch.get("ideal"),
            "state": batch.get("state"),
        }

    # ----------------------------
    # Rigid Body Fitting (no training, only inference)
    # ----------------------------
    elif task_mode == "rb_fitting_to_distorted":
        x = batch["distorted"]
        y = y = batch["state"]

    elif task_mode == "rb_fitting_to_ideal":
        x = batch["ideal"]
        y = y = batch["state"]

    # ----------------------------
    # Experiment Image Inference (no training, only inference)
    # ----------------------------
    elif task_mode == "experiment_image_inference":
        x = batch
        batch_size = x.shape[0]

        y = {
            "ideal": x,
            "state": [None] * batch_size,  # dummy value
        }

    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")

    return x, y
