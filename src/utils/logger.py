import logging
import os

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "fidelity_summary.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Fidelity")
