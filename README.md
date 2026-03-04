# DeepAFM

## Overview

Official implementation of **"Estimating Protein Conformational States from High-Speed AFM Images Using Molecular Dynamics and Deep Learning"** (in revision)

This repository provides a framework for HS-AFM image analysis of protein dynamics, with a focus on denoising and conformational state estimation. By integrating molecular dynamics (MD) simulations, AFM image generation that mimics experimental conditions (including temporal lag effects between line scans), and Vision Transformer models, DeepAFM enables:

- **Simulation-driven AFM data generation** from MD snapshots with scan-induced distortions and realistic noise
- **Joint denoising and conformational state estimation** for robust interpretation of noisy HS-AFM measurements

**Key features:**
- Generate large-scale AFM datasets from MD simulations in WebDataset format
- Train ViT-based multi-task autoencoders for denoising and conformational state estimation
- Transfer pretrained models to new protein systems for efficient adaptation

**For researchers in:** Structural biology, AFM imaging, protein dynamics, computational biophysics

---

## Quick Start

### Installation

**Requirements:** Python 3.9+

1. Clone this repository:
   ```bash
   git clone https://github.com/kasatou/DeepAFM
   cd DeepAFM
   ```

2. Create and activate a Python virtual environment (recommended):
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate  
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and set up the dataset:
   
   Download the dataset archive (`18587081.zip`) from [Zenodo](https://zenodo.org/records/18587081) and place it in the repository root directory. Then run the setup script to extract and organize the files:
   
   ```bash
   bash setup.sh
   ```
   
   This will extract:
   - MD simulation files (PDB/DCD trajectories) → `data/md_raw/`
   - Pretrained model weights → `runs/pretrained_model/`

### Tutorial

Run the interactive notebooks:

```bash
jupyter lab tutorial.ipynb
# Optional: transfer learning demos
jupyter lab transfer_learning_MgtE.ipynb
jupyter lab transfer_learning_HECT.ipynb
```

The tutorial covers:
1. AFM image generation from MD simulations
2. Dataset construction in WebDataset format
3. Training a ViT-based multi-task autoencoder (denoising + conformational state estimation)
4. Model evaluation and result visualization
5. Transfer learning to new protein systems (MgtE and HECT examples)

---

## Repository Structure

```
DeepAFM/
├── tutorial.ipynb                # Interactive tutorial notebook
├── transfer_learning_MgtE.ipynb  # Transfer learning demo: denoising
├── transfer_learning_HECT.ipynb  # Transfer learning demo: multi-task learning
├── run.py                        # Main entry point
├── requirements.txt              # Python dependencies
├── data/                         # Datasets (MD trajectories, generated images)
│   └── md_raw/                   # Raw MD simulation data
├── runs/                         # Training logs, checkpoints, and results
│   └── pretrained_model/         # Pretrained model weights extracted from Zenodo
└── src/                          # Core source code
    ├── afm_image_generation/     # AFM image synthesis
    ├── configs/                  # Configuration files
    ├── dataload/                 # Data loading and preprocessing
    ├── models/                   # Model architectures (ViT, autoencoder)
    ├── training/                 # Training utilities
    └── evaluation/               # Evaluation and visualization
```


## Notes

- Keep large generated data outside version control; `data/` and `runs/` are in `.gitignore`.
- GPU recommended for training (CPU possible but slower).
- MD simulation data must be added to `data/md_raw` beforehand.
- **Environment:** Tested on Python 3.9 + CUDA 11.3 + PyTorch 1.12.1

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2026deepafm,
  title={Estimating Protein Conformational States from High-Speed AFM Images Using Molecular Dynamics and Deep Learning},
  author={Your Name and Co-authors},
  journal={},
  year={2026},
  doi={}
}
```

---

## Contact

For questions or issues:
- Open an issue on [GitHub Issues](https://github.com/TUS-MoriLaboratory/DeepAFM/issues)
- Email: t.mori@rs.tus.ac.jp

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

