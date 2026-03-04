# afm_image_generation

This package contains utilities and simulators to generate synthetic AFM (Atomic Force Microscopy) images
from molecular coordinates and to perform common pre/post-processing tasks such as coordinate
rotation/translation and saving/loading trajectories using MDTraj.

Key goals:
- Provide reproducible AFM image generation primitives.
- Offer utilities for working with PDB/DCD trajectories and converting coordinate arrays to PDB files.
- Expose configurable parameters for fixed/random image generation and Brownian-like random transforms.

## Features
- AFM simulation core (height-map generation) and Brownian-motion style random transforms.
- PDB/DCD loading via MDTraj and helpers to save coordinates back to PDB.
- Parameter dataclasses for experiment-friendly configuration.

## Installation

This project uses a `src/` layout. To work with the `afm_image_generation` package from the repository root
you can either add `src` to `PYTHONPATH` or install the package in editable mode.

One-off (run tests / scripts):

```bash
cd /path/to/Simple_AFM
PYTHONPATH=src python -c "import afm_image_generation; print('ok')"
```

Editable install (recommended for development):

```bash
cd /path/to/Simple_AFM
python -m pip install -e .
```

Dependencies (examples):
- Python 3.8+
- numpy
- torch
- mdtraj (recommended to read/write PDB/DCD)
- matplotlib, seaborn (optional, for visualization)

Install with pip (example):

```bash
pip install numpy torch mdtraj matplotlib seaborn
```

## Directory structure

Overview of the main subpackages and files:

```
src/afm_image_generation/
├── configs/
│   └── afm_image_parameter.py        # dataclasses for fixed/random AFM params
├── constants/
│   └── atomic_radii.py               # mapping of residue/atom names to radii
├── core/
│   ├── afm_simulator.py              # AFM height-map simulator
│   └── brownian_motion.py            # random transform utilities
├── utils/
│   ├── afm_utils.py                  # helper utilities for AFM pipeline
│   └── pdb_utils.py                   # load/save/transform coordinates using mdtraj/torch
└── README.md                         # this file
```

## Quick usage examples

1) Load a trajectory and get coordinates + radii using `PDBUtils`:

```python
from afm_image_generation.utils.pdb_utils import PDBUtils
from configs.experiment_config import ExperimentConfig
from afm_image_generation.configs.afm_image_parameter import AFMImageFixedParams

cfg = ExperimentConfig()
cfg.system.device = 'cpu'
pdb_utils = PDBUtils(cfg)
fixed = AFMImageFixedParams()
fixed.pdb_path = 'data/example.pdb'
fixed.dcd_path = 'data/example.dcd'
xyz, radii = pdb_utils.load_mdtrj(fixed)
```

2) Save a single-frame coordinate array as a PDB (requires a matching topology PDB):

```python
# coords: numpy array shape (n_atoms, 3) in nanometers
pdb_utils.save_xyz_to_pdb(coords, out_path='out.pdb', topology_path='data/example.pdb')
```

3) Generate AFM images (high-level example):

```python
from afm_image_generation.core.afm_simulator import AFMSimulator
sim = AFMSimulator(params)
height = sim.render(xyz, radii)
```

Refer to the module docstrings for exact function/class names and parameters.

## Units and conventions
- Coordinates used by mdtraj are expected in nanometers (nm). If your coordinates are in Ångström,
  divide them by 10.0 before creating an mdtraj.Trajectory or calling save utilities.
- Rotations are specified in degrees in the helper utilities and converted internally to radians.

## Tests

Run tests from the repository root with `src` on PYTHONPATH:

```bash
cd /path/to/Simple_AFM
PYTHONPATH=src pytest -q
```

## Notes and tips
- If imports fail (`ModuleNotFoundError: afm_image_generation`), ensure `PYTHONPATH=src` or perform an editable install.
- If you need reproducible randomness, set torch.manual_seed(...) and numpy.random.seed(...).

## Contributing
- Keep code style consistent with the project (snake_case for functions/modules, PascalCase for classes).
- Add tests for new utilities under `src/tests/` and follow the existing pytest patterns.

If you'd like, I can also generate a minimal `pyproject.toml` and editable install instructions for this package.
