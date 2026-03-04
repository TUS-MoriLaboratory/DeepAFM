#!/bin/bash

# Setup script for DeepAFM tutorial
# This script extracts and organizes MD simulation files and pretrained model weights

# File paths
ZIP_FILE="18587081.zip"
DATA_DIR="./data"
MD_RAW="$DATA_DIR/md_raw"

PRE_DIR="runs/pretrained_model"
CHK_DIR="$PRE_DIR/checkpoints"

# Create directory structure for MD simulation data
mkdir -p "$MD_RAW/SecYAEG_NDs"
mkdir -p "$MD_RAW/MgtE_NDs"
mkdir -p "$MD_RAW/HECT"

# Create directory for pretrained model
mkdir -p "$CHK_DIR" 

# Extract all files from the archive
unzip $ZIP_FILE

# Move SecYAEG-NDs files 
# - rst_total.dcd: MD trajectory file
# - initial.pdb: Initial structure file
mv rst_total.dcd "$MD_RAW/SecYAEG_NDs/"
mv initial.pdb "$MD_RAW/SecYAEG_NDs/"

# Move MgtE-NDs files 
mv 2zy9_NDs_CA_only.pdb "$MD_RAW/MgtE_NDs/"
mv 2zy9_NDs.pdb "$MD_RAW/MgtE_NDs/"

# Move HECT domain files 
mv 1D5F_CA_only.pdb "$MD_RAW/HECT/"
mv 1D5F.pdb "$MD_RAW/HECT/"
mv 1ND7_CA_only.pdb "$MD_RAW/HECT/"
mv 1ND7.pdb "$MD_RAW/HECT/"
mv 3JVZ_CA_only.pdb "$MD_RAW/HECT/"
mv 3JVZ.pdb "$MD_RAW/HECT/"

# Move pretrained model files for transfer learning
# - best_model.pt: Pretrained weights
# - config.yaml: Model configuration
mv best_model.pt "$CHK_DIR"
mv config.yaml "$PRE_DIR"

echo "Setup complete! MD simulation files and pretrained model are now organized."