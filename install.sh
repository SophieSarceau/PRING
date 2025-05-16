#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Installing packages..."

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install bioservices==1.12.1
pip install gprofiler==1.2.2
pip install matplotlib==3.8.4
pip install networkx==3.4.2 # Using the latest specified version
pip install numpy==2.2.5
pip install pandas==2.2.3
pip install pyemd==1.0.0
pip install python_igraph==0.11.8
pip install python_louvain==0.16
pip install scikit_learn==1.6.1
pip install scipy==1.15.3
pip install seaborn==0.13.2
pip install tqdm==4.67.1 # Using the latest specified version
pip install wwl==0.1.2

echo "All packages installed successfully."
