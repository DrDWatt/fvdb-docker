#!/bin/bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export MODEL_DIR=/home/dwatkins3/fvdb-docker/models
source /home/dwatkins3/miniforge3/bin/activate fvdb
exec python3 /home/dwatkins3/fvdb-docker/fvdb-viewer/viewer_service.py
