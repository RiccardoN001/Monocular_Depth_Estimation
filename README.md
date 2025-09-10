# Monocular Depth Estimation

This repository implements a deep neural network in PyTorch to tackle the problem of monocular depth estimation for a university challenge. The objective is to train a model capable of predicting depth from a single image, providing insights for applications in computer vision fields like autonomous driving and augmented reality.

## Overview
- **Task:** dense prediction (one depth value per pixel).
- **Dataset:** 3,469 images at **256×144** with corresponding ground-truth depth maps, plus a validation split.
- **Metrics:** RMSE (lower is better) and SSIM (higher is better).

## Architecture (high level)
- **Fully convolutional encoder–decoder** with U-Net-style **symmetric skip connections** to preserve spatial detail.
- **Residual blocks in the decoder** after each upsampling step to stabilize training and fuse local/global features.
- **Pretrained ConvNeXt encoder** in `features_only` mode to extract multi-scale features at 1/4, 1/8, 1/16, and 1/32 of the input resolution.
- Rationale and diagrams are provided in the report.

## Results (validation)
- **RMSE:** 2.3157  
- **SSIM:** 0.56997  
- Best checkpoint around **epoch 60**.

See the accompanying report [`MDE_Paper`](./MDE_paper_Nicolini.pdf) for details.

## Repository Structure

- **main.py**: This script is the main entry point of the project. It orchestrates the training process, including loading the dataset and setting up the model.
- **model.py**: This script contains the architecture of the deep neural network. 
- **solver.py**: Contains the logic for training and evaluating the model. It includes functions for managing the training loop, computing the loss, updating weights, and evaluating model performance.
- **dataset.py**: Defines a custom dataset class used to load and preprocess the images and depth maps. It ensures that the data is prepared in a format suitable for training the neural network.
- **utils.py**: Provides utility functions used throughout the project, such as functions for visualizing depth maps, calculating evaluation metrics, and handling data preprocessing.

## Requirements
- Python 3.9+
- PyTorch (CUDA if available)
- `timm` (for pretrained ConvNeXt)

