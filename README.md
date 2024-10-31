# Monocular Depth Estimation

This repository implements a deep neural network in PyTorch to tackle the problem of monocular depth estimation for a university challenge. The objective is to train a model capable of predicting depth from a single image, providing insights for applications in computer vision fields like autonomous driving and augmented reality.

## Project Overview

Monocular depth estimation is the task of estimating the depth of a scene using only one RGB image as input. This project aims to develop and train a deep neural network that can accurately predict depth maps from such images. The dataset used consists of urban scenes, and the model is evaluated using a validation set provided as part of the challenge.

## File Descriptions

- **main.py**: This script is the main entry point of the project. It orchestrates the training process, including loading the dataset and setting up the model.
- **model.py**: This script contains the architecture of the deep neural network. 
- **solver.py**: Contains the logic for training and evaluating the model. It includes functions for managing the training loop, computing the loss, updating weights, and evaluating model performance.
- **dataset.py**: Defines a custom dataset class used to load and preprocess the images and depth maps. It ensures that the data is prepared in a format suitable for training the neural network.
- **utils.py**: Provides utility functions used throughout the project, such as functions for visualizing depth maps, calculating evaluation metrics, and handling data preprocessing.

