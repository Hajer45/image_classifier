# Flower Image Classification Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Files Included](#files-included)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Making Predictions](#making-predictions)
6. [Model Architecture](#model-architecture)
7. [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements a deep learning model to classify images of flowers into 102 categories. The model is built using transfer learning with pre-trained architectures such as ResNet-34 or EfficientNet-V2-S. The project includes two command-line scripts:
- `train.py`: Trains a new network on a dataset and saves the trained model as a checkpoint.
- `predict.py`: Uses a trained network to predict the class (and associated probability) of an input flower image.

The project adheres to best practices in PyTorch and modularizes code into reusable components (`model.py` and `utils.py`).

---

## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.7 or higher
- PyTorch and torchvision (`pip install torch torchvision`)
- PIL (Python Imaging Library) for image processing (`pip install pillow`)
- Matplotlib for visualization (`pip install matplotlib`)
- NumPy for numerical operations (`pip install numpy`)

Additionally, download the dataset and `cat_to_name.json` file:
- Dataset: [Flowers Dataset](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip)
- JSON File: `cat_to_name.json` (included in the project directory)

---

## Files Included

- **`train.py`**: Script to train a new network on the flower dataset and save the model checkpoint.
- **`predict.py`**: Script to load a trained network and predict the class of an input image.
- **`model.py`**: Contains functions for building, saving, and loading the model.
- **`utils.py`**: Utility functions for data loading and image preprocessing.
- **`cat_to_name.json`**: JSON file mapping category labels to real flower names.
- **`flowers/`**: Directory containing the training, validation, and test datasets.

---

## Setup Instructions

1. **Install Dependencies**:
   Ensure all required libraries are installed:
   ```bash
   pip install torch torchvision pillow matplotlib numpy
