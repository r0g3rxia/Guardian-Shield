
# Guardian Shield

## Overview

Guardian Shield is a project designed to detect and classify Personal Identifiable Information (PII) in images using Optical Character Recognition (OCR) and machine learning techniques. This tool aims to enhance data privacy and security by identifying sensitive information in images and providing a user-friendly interface for interaction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Detailed Workflow](#detailed-workflow)
  - [1. Import Necessary Libraries](#1-import-necessary-libraries)
  - [2. Create Dataset Directories](#2-create-dataset-directories)
  - [3. Define Functions to Generate Personal Information](#3-define-functions-to-generate-personal-information)
  - [4. Generate Images with Personal Information](#4-generate-images-with-personal-information)
  - [5. Generate Training and Validation Datasets](#5-generate-training-and-validation-datasets)
  - [6. Initialize EasyOCR and Define OCR and Data Extraction Functions](#6-initialize-easyocr-and-define-ocr-and-data-extraction-functions)
  - [7. Create Training and Validation Datasets](#7-create-training-and-validation-datasets)
  - [8. Check if All Required Labels are Present in the Dataset](#8-check-if-all-required-labels-are-present-in-the-dataset)
  - [9. Text Preprocessing and Model Training](#9-text-preprocessing-and-model-training)
  - [10. Define OCR and Prediction Functions](#10-define-ocr-and-prediction-functions)
  - [11. Create Graphical User Interface (GUI)](#11-create-graphical-user-interface-gui)
- [Future Improvements](#future-improvements)
- [References](#references)

## Features

- **PII Detection**: Automatically detect and classify various types of PII in images.
- **EasyOCR Integration**: Utilizes EasyOCR for text extraction from images.
- **Deep Learning Model**: Employs a neural network with embedding, convolutional, and LSTM layers for PII classification.
- **User-Friendly GUI**: Provides a Tkinter-based graphical interface for easy interaction.

## Setup

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/r0g3rxia/Guardian-Shield.git
   cd Guardian-Shield
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Guardian\ Shield.ipynb
   ```

2. Follow the steps in the notebook to generate datasets, train the model, and interact with the GUI.

## Project Structure

```
Guardian-Shield/
│
├── data/
│   ├── train/
│   └── validation/
│
├── models/
│   └── saved_model.h5
│
├── Guardian Shield.ipynb
├── requirements.txt
└── README.md
```

## Detailed Workflow

### 1. Import Necessary Libraries

Import various required libraries for image processing, generating fake data, OCR, and machine learning. Initialize the `Faker` library for generating UK formatted personal information.

### 2. Create Dataset Directories

Create folders to store the training and validation datasets, ensuring an organized structure.

### 3. Define Functions to Generate Personal Information

Define functions that generate personal information such as name, phone number, and email, creating a diverse dataset.

### 4. Generate Images with Personal Information

Generate images containing personal information, ensuring non-overlapping data within images, and save these images with corresponding XML annotation files.

### 5. Generate Training and Validation Datasets

Create 1000 training images and 200 validation images, saving their paths and annotation files to support model training and validation.

### 6. Initialize EasyOCR and Define OCR and Data Extraction Functions

Initialize EasyOCR and define functions to extract text from images. Read text and labels from annotation files and match them with OCR results to generate structured datasets.

### 7. Create Training and Validation Datasets

Use OCR and annotation files to generate DataFrames, facilitating efficient data manipulation and model training.

### 8. Check if All Required Labels are Present in the Dataset

Ensure that all necessary labels are present in the dataset, printing any missing labels to maintain data integrity.

### 9. Text Preprocessing and Model Training

Preprocess text data by converting it to sequences and padding it. Encode labels using `LabelEncoder`, build and train a neural network model, and save the trained model and preprocessors.

### 10. Define OCR and Prediction Functions

Utilize EasyOCR to detect text in images and classify the extracted text into PII categories using a trained neural network model. Annotate images with color-coded bounding boxes for visual feedback.

### 11. Create Graphical User Interface (GUI)

Develop a Tkinter-based GUI to facilitate easy interaction, allowing users to load images and view annotated results. The GUI displays processed images with highlighted and labeled PII text, providing immediate feedback.

## Future Improvements

- **Enhanced Model Performance**: Explore advanced neural network architectures and fine-tuning techniques to improve PII classification accuracy.
- **Expanded PII Categories**: Extend the model to detect additional types of PII beyond the current scope.
- **User Experience Enhancements**: Improve the GUI with more interactive features and better usability.

## References

- [EasyOCR Documentation](https://www.jaided.ai/easyocr/)
- [Faker Documentation](https://faker.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
