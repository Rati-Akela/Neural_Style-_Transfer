# AI Neural Style Transfer

An AI-powered image transformation application that combines the **content of one image with the artistic style of another image** using deep learning-based Neural Style Transfer.

## Overview

**AI Neural Style Transfer** uses deep learning to transform ordinary images into artistic compositions by separating an image's **content** from its **visual style**.

For example, a photograph can be transformed into an artwork inspired by styles such as **sketch, painting, or Ghibli-inspired aesthetics** while preserving the main structure and content of the original image.

The project demonstrates the application of **Convolutional Neural Networks (CNNs), feature extraction, image processing, and deep learning optimization** for creative AI applications.

## Key Features

* **AI-Powered Image Transformation** – Converts input images into artistic outputs.
* **Content Preservation** – Maintains the important structures and objects from the original image.
* **Style Transfer** – Applies the visual characteristics of a selected style image.
* **Multiple Artistic Styles** – Supports styles such as sketch, painting, and Ghibli-inspired transformations.
* **Deep Learning-Based Processing** – Uses CNN feature representations to understand image content and style.
* **Interactive Interface** – Allows users to upload images and generate stylized outputs.
* **Output Preview** – Displays the transformed image for easy comparison.

## AI & Deep Learning Concepts

The project applies:

* Neural Style Transfer
* Convolutional Neural Networks (CNNs)
* Deep Learning
* Feature Extraction
* Content Representation
* Style Representation
* Image Processing
* Optimization
* Computer Vision

## How Neural Style Transfer Works

The system separates an image into two important components:

```text
Content Image
     │
     ├──────────────┐
     │              │
     ▼              ▼
Content Features   Style Image
     │              │
     │              ▼
     │         Style Features
     │              │
     └───────┬──────┘
             ▼
      Feature Optimization
             │
             ▼
      Stylized Output
```

The model attempts to preserve the **content features** of the original image while reproducing the **style features** of the reference artwork.

## Tech Stack

| Category           | Technologies               |
| ------------------ | -------------------------- |
| Language           | Python                     |
| Deep Learning      | PyTorch / TensorFlow       |
| Computer Vision    | OpenCV                     |
| Image Processing   | PIL, NumPy                 |
| Model Architecture | CNN / Pre-trained CNN      |
| Interface          | Streamlit                  |
| Development        | Jupyter Notebook / VS Code |
| Version Control    | Git, GitHub                |

## Workflow

```text
Upload Content Image
        ↓
Upload / Select Style Image
        ↓
Preprocess Images
        ↓
Extract Deep Features
        ↓
Calculate Content & Style Loss
        ↓
Optimize Generated Image
        ↓
Generate Stylized Image
        ↓
Display Result
```

## Project Structure

```text
AI-NEURAL-STYLE-TRANSFER/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── ...
│
├── styles/
│   ├── sketch/
│   ├── painting/
│   └── ...
│
├── utils/
│   ├── preprocessing.py
│   └── style_transfer.py
│
├── outputs/
│   └── ...
│
└── assets/
    └── ...
```

> Update the structure to match the actual files in your repository.

## Installation

### 1. Clone the repository

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

If the project uses Streamlit:

```bash
streamlit run app.py
```

## Applications

Neural Style Transfer can be used in:

* Digital art generation
* Photo editing
* Creative design
* AI-assisted content creation
* Image enhancement
* Personalized artwork
* Creative media applications

## Future Improvements

* Add more artistic styles
* Implement faster real-time style transfer
* Support video style transfer
* Add user-created style presets
* Improve output resolution
* Optimize GPU processing
* Add batch image processing
* Deploy the model as a cloud-based AI service

## Author

**Rati Kumari Akela**

Computer Science Engineering Student
Interested in **Artificial Intelligence, Machine Learning, Deep Learning and Computer Vision**

