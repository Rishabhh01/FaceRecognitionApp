# Face Recognition App using Siamese Neural Network

This project implements a **real-time facial verification system** using a **Siamese Neural Network (SNN)** in TensorFlow and OpenCV. The model compares two images and predicts whether they belong to the **same person or not** — enabling **face verification**, not just classification.

---

## Project Structure

<pre lang="markdown"> ``` FaceRecognitionApp/
│
├── Facial Verification with Siamese Network.ipynb       # Main training notebook
│
├── data/                                    # Captured images
│   ├── anchor/
│   ├── positive/
│   └── negative/
│
├── archive/                                               # Dataset folder
│   └── lfw-deepfunneled/                                # (original LFW data)
│
├── siamesemodelv2.h5                                    ❌ Model file (not uploaded due to size)
├── siamesemodelv2.keras                                 ❌ Keras format model (also too large for GitHub)
│
└── .gitattributes                                       # For Git LFS tracking (if enabled)
 ``` </pre>

> **Note:**  
> The model files (`.h5` / `.keras`) and dataset were **not uploaded to GitHub** because they exceeded [GitHub’s 100MB file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).  
>  
> You can **generate your own dataset** and **train the model** by running the provided notebook.

---

## Dataset Details

This project uses a combination of **custom webcam images** and the **Labeled Faces in the Wild (LFW)** dataset to train and validate the Siamese Neural Network.

### 1. Custom Captured Images
Images collected in real-time using your webcam:

```
data/
├── anchor/      # Anchor images (press 'a')
├── positive/    # Positive images (press 'p')
└── negative/    # Negative images (optional)
```

These images are used to train the model on verifying facial similarity.

### 2. LFW Dataset (from Kaggle)
Additional face data is sourced from the **Labeled Faces in the Wild (LFW)** dataset.

```
archive/
└── lfw-deepfunneled/    # Original LFW dataset (unzipped)
```

- **Download from Kaggle:**  
  https://www.kaggle.com/datasets/scikit-learn/lfw-people

- ⚠️ **Note:** This dataset is not included in the GitHub repo due to size limits.  
  You must manually download and extract it to:  
  `archive/lfw-deepfunneled/`


## What is a Siamese Network?

A **Siamese Neural Network** is a type of deep learning model that learns **similarity** between pairs of inputs. Instead of classifying individual images, it **compares two images** and outputs a similarity score.

### Key Features:
- Two identical subnetworks (CNNs with shared weights)
- Embedding output of shape `(4096,)` for both images
- A custom **L1 Distance Layer** compares embeddings
- A sigmoid classifier predicts match probability

This approach enables **one-shot learning** — the model doesn't need retraining when new people are added.

---

## Pipeline Overview

### 1. **Data Preparation**
- **Anchor Images**: Captured from webcam using key `'a'`
- **Positive Images**: Same person as anchor using key `'p'`
- **Negative Images**: Random faces from LFW dataset

### 2. **Preprocessing**
- Resize images to `(100, 100)`
- Normalize pixel values to range `[0, 1]`

### 3. **Model Architecture**
- CNN-based embedding network with multiple Conv2D layers
- Outputs a 4096-dimensional embedding
- Custom `L1Dist` layer to compute absolute difference
- Dense layer (sigmoid) for final prediction

### 4. **Training**
- Triplet input: Anchor, Positive, Negative
- Binary label: `1` if match, `0` if not
- Loss: **Binary Crossentropy**
- Optimizer: **Adam**

### 5. **Verification**
- Real-time webcam capture
- Predicts similarity score between test image and anchor

---

## Tech Stack & Libraries

| Tool            | Purpose                      |
|-----------------|------------------------------|
| **TensorFlow**  | Model architecture & training|
| **Keras**       | Layers and training loop     |
| **OpenCV**      | Webcam access and image IO   |
| **Matplotlib**  | Visualization                |
| **NumPy**       | Numerical processing         |
| **Git & GitHub**| Version control & sharing    |

---

## Requirements

Install Python dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy 
```
If using Conda:
```bash
conda create -n face-rec-env python=3.11
conda activate face-rec-env
pip install tensorflow==2.18 opencv-python matplotlib numpy==1.26.4
```
## Running the Application

- Press **`a`** to capture **anchor** images  
- Press **`p`** to capture **positive** images  
- Press **`q`** to **quit** the capture loop  

> These images are saved into respective folders and later used to train and validate the Siamese Neural Network.

---

## Outcomes

- Trained **Siamese Neural Network** for facial verification  
- Can be extended into a **face login system** or **security access gate**  
- Lightweight and accurate enough for **real-time applications**

---
## Author
**Rishabh Chauhan**  
Email: rishabhc2234@gmail.com

