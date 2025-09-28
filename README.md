# 🈶 Chinese MNIST Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify handwritten Chinese numerals (0–14) using the Chinese MNIST dataset. It demonstrates how deep learning can be applied to multilingual character recognition and image classification tasks.

---

## 📂 Dataset Overview

- **Source**:
  - [Chinese MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist)

- **Total Images**:
  15,000 grayscale `.jpg` images
  
- **Classes**:
  15 Chinese numerals (0–14)
  
- **Structure**:
Chinese_MNIST/

├── chinese_mnist.csv

├── input_0_0_0.jpg

├── input_0_0_1.jpg

└── ...

- **Labels**:
  - `value`: Integer label (0–14)
  - `character`: Corresponding Chinese character
  
- **Format**:
  Grayscale images, 64×64 resolution

---

## 🧠 Model Architecture

- **Framework**: TensorFlow / Keras
- **Model**: Custom CNN with:
- Conv2D + MaxPooling layers
- Dropout for regularization
- Dense layers with softmax output
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

---

## 🚀 How to Run

### Clone the repo:
   
 git clone https://github.com/AK-Jeevan/Chinese-MNIST-Classification-using-CNN.git
 cd Chinese-MNIST-Classification-using-CNN
 
### Install dependencies:

pip install -r requirements.txt

### Train the model:

python cnn_train.py

## 📊 Results

Achieved high accuracy on validation set

Visualized predictions and confusion matrix

Demonstrated effective classification of Chinese numerals

## 📌 Applications

Multilingual OCR systems

Educational tools for Chinese language learners

Character recognition research

## 🧾 License
This project is released under the MIT License. Dataset is under CC0: Public Domain.
