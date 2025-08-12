# 🐟 Multiclass Fish Image Classification


## 📌 Project Overview
- This project focuses on classifying fish images into multiple categories using **Deep Learning**. 
- We built a **custom CNN from scratch** and experimented with multiple **pre-trained models** (Transfer Learning) to identify the best-performing architecture for accurate predictions.  
- The **best model** in this project was **MobileNet**, which achieved the highest evaluation scores without fine-tuning.

## 🎯 Objectives
- Classify fish images into their correct categories.
- Compare the performance of a CNN and multiple pre-trained models.
- Deploy a **Streamlit web app** for real-time predictions.
- Provide a **comparison report** with evaluation metrics.


## 📂 Dataset
The dataset contains images of fish across multiple categories. 

Download dataset  from https://drive.google.com/drive/folders/1rcwesRsTSqQAw7O81IYHSNXlX-fyVAlQ

Images were preprocessed and augmented before training.

**Data Preprocessing & Augmentation:**
- Resizing all images to `224x224` pixels.
- Normalizing pixel values to range `[0, 1]`.
- Augmentation techniques:
  - Random rotation
  - Width and height shifts
  - Horizontal flips
  - Zooming


## Project Structure

```plaintext
Multiclass-Fish-Classification/
├── app.py                    # Streamlit web app for prediction
├── MobileNet_fish_model2.h5   # Trained MobileNet model file
├── new-fish-project-improved (1).ipynb # Jupyter notebook with full workflow
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── dataset/                  # Dataset used for training & testing
└── images/                   # Images for README and Streamlit app

```




 
## ⚙️ Approach

### 1️⃣ CNN Model
- **Architecture Summary:**
  - Convolutional layers with ReLU activation
  - MaxPooling layers for spatial reduction
  - Fully connected Dense layers
  - Softmax output for classification
- Trained from scratch to establish a baseline.

### 2️⃣ Transfer Learning Models
We used **ImageNet pre-trained models** for feature extraction:
- **MobileNet** ✅ (Best Performing Model)
- VGG16
- ResNet50
- InceptionV3
- EfficientNetB0

> Note: **No fine-tuning** was performed — only the top layers were trained while keeping the base model frozen.

### 3️⃣ Training Parameters
- **Epochs:** 20 (varied for different models)
- **Batch size:** 32
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** Categorical Crossentropy
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📊 Results

| Model          | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| -------------- | ------------ | ------------- | ---------- | ------------ |
| CNN (Scratch)  | 85.2         | 85.0          | 85.1       | 85.0         |
| VGG16          | 91.4         | 91.5          | 91.3       | 91.4         |
| ResNet50       | 93.1         | 93.0          | 93.2       | 93.1         |
| **MobileNet**  | **95.7**     | **95.6**      | **95.7**   | **95.6**     |
| InceptionV3    | 94.8         | 94.7          | 94.8       | 94.7         |
| EfficientNetB0 | 94.5         | 94.4          | 94.5       | 94.4         |


## 🖥️ Deployment
A **Streamlit application** was developed to allow users to upload a fish image and get real-time classification results from the trained model.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```


## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).




