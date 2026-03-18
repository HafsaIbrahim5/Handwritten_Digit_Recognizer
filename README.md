# 🔢 MNIST CNN Digit Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-D00000?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-99.7%25-brightgreen?style=for-the-badge)

A **professional interactive web app** for handwritten digit recognition using a **Convolutional Neural Network** (CNN) built with TensorFlow/Keras, deployed with Streamlit.

Based on the famous Kaggle notebook: *"Introduction to CNN Keras — 0.997 accuracy (Top 8%)"*

[🚀 Live Demo](#) · [📓 Original Notebook](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook)

</div>

---

## 📸 App Preview

| Draw & Predict | Training Analytics | Error Analysis |
|---|---|---|
| ✍️ Interactive canvas | 📈 Accuracy/Loss curves | 🔍 Hardest mistakes |
| 🎯 Real-time prediction | 🧩 Confusion matrix | 📊 Classification report |

---

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
[Conv2D(32, 5×5, ReLU) → Conv2D(32, 5×5, ReLU) → MaxPool2D(2×2) → Dropout(0.25)]
    ↓
[Conv2D(64, 3×3, ReLU) → Conv2D(64, 3×3, ReLU) → MaxPool2D(2×2) → Dropout(0.25)]
    ↓
Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(10, Softmax)
```

**~1.6M parameters · RMSprop optimizer · Categorical Crossentropy loss**

---

## 🔑 Key Techniques

| Technique | Purpose |
|---|---|
| **Data Augmentation** | rotation±10°, zoom±10%, shift±10% — prevents overfitting |
| **ReduceLROnPlateau** | Halves LR when val_accuracy plateaus for 3 epochs |
| **EarlyStopping** | Stops training if no improvement for 5 epochs |
| **Dropout (0.25, 0.5)** | Regularization to reduce overfitting |
| **One-Hot Encoding** | Labels encoded for categorical crossentropy |

---

## 🚀 Features

- ✍️ **Draw & Predict** — Interactive canvas, draw a digit and get instant CNN prediction
- 📤 **Upload Image** — Upload PNG/JPG and classify it
- 🎲 **Random Samples** — Test on random MNIST images with ✅/❌ feedback
- 📈 **Training Analytics** — Accuracy/Loss curves, LR schedule
- 🧩 **Confusion Matrix** — Full heatmap with per-class breakdown
- 📊 **Classification Report** — Precision, Recall, F1-Score per digit
- 🔍 **Error Analysis** — Most confident mistakes + confused digit pairs
- 🏗️ **Model Architecture** — Layer-by-layer visualization + model summary
- 🔬 **Dataset Explorer** — Class distribution, sample grid, pixel histogram

---

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/HafsaIbrahim5/mnist-cnn-classifier.git
cd mnist-cnn-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

> **Note:** On first run, TensorFlow downloads MNIST (~11 MB) and trains the model (~1–2 min). Results are cached for all subsequent runs.

---

## 📦 Dependencies

```
tensorflow >= 2.10
streamlit >= 1.28
streamlit-drawable-canvas >= 0.9
numpy, pandas, matplotlib, seaborn, scikit-learn, Pillow
```

---

## 📁 Project Structure

```
mnist-cnn-classifier/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── introduction-to-cnn-keras_      # Original Jupyter notebook
    using_Mnist_dataset.ipynb
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **~99.7%** |
| Architecture | 5-layer CNN |
| Training Data | 54,000 images |
| Validation Data | 6,000 images |
| Test Data | 10,000 images |

---

## 👩‍💻 Author

**Hafsa Ibrahim** — AI & Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/HafsaIbrahim5)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<div align="center">
  ⭐ <strong>Star this repo if you found it helpful!</strong> ⭐
</div>
