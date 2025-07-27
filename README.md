# 🧠 Simple RNN Sentiment Classifier

A lightweight and interactive sentiment analysis web app using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow and deployed via **Streamlit**.

🔗 **Live Demo**: [Try the App Here](https://simplernn-impletation-cv74zuyjkgfbgzw2c6cqwy.streamlit.app/)

---

## 📌 Features

- ✅ Sentiment classification (+ve or -ve) for custom text input
- ✅ Trained using IMDB movie reviews dataset
- ✅ Built with TensorFlow / Keras
- ✅ Deployed via Streamlit Cloud
- ✅ Lightweight, fast, and easy to use

---

## 📂 Project Structure
.
├── simpleRnn.ipynb        # Jupyter Notebook with model training & evaluation
├── main.py                # Streamlit app source code
├── model.h5               # Trained SimpleRNN model
├── tokenizer.pkl          # Tokenizer used for preprocessing
├── requirements.txt       # Python dependencies
└── README.md              # You’re here

---

## 🚀 How It Works

1. **Preprocessing**: Text is tokenized and padded using a Keras tokenizer.
2. **Model**: A simple RNN-based binary classifier is used.
3. **Prediction**: The model outputs a probability which is classified as:
   - `+ve` if > 0.5
   - `-ve` otherwise

---

## 📊 Model Architecture

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=max_length),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])
