# 😄 Face Detection to Emoji Project

## 📌 Project Overview
This project detects a human face using a webcam in real-time, analyzes facial expressions using a deep learning CNN model, and displays a matching emoji based on the detected emotion. It also shows the predicted emotion name on a Tkinter GUI.

## ✨ Features
- Real-time face detection using webcam  
- Emotion recognition using a CNN model (Keras/TensorFlow)  
- Emoji display based on detected emotion  
- GUI interface built with Tkinter  
- Supports 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised  

## 🧠 Technologies Used
- Python  
- OpenCV (Face detection)  
- TensorFlow / Keras (Deep Learning model)  
- NumPy  
- Tkinter (GUI)  
- PIL (Image processing)  

## 🧩 Model Architecture
A Convolutional Neural Network (CNN) is used with:
- Conv2D layers  
- MaxPooling layers  
- Dropout layers  
- Dense fully connected layers  
- Softmax output for 7 emotion classes  

## ⚙️ How It Works
1. Webcam captures live video  
2. Haar Cascade detects face  
3. Image is preprocessed (48x48 grayscale)  
4. CNN model predicts emotion  
5. Matching emoji is displayed in GUI  

## 🚀 Installation

### 1. Clone repository
```bash
git clone <your-repo-link>
cd face-emoji-project
