# 🌱 Plant Disease Classification

An end-to-end project that leverages a dense Convolutional Neural Network (CNN) to classify plant leaf images into healthy or diseased categories. The project features an interactive Streamlit frontend where users can upload images, view disease prediction probabilities, analyze dominant colors, and receive treatment advice.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Results](#results)

---

## Overview
The **Plant Disease Classification** project utilizes a dense CNN model built with TensorFlow/Keras to automatically detect plant diseases from leaf images. The Streamlit-based frontend provides a user-friendly interface for real-time predictions, where users can:
- Upload a plant image.
- View the prediction probabilities for different diseases.
- See a pie chart of the dominant colors in the image.
- Get short treatment advice based on the prediction.

![image](https://github.com/user-attachments/assets/a5e07cc4-65bb-4c3f-9a88-21dc10a29eb7)

---

## Features
- **Dense CNN Model:**
  - Built with multiple convolutional layers, batch normalization, max pooling, and dropout layers.
  - Uses Global Average Pooling before the dense layers for robust feature extraction.
- **Real-Time Prediction:**
  - Upload an image and receive a probability distribution for various plant diseases.
- **Visualization:**
  - Generates a pie chart that displays the dominant colors in the image along with their percentage values.
- **Treatment Advice:**
  - Provides concise, actionable treatment recommendations based on the predicted disease.
- **Interactive Streamlit Frontend:**
  - Simple and intuitive dashboard for seamless user interaction.

---

## Technologies
- **Python**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy & Pandas**
- **Streamlit**
- **Matplotlib/Seaborn**

---

## Dataset
The dataset consists of plant leaf images categorized into:
- **Healthy**
- **Diseased** (various disease classes)

*Dataset Source:* [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## Sample Demo Features

When a user uploads a plant leaf image (e.g., `multi.jpeg`), the application provides:

1. **Image Preview**  
   - Displays the **original uploaded image** for reference.
  ![image](https://github.com/user-attachments/assets/e8dda7db-d328-4686-8ff0-90796903d7d0)


2. **Prediction & Disease Probabilities**  
   - Shows a **primary predicted disease** (e.g., `Corn_(maize)__healthy`).
   - Provides a **horizontal bar chart** indicating **top prediction probabilities** for all possible diseases.
  ![image](https://github.com/user-attachments/assets/0a3e88b1-8a82-4332-8aa5-38afa4da0609)


3. **Treatment Advice**  
   - Offers a **short recommendation** based on the predicted disease (e.g., “The corn is healthy. No treatment required.”).

4. **Color Analysis**  
   - Generates a **pie chart** depicting **dominant colors** in the uploaded image, along with **percentage values** for each color segment.
  ![image](https://github.com/user-attachments/assets/390bba2d-5ff6-4da4-ba40-2b0fbd09c685)

