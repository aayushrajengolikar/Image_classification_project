# app_classifier.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# Load Classifier Model
# -----------------------------
st.title("Image Classification App")
classifier_model = load_model("best_image_classifier.keras")  # updated path

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img: Image.Image, target_size=(256, 256)):
    img = img.resize(target_size)
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    st.write("Processing classification...")
    img_array = preprocess_image(image)
    prediction = classifier_model.predict(img_array)[0][0]

    if prediction >= 0.5:
        predicted_class = "Drone"
    else:
        predicted_class = "Bird"

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {prediction*100:.2f}%")
