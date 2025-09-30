import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cats_vs_dogs_cnn.h5")
    return model

model = load_model()

# Streamlit App UI
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image, and the CNN model will predict whether it's a **Cat** or a **Dog**.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img = img.resize((128, 128))  # same size as training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    pred_prob = model.predict(img_array)[0][0]
    pred_label = "🐶 Dog" if pred_prob > 0.5 else "🐱 Cat"
    
    st.subheader(f"Prediction: {pred_label}")
    st.write(f"Confidence: {pred_prob:.2f}")
