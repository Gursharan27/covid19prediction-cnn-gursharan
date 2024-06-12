#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Title of the app
st.title('COVID-19 Prediction using CNN')

# Sidebar for user input
st.sidebar.header('Upload X-ray Image')
uploaded_file = st.sidebar.file_uploader("Choose an X-ray image...", type="jpg")

# Load your model
@st.cache(allow_output_mutation=True)
def load_cnn_model():
    model = load_model('model.keras')  # Provide the path to your model file
    return model

model = load_cnn_model()

# Preprocess the uploaded image
def preprocess_image(image):
    image = Image.open(image).convert('L')
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Make prediction
def make_prediction(image, model):
    prediction = model.predict(image)
    return prediction

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    st.image(image[0], caption='Uploaded X-ray Image', use_column_width=True)
    
    prediction = make_prediction(image, model)
    st.write(f"Prediction: {'Positive' if prediction[0][0] > 0.5 else 'Negative'}")


# In[ ]:




