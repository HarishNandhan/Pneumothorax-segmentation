import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load the pre-trained model
model = load_model("Pneumothorax_classification.h5")

# Define the labels for the binary classification
labels = ["No Pneumothorax", "Pneumothorax"]

# Set up the Streamlit app
st.title("Pneumothorax Detection")
st.write("Upload an image to see if it contains Pneumothorax")

# Allow the user to upload an image
file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Make a prediction if an image is uploaded
if file is not None:
    # Read the image and preprocess it
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # For display purposes only
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    # Make a prediction and display the result
    pred = model.predict(img)[0][0]
    prob = round(pred * 100, 2)
    label = labels[1] if prob >= 99 else labels[0]
    st.write("Prediction: " + label)
    st.write("Probability: " + str(prob) + "%")

    # Display the original image
    st.write("Original Image:")
    st.image(img_disp, channels="BGR")

