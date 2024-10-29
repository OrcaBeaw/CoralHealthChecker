import numpy as np
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import tensorflow as tf

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model("my_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

model = load_model()


# Define class names
class_names = ['Bleached', 'Healthy']

# Streamlit App Title
st.title("Coral Health Classification App")
st.write("This app classifies coral images into two categories: **Bleached** and **Healthy**.")
st.write("It was developed using python, tensorflow, and streamlit.")
st.write("Feel free to upload your own coral images or paste the url of an image!")

# Input for Image URL OR upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
coral_url = st.text_input("Enter the URL of a coral image", "")

if uploaded_file is not None:
  if st.button("Classify Coral - Image Upload", key="upload_button"):
    img = Image.open(uploaded_file)
    img = img.resize((180, 180))
    st.image(img, caption="Uploaded Coral Image", use_column_width=True)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display image
    st.image(img, caption="Uploaded Coral Image", use_column_width=True)
    st.write(f"This image represents a **{class_names[np.argmax(score)]}** coral, with a **{100 * np.max(score):.2f}%** confidence.")


elif coral_url:
    try:
        # Download the image using requests
        response = requests.get(coral_url)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.resize((180, 180))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            # Display the image and classification result
            st.image(img, caption="Uploaded Coral Image", use_column_width=True)
            st.write(f"This image represents a **{class_names[np.argmax(score)]}** coral, with a **{100 * np.max(score):.2f}%** confidence.")
        else:
            st.write("Failed to retrieve the image. Please check the URL.")
    except Exception as e:
        st.write("Error loading model or classifying image:", e)
