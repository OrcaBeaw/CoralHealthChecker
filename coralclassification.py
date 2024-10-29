import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("my_model.keras")  # Make sure the file is in the same directory
    return model

model = load_model()

# Define class names
class_names = ['Bleached', 'Healthy']

# Streamlit App Title
st.title("Coral Health Classification App")

# Allow user to upload an image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_file)
    img = img.resize((180, 180))  # Resize to match model input size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Display the uploaded image
    st.image(img, caption="Uploaded Coral Image", use_column_width=True)

    if st.button("Classify Coral"):
        # Make predictions
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Display the result
        st.write(f"This image most likely belongs to **{class_names[np.argmax(score)]}** with a **{100 * np.max(score):.2f}%** confidence.")
