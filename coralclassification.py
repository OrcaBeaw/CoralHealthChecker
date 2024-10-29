import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO


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

# Selection for image upload method
option = st.radio("Choose image input method:", ("Upload Image", "Link Image URL"))

# If the user chooses to upload an image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and preprocess the uploaded image
        img = Image.open(uploaded_file)
        img = img.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        # Display the uploaded image
        st.image(img, caption="Uploaded Coral Image", use_column_width=True)

        # Classify the uploaded image
        if st.button("Classify Coral"):
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            # Display the classification result
            st.write(
                f"This image most likely belongs to **{class_names[np.argmax(score)]}** with a **{100 * np.max(score):.2f}%** confidence.")

# If the user chooses to provide a URL
elif option == "Link Image URL":
    coral_url = st.text_input("Enter the URL of a coral image",
                              "https://i0.wp.com/sitn.hms.harvard.edu/wp-content/uploads/2021/05/coral-bleaching.jpeg?resize=1500%2C768&ssl=1")

    if st.button("Classify Coral"):
        try:
            # Download and preprocess the image from the URL
            response = requests.get(coral_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.resize((180, 180))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

                # Display the linked image
                st.image(img, caption="Coral Image from URL", use_column_width=True)

                # Classify the image from the URL
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                # Display the classification result
                st.write(
                    f"This image most likely belongs to **{class_names[np.argmax(score)]}** with a **{100 * np.max(score):.2f}%** confidence.")
            else:
                st.write("Failed to retrieve the image. Please check the URL.")
        except Exception as e:
            st.write("An error occurred while loading the image:", e)