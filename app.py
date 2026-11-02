
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎"
)

st.title("🍎 Fruit Image Classifier")
st.write("Upload a picture to identify an apple, banana, or orange.")

# Class order must match the training order
CLASS_NAMES = ["apple", "banana", "orange"]

# Load the model once and reuse it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_cnn.keras")

try:
    model = load_model()
except Exception as error:
    st.error(
        "Could not load the trained model. "
        "Run train_cnn.py first and check that "
        "fruit_cnn.keras is in this project folder."
    )
    st.stop()

# Upload an image
uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png","webp","avif"]
)

if uploaded_file is not None:

    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Show the original image
    st.image(
        image,
        caption="Uploaded image",
        width=350
    )

    # Resize to the same size used during training
    resized_image = image.resize((128, 128))

    # Convert image into a NumPy array
    image_array = np.array(resized_image)

    # Add batch dimension: (128, 128, 3)
    # becomes (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the fruit
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_fruit = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    # Display prediction
    st.subheader(f"Prediction: {predicted_fruit.title()}")
    st.metric("Model confidence", f"{confidence:.2f}%")

    # Display scores for all classes
    st.write("Scores for each fruit:")

    for index, fruit in enumerate(CLASS_NAMES):
        score = float(probabilities[index])
        st.write(f"{fruit.title()}: {score * 100:.2f}%")
        st.progress(score)
