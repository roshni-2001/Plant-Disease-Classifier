import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Absolute paths to the model and class indices
model_path = r'F:\Desktop\plant-disease-prediction-cnn-deep-leanring-project-main\model_training_notebook\drive\MyDrive\Youtube\trained_models\plant_disease_prediction_model.h5'
class_indices_path = r'F:\Desktop\plant-disease-prediction-cnn-deep-leanring-project-main\model_training_notebook\class_indices.json'

# Print the paths to ensure they're correctly specified
print(f"Model path: {model_path}")
print(f"Class indices path: {class_indices_path}")

# Verify the model and class indices files exist
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.isfile(class_indices_path):
    raise FileNotFoundError(f"Class indices file not found at {class_indices_path}")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Save the uploaded image to a temporary file
            temp_image_path = 'temp_image.png'
            image.save(temp_image_path)
            # Predict the class of the uploaded image
            prediction = predict_image_class(model, temp_image_path, class_indices)
            # Remove the temporary file
            os.remove(temp_image_path)
            # Display the prediction
            st.success(f'Prediction: {str(prediction)}')
