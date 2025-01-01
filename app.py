from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)
model = load_model('model.h5')  # Load your trained model

# Preprocess the image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Resize image
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Updated class labels
class_labels = {
    0: 'daisy',
    1: 'dandelion',
    2: 'rose',
    3: 'sunflower',
    4: 'tulip'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded!", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected!", 400

    # Save the uploaded image
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    target_size = (150, 150)  # Adjust according to your model's input size
    preprocessed_image = preprocess_image(file_path, target_size)

    # Make a prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Get the class label
    label = class_labels.get(predicted_class, "Unknown")

    return render_template('result.html', label=label, confidence=confidence, image_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)
