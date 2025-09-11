from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from utils.utils import disease_info
import logging

# Optional: gdown for Google Drive download
import gdown

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif', 'heic'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------
# MODEL LOADING LOGIC
# ---------------------

# Google Drive File ID for the model
GOOGLE_DRIVE_FILE_ID = "10y_AMIwpk7Z5CZxXup8FpJ0rPzcI_fU-" #Replace with Your 
MODEL_PATH = "model/crop_disease_model.h5" #Replace with Your
os.makedirs("model", exist_ok=True)

def download_model_from_drive():
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    print(f"Downloading model from Google Drive: {url}")
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download complete!")

# --- Mode 1: Local-first, then download if missing (for production) ---
if os.path.exists(MODEL_PATH):
    print("Loading model from local storage...")
else:
    print("Local model not found, downloading from Google Drive...")
    download_model_from_drive()

model = load_model(MODEL_PATH)

# --- Mode 2: Always download from Google Drive (for Colab) ---
# Uncomment this section for Colab use:
# NoteBookLink In Readme
# download_model_from_drive()
# model = load_model(MODEL_PATH)

# Class labels from disease_info dictionary
class_labels = list(disease_info.keys())

# ---------------------
# HELPER FUNCTIONS
# ---------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------
# API ROUTE
# ---------------------
@app.route('/analysis', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            app.logger.debug(f"Image shape: {img_array.shape}")

            prediction = model.predict(img_array)
            app.logger.debug(f"Prediction array: {prediction}")
            predicted_index = np.argmax(prediction)

            if predicted_index >= len(class_labels):
                raise ValueError("Predicted index out of range")

            predicted_class = class_labels[predicted_index]
            app.logger.info(f"Predicted class: {predicted_class}")

            info = disease_info.get(predicted_class, {
                "name_en": "Unknown",
                "name_hi": "अज्ञात",
                "cause": "Not found",
                "symptoms": "Not found",
                "suggestion": "No suggestion available"
            })

            return jsonify({
                "prediction": predicted_class,
                "info": info,
                "image_url": file_path
            })

        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only JPG, PNG, JPEG, etc. are allowed."}), 400

# ---------------------
# RUN APP
# ---------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8082))
    logging.info(f"Running on port: {port}")
    app.run(host="0.0.0.0", port=port)

