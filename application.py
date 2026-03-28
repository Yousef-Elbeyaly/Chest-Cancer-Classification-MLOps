import os
import sys
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)
app = application

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        pipeline = PredictPipeline()
        result = pipeline.predict(file_path)

        return render_template('index.html', prediction=result, image_path = file_path)
    
    except Exception as e:
        logging.info(CustomException(e, sys))
        return f"Something went wrong: {str(e)}"
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)