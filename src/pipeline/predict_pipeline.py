import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.h5")
        self.labels_path = os.path.join("artifacts", "class_labels.pkl")
        
        self.model = load_model(self.model_path)
        self.model_labels = load_object(self.labels_path)
        self.inv_labels = {v: k for k, v in self.model_labels.items()}
    
    def predict(self, image_path):
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = self.model.predict(img_array)
            prediction_class_index = np.argmax(prediction, axis=1)[0]
            
            result = self.inv_labels[prediction_class_index]
            return result
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    test_image = input("Enter image path: ")
    pipeline = PredictPipeline()
    print(f"Prediction: {pipeline.predict(test_image)}")