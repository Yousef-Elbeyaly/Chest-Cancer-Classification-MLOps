import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.h5")

class Model_trainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
    
    def get_vgg16_model(self, num_classes):
        try:
            logging.info("Start using and modifing VGG16")

            base_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

            for layer in base_model.layers:
                layer.trainable = False
            
            x = Flatten()(base_model.output)
            prediction = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=base_model.input, outputs = prediction)
            model.compile(
                optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self, train_generator, valid_generator):
        try:
            logging.info("Start training VGG16")

            num_classes = len(train_generator.class_indices)

            model = self.get_vgg16_model(num_classes)

            logging.info("Start fitting")

            model.fit(
                train_generator,
                validation_data = valid_generator,
                epochs = 10,
                steps_per_epoch = len(train_generator),
                validation_steps = len(valid_generator)
            )

            os.makedirs(os.path.dirname(self.model_config.trained_model_file_path), exist_ok=True)

            model.save(self.model_config.trained_model_file_path)
            logging.info(f"Model saved in {self.model_config.trained_model_file_path}")

            return self.model_config.trained_model_file_path
        
        except Exception as e:
            raise CustomException(e, sys)