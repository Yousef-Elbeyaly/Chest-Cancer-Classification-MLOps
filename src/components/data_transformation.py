import os
import sys
from dataclasses import dataclass
import tensorflow as tf
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    IMAGE_SIZE : tuple = (224, 224)
    BATCH_SIZE : int = 32

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Start preparing Image Data Generator")
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            
            )

            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

            return train_datagen, test_datagen
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, valid_path):
        try:
            logging.info("Start transform images to batches")
            train_datagen, test_datagen = self.get_data_transformation_object()

            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size = self.data_transformation_config.IMAGE_SIZE,
                batch_size = self.data_transformation_config.BATCH_SIZE,
                class_mode = 'categorical'
            )
            valid_generator = test_datagen.flow_from_directory(
                valid_path,
                target_size=self.data_transformation_config.IMAGE_SIZE,
                batch_size=self.data_transformation_config.BATCH_SIZE,
                class_mode='categorical'
            )
            test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=self.data_transformation_config.IMAGE_SIZE,
                batch_size=self.data_transformation_config.BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
            logging.info("Train, Valid, and Test generators created successfully")
            
            return (
                train_generator,
                valid_generator,
                test_generator
            )
        except Exception as e:
            raise CustomException(e, sys)