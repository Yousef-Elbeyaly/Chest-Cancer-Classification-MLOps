import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('Data', 'train')
    test_data_path : str = os.path.join('Data', 'test')
    valid_data_path : str = os.path.join('Data', 'valid')
    artifacts_path : str = os.path.join('artifacts')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            os.makedirs(self.ingestion_config.artifacts_path, exist_ok=True)
            for path in [self.ingestion_config.train_data_path,
                         self.ingestion_config.test_data_path,
                         self.ingestion_config.valid_data_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Folder not found at {path}")
            
            logging.info("Folders Found")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.valid_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_path, test_path, valid_path = obj.initiate_data_ingestion()
        print("Success!")
    except Exception as e:
        print("Fail!")