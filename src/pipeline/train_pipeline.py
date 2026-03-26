import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Model_trainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            ingestion = DataIngestion()

            train_path, test_path, valid_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()

            train_gen, valid_gen, test_gen = transformation.initiate_data_transformation(
                train_path, test_path, valid_path
            )

            trainer = Model_trainer()

            model_path = trainer.initiate_model_trainer(train_gen, valid_gen)

            return model_path
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()