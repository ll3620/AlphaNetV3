# AlphaNetV3_PyTorch

### Description
This code provides a comprehensive framework for preparing, training, and validating a neural network on financial data, with an emphasis on time series. It includes periodic retraining, model evaluation, and prediction generation.

### Workflow:
#### Data Preparation: 
* Load the data using the `DataPreparation` class, which takes in the financial data dictionary and a mask.
* Initialize an empty data frame, `predicted_df`, to store predictions for future timestamps.

#### Training Schedule Calculation:
* Calculate the index `first_train_idx` to determine when the first training should occur. The training starts after the 30th day from the start date and on either the 10th, 20th, or 30th of the month.

#### Periodic Model Training
Retrain the model at specific times (10th, 20th, 30th of the month). At each retraining event:
* Data is prepared into training and validation sets.
* The model is instantiated and trained using the `ModelTrainer` class, which handles everything from data ingestion to model evaluation.
* After training, the model is ready for making predictions.

#### Service Routine:
* Periodically, the trained model is used to generate predictions (or "service" the predictions). These predictions are stored in `predicted_df`.

#### Progress Tracking: 
* Utilize the `tqdm` library to visually track the progress of the service routine until the next training event.

### Key Classes and Methods:
`DataPreparation`: Processes raw data into a format suitable for model training and validation.

`ModelTrainer`: Contains the entire lifecycle of the model: from initialization and training to validation and service/prediction.

`DataSet`: Used to create PyTorch datasets for training and validation.

`DataLoader`: Standard PyTorch class for loading data in batches.

