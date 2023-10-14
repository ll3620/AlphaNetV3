# AlphaNetV3_PyTorch

### Description
This code provides a comprehensive framework for preparing, training, and validating a neural network on financial data, with an emphasis on time series. It includes periodic retraining, model evaluation, and prediction generation.

### Workflow:
#### Data Preparation: 
* Load the data using the DataPreparation class, which takes in the financial data dictionary and a mask.
* Initialize an empty data frame, predicted_df, to store predictions for future timestamps.

#### Training Schedule Calculation:
* Calculate the index first_train_idx to determine when the first training should occur. The training starts after the 30th day from the start date and on either the 10th, 20th, or 30th of the month.
