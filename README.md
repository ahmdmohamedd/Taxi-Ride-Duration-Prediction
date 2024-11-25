# Taxi Ride Duration Prediction

This MATLAB system predicts the duration of a taxi ride based on various features such as passenger count, pickup and dropoff locations, and time of day. The system uses regression models like Linear Regression and Support Vector Regression (SVR) to perform the prediction.

## Features
- **Data Preprocessing**: Handles missing values and scales the features for model training.
- **Regression Models**: Implements Linear Regression and Support Vector Regression (SVR).
- **Model Evaluation**: Evaluates the model's performance using MAE, RMSE, and R² metrics.
- **Visualization**: Plots actual vs predicted durations and residuals to assess model accuracy.

## Dataset
The system uses a dataset containing taxi ride details, such as:
- Pickup and dropoff locations (latitude and longitude)
- Pickup and dropoff timestamps
- Passenger count
- Trip duration (target variable)

You can replace this dataset with any other dataset that includes similar features for taxi ride duration prediction.

## Usage
1. Clone the repository or download the files.
2. Ensure you have MATLAB installed with the necessary toolboxes.
3. Run the `taxi_duration_prediction.m` script to preprocess the data, train the regression models, and evaluate the results.

```matlab
run('taxi_duration_prediction.m')
```

## Requirements
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox (for SVR and regression models)

## Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average of the absolute differences between predicted and actual values.
- **RMSE (Root Mean Squared Error)**: Square root of the average squared differences.
- **R² (Coefficient of Determination)**: Measures the proportion of variance explained by the model.

---
