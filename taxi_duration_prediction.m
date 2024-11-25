% Load the dataset
data = readtable('nyc_taxi_trip_duration.csv');

% Convert pickup and dropoff times to datetime
data.pickup_datetime = datetime(data.pickup_datetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
data.dropoff_datetime = datetime(data.dropoff_datetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

% Calculate trip duration in minutes (already in seconds in dataset, so divide by 60)
data.trip_duration = data.trip_duration / 60; % Convert to minutes

%% Feature engineering
data.hour = hour(data.pickup_datetime); % Hour of the day
data.day_of_week = weekday(data.pickup_datetime); % Day of the week (1=Sunday, 7=Saturday)
data.passenger_count = categorical(data.passenger_count); % Convert to categorical if needed

% Calculate trip distance using Haversine formula
R = 6371; % Radius of Earth in kilometers
lat1 = deg2rad(data.pickup_latitude);
lat2 = deg2rad(data.dropoff_latitude);
lon1 = deg2rad(data.pickup_longitude);
lon2 = deg2rad(data.dropoff_longitude);
dlat = lat2 - lat1;
dlon = lon2 - lon1;
a = sin(dlat/2).^2 + cos(lat1) .* cos(lat2) .* sin(dlon/2).^2;
c = 2 * atan2(sqrt(a), sqrt(1-a));
data.trip_distance = R * c; % Distance in kilometers

% Convert categorical features to numerical values
data.passenger_count = double(data.passenger_count); % Convert passenger_count to double

% Select features and target variable
features = {'hour', 'day_of_week', 'passenger_count', 'trip_distance'};
X = [data{:, {'hour', 'day_of_week', 'passenger_count'}}, data.trip_distance];
y = data.trip_duration;

%% Split data into training and testing sets (80% training, 20% testing)
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

%% Build a Linear Regression Model
mdl_lr = fitlm(X_train, y_train);

% Make predictions on the test set
y_pred_lr = predict(mdl_lr, X_test);

% Evaluate model performance for Linear Regression
mae_lr = mean(abs(y_pred_lr - y_test));
rmse_lr = sqrt(mean((y_pred_lr - y_test).^2));
r2_lr = 1 - sum((y_pred_lr - y_test).^2) / sum((y_test - mean(y_test)).^2);

% Display evaluation metrics for Linear Regression
fprintf('Linear Regression Model Evaluation:\n');
fprintf('MAE: %.2f minutes\n', mae_lr);
fprintf('RMSE: %.2f minutes\n', rmse_lr);
fprintf('R^2: %.2f\n', r2_lr);

% Visualize actual vs predicted values for Linear Regression
figure;
scatter(y_test, y_pred_lr, 'b');
xlabel('Actual Duration (minutes)');
ylabel('Predicted Duration (minutes)');
title('Linear Regression: Actual vs Predicted');

% Plot residuals for Linear Regression
figure;
plot(y_pred_lr, y_pred_lr - y_test, 'o');
xlabel('Predicted Duration (minutes)');
ylabel('Residuals');
title('Linear Regression: Residuals');

%% Build a Support Vector Regression (SVR) Model

% Preprocess the Data (Scaling)
% Support Vector Regression is sensitive to the scale of the features,
% so we will standardize the features to have zero mean and unit variance.

% Compute the mean and standard deviation of the training data
mu = mean(X_train);  % Mean of the training data
sigma = std(X_train);  % Standard deviation of the training data

% Standardize training and test data using the computed mean and standard deviation
X_train_scaled = (X_train - mu) ./ sigma;  % Standardize training data
X_test_scaled = (X_test - mu) ./ sigma;  % Standardize test data using training statistics

% Train Support Vector Regression (SVR) Model
% Using a Radial Basis Function (RBF) kernel
svm_model = fitrsvm(X_train_scaled, y_train, 'KernelFunction', 'gaussian', 'Standardize', true, 'Solver', 'SMO');

% Make Predictions
y_pred_svr = predict(svm_model, X_test_scaled);

% Evaluate Model Performance
mae_svr = mean(abs(y_pred_svr - y_test));  % Mean Absolute Error
rmse_svr = sqrt(mean((y_pred_svr - y_test).^2));  % Root Mean Squared Error
r2_svr = 1 - sum((y_pred_svr - y_test).^2) / sum((y_test - mean(y_test)).^2);  % R-squared

% Display Evaluation Metrics
fprintf('\nSupport Vector Regression Model Evaluation:\n');
fprintf('MAE: %.2f minutes\n', mae_svr);
fprintf('RMSE: %.2f minutes\n', rmse_svr);
fprintf('R^2: %.2f\n', r2_svr);

% Visualize Actual vs Predicted Values
figure;
scatter(y_test, y_pred_svr, 'r');
xlabel('Actual Duration (minutes)');
ylabel('Predicted Duration (minutes)');
title('SVR: Actual vs Predicted');

% Plot Residuals for SVR
figure;
plot(y_pred_svr, y_pred_svr - y_test, 'o');
xlabel('Predicted Duration (minutes)');
ylabel('Residuals');
title('SVR: Residuals');