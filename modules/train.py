import pickle
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

datafiles = ["./data/data-" + str(i+1).zfill(2) +
             ".csv" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

data = None
for idx, datafile in enumerate(datafiles):
    new_data = pd.read_csv(datafile, parse_dates=[0], infer_datetime_format=True, usecols=["tpep_pickup_datetime",  "trip_distance", 
                                                                                           "payment_type", "fare_amount", "tip_amount", "total_amount"])
    # Reject meaningless data
    new_data = new_data.dropna()
    new_data = new_data[new_data['tip_amount'] >= 0]
    new_data = new_data[new_data['fare_amount'] >= 0]


    new_data['pickup_month'] = new_data['tpep_pickup_datetime'].dt.month
    # There's some bad data for each month
    new_data = new_data[new_data['pickup_month'] == (idx + 1)]
    
    new_data['pickup_hour'] = new_data['tpep_pickup_datetime'].dt.hour
    new_data['pickup_day_of_week'] = new_data['tpep_pickup_datetime'].dt.day_of_week
    
    # Drop non-number column
    new_data = new_data.drop(columns=["tpep_pickup_datetime"])
    
    # Concatenate datasets one by one
    if data is None:
        data = new_data
    else:
        data = pd.concat([data, new_data], ignore_index=True)

# split 70%/30%
x_train, x_test = train_test_split(data, test_size=0.3)
y_train, y_test = x_train['tip_amount'], x_test['tip_amount']

# Drop column we're predicting from the dataset
del x_train['tip_amount']
del x_test['tip_amount']

# Normalize
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_test = np.array(x_test)
y_test = np.array(y_test)

# Run the regression
clf_ = SGDRegressor()
clf_.fit(x_train, y_train)

# Print and save errors
with open(f'./results/predict-final.txt', 'w') as f:
    mean_error = mean_squared_error(y_test, clf_.predict(x_test))
    abs_error = mean_absolute_error(y_test, clf_.predict(x_test))
    mean_error_str = f"Mean Squared Error: {mean_error}"
    abs_error_str = f'Mean Absolute Error: {abs_error}'
    print(mean_error_str)
    print(abs_error_str)
    f.write(mean_error_str + "\n")
    f.write(abs_error_str + "\n")


# Save the model and the scaler
model_filename = f"./results/predict-model-final.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(clf_, file)

scaler_filename = f"./results/predict-scaler-final.pkl"
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

test_data_sample = data.sample(100)
test_data_sample.to_csv('./results/test.csv')