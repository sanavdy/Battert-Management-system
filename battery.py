import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\\Users\\sanam\\Downloads\\archive (3)\\Battery_RUL.csv")
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
target_col = 'RUL'
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)',
            'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)',
            'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']
X = df[features].values
y = df[target_col].values
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)


corrmat = df.corr()
target = df['RUL']
features = df.drop(['RUL'], axis=1)
features = features.drop(['Cycle_Index'], axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

features_std = pd.DataFrame(features_std, columns = features.columns)
from sklearn.model_selection import (train_test_split, StratifiedKFold)

X_train, X_test, y_train, y_test = train_test_split(features_std, target, test_size=0.2, random_state=2301)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
target_column = 'RUL'
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)',
            'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']

# Extract features (X) and target variable (y)
X = df[features]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2301)

# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=2301, n_estimators=100)

# Fit the model to the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_test_pred_rf = rf_regressor.predict(X_test)

# Calculate RMSE on the testing set
rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with your data
# Replace 'your_target_column' with the actual column name containing RUL
# Replace 'your_features' with the actual column names containing the features
target_column = 'RUL'
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)',
            'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']

# Extract features (X) and target variable (y)
X = df[features]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2301)

# Initialize the DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=2301)

# Fit the model to the training data
dt_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_test_pred_dt = dt_regressor.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
data=df.drop(['Cycle_Index','Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Time constant current (s)','Charging time (s)'],axis=1)
X = data.drop(['RUL'], axis=1)
y = data['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023, shuffle =True)
class Pipeline:
    def __init__(self, scalar):
        self.scalar = scalar

    def fit(self, X, y):
        X = self.scalar.fit_transform(X)
        return X, y

    def transform(self, X, y):
        X = self.scalar.transform(X)
        return X, y
    
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as pl
import warnings
# from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import roc_curve, auc
robust = RobustScaler()
pipeline = Pipeline(robust)
X_train, y_train = pipeline.fit(X_train, y_train)
X_test, y_test = pipeline.transform(X_test, y_test)
random_forest = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, scoring='r2', cv=5)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# Assuming you have X_train, X_test, y_train, y_test defined
# Define the RandomForestRegressor
random_forest = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [None,3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
best_model = grid_search.best_estimator_

# Make predictions on the training set
y_train_pred = best_model.predict(X_train)

# Make predictions on the test set
y_test_pred = best_model.predict(X_test)

# Calculate RMSE for training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
threshold = 10  # Example threshold for classification
y_test_class = np.where(y_test <= threshold, 1, 0)
y_test_pred_class = np.where(y_test_pred <= threshold, 1, 0)
accuracy = accuracy_score(y_test_class, y_test_pred_class)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate additional regression metrics
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
feature_importances = best_model.feature_importances_
df = df.dropna()
X = df.drop('RUL', axis=1)
y = df['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rfr = RandomForestRegressor(random_state=42, n_estimators=100)
rfr.fit(X_train_scaled, y_train)

y_pred = rfr.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
feature_importances = rfr.feature_importances_
cycle_index = float(input("Enter Cycle Index: "))
discharge_time = float(input("Enter Discharge Time (s): "))
decrement_3_6_3_4V = float(input("Enter Decrement 3.6-3.4V (s): "))
max_voltage_discharge = float(input("Enter Max. Voltage Discharge (V): "))
min_voltage_charge = float(input("Enter Min. Voltage Charge (V): "))
time_at_415v = float(input("Enter Time at 4.15V (s): "))
time_constant_current = float(input("Enter Time Constant Current (s): "))
charging_time = float(input("Enter Charging Time (s): "))

st.header("Volt Armour")
cycle_index = st.number_input("Enter Cycle Index: ")
discharge_time = st.number_input("Enter Discharge Time (s): ")

user_input = np.array([[cycle_index, discharge_time, decrement_3_6_3_4V,
                        max_voltage_discharge, min_voltage_charge, time_at_415v,
                        time_constant_current, charging_time]])

# Scale the user input using the same scaler used during training
user_input_scaled = scaler.transform(user_input)

# Make predictions using the trained model
predicted_rul = rfr.predict(user_input_scaled)

# Display the predicted RUL
print("Predicted RUL:", predicted_rul[0])