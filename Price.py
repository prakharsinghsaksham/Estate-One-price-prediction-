# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (change path as per your dataset location)
data = pd.read_csv('path_to_your_dataset.csv')

# Data preprocessing and feature selection (example)
# Assuming 'SalePrice' is the target variable
# Select relevant features for training
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# Prepare data
X = data[features]
y = data['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2)  # Adjust degree as needed
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train_poly, y_train)

# Predictions
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction
# Assuming new_data contains features in the same format as X
new_data = pd.DataFrame([[8, 2000, 2, 1000, 2, 1990]], columns=features)
new_data_poly = poly.transform(new_data)
predicted_price = model.predict(new_data_poly)
print(f"Predicted Price: ${predicted_price[0]}")
