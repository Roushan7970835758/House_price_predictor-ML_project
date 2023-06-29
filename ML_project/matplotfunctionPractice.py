from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Split the data into features (X) and target variable (y)
X = diabetes.data
y = diabetes.target

# Create a linear regression model
regression = linear_model.LinearRegression()

# Train the model
regression.fit(X, y)

# Make predictions
new_data = [[0.05, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]]
prediction = regression.predict(new_data)
print(diabetes.data)
print(prediction)