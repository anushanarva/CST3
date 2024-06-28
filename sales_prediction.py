import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Load the dataset (replace 'path_to_your_dataset.csv' with the actual path to your dataset)
data_path = 'C:\\Users\\hp\\Downloads\\advertising.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print(data.head())

# Get basic information about the dataset
print(data.info())

# Get summary statistics of the dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Drop any missing values (if necessary)
data = data.dropna()

# Define the feature columns and the target column
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate and print the R-squared value
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Example of making predictions on new data
new_data = pd.DataFrame({'TV': [100, 200], 'Radio': [20, 30], 'Newspaper': [10, 15]})
predictions = model.predict(new_data)
print(predictions)
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Sales'])
predictions_df.to_csv('predicted_sales.csv', index=False)