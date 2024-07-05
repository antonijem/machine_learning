import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import pickle

# Function to perform one-hot encoding using unique values from the entire dataset
def one_hot_encode_with_unique_values(df, column, unique_values):
    one_hot_matrix = np.zeros((df.shape[0], len(unique_values)))
    for i, unique_value in enumerate(unique_values):
        one_hot_matrix[:, i] = (df[column] == unique_value).astype(float)
    return one_hot_matrix

# Connect to the database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    passwd='1234',
    database='psz2'
)

cursor = conn.cursor()
query = """
select price, author, category, publisher, year, pages, cover, format
from books_tb
"""
cursor.execute(query)
results = cursor.fetchall()

# Convert to DataFrame
columns = ['price', 'author', 'category', 'publisher', 'year', 'pages', 'cover', 'format']
df_ceo = pd.DataFrame(results, columns=columns)

# Collect unique values for all categorical features
categorical_features = ['author', 'category', 'publisher', 'cover', 'format']
unique_values_dict = {col: df_ceo[col].unique() for col in categorical_features}

# Sample 70% of the data for training
df_train = df_ceo.sample(frac=0.7, random_state=10)

# Perform one-hot encoding using unique values from the entire dataset
encoded_train_features = np.hstack([one_hot_encode_with_unique_values(df_train, col, unique_values_dict[col]) for col in categorical_features])

# Normalize numeric features
numeric_features = ['year', 'pages']
numeric_train_data = df_train[numeric_features].values
mean_numeric = np.mean(numeric_train_data, axis=0)
std_numeric = np.std(numeric_train_data, axis=0)
normalized_train_numeric_data = (numeric_train_data - mean_numeric) / std_numeric

# Combine all features
X_train = np.hstack([encoded_train_features, normalized_train_numeric_data])
y_train = df_train['price'].values

# Normalize the target variable
mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)
y_train_normalized = (y_train - mean_y_train) / std_y_train

# Add intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Gradient descent
learning_rate = 1
iterations = 10000
N = y_train.size
coeff = np.zeros(X_train.shape[1])

def gradientDescend(x, y, coeff, iterations, learning_rate):
    past_costs = []
    past_coeff = [coeff]
    for i in range(iterations):
        prediction = np.dot(x, coeff)
        error = prediction - y
        cost = 1/(2*N)*np.dot(error.T, error)  # squared error cost
        past_costs.append(cost)
        der = (1/N)*learning_rate*np.dot(x.T, error)
        coeff = coeff - der
        past_coeff.append(coeff)
        if i % (iterations // 10) == 0:
            print(f"Iteracija: {i}, cena: {cost}")
    return past_coeff, past_costs

past_coeff, past_costs = gradientDescend(X_train, y_train_normalized, coeff, iterations, learning_rate)
coeff = past_coeff[-1]

print("Final coefficients: ", coeff)

# Plot the cost history
plt.plot(past_costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Cost over Iterations')
plt.savefig('cost_function_convergence_linear_regression.png')

# Test the predictions using 30% of the data
df_test = df_ceo.drop(df_train.index)

# Perform the same one-hot encoding on test data
encoded_test_features = np.hstack([one_hot_encode_with_unique_values(df_test, col, unique_values_dict[col]) for col in categorical_features])
numeric_test_data = df_test[numeric_features].values
normalized_test_numeric_data = (numeric_test_data - mean_numeric) / std_numeric  # Normalize using train data stats
X_test = np.hstack([encoded_test_features, normalized_test_numeric_data])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])  # Add intercept term

y_test = df_test['price'].values

# Normalize the target variable
y_test_normalized = (y_test - mean_y_train) / std_y_train

# Make predictions on the test data
predicted_test_price_normalized = X_test.dot(coeff)
predicted_test_price = predicted_test_price_normalized * std_y_train + mean_y_train  # Reverse normalization

y_pred = np.dot(X_test, coeff)
error = (1 / y_test.size) * np.sum(np.abs(y_pred - y_test_normalized))

print("Prediction Error: ", error * 100, "%")
print("Accuracy: ", (1 - error) * 100, "%")

# Save the model coefficients and other necessary data to a file
with open('model_data.pkl', 'wb') as f:
    pickle.dump({
        'coeff': coeff,
        'mean_numeric': mean_numeric,
        'std_numeric': std_numeric,
        'mean_y_train': mean_y_train,
        'std_y_train': std_y_train,
        'unique_values_dict': unique_values_dict
    }, f)
