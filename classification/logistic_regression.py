import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import pickle

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

# Function to perform one-hot encoding using unique values from the entire dataset
def one_hot_encode_with_unique_values(df, column, unique_values):
    one_hot_matrix = np.zeros((df.shape[0], len(unique_values)))
    for i, unique_value in enumerate(unique_values):
        one_hot_matrix[:, i] = (df[column] == unique_value).astype(float)
    return one_hot_matrix

# Define price ranges
def categorize_price(price):
    if price < 500:
        return 0
    elif 501 <= price <= 1000:
        return 1
    elif 1001 <= price <= 1500:
        return 2
    elif 1501 <= price <= 3000:
        return 3
    elif 3001 <= price <= 10000:
        return 4
    elif 10001 <= price <= 15000:
        return 5
    else:
        return 6

df_ceo['price_category'] = df_ceo['price'].apply(categorize_price)

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
y_train = df_train['price_category'].values

# Add intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic regression model
def logistic_regression(X, Y, learning_rate, iterations):
    m = X.shape[1]
    n = X.shape[0]

    W = np.zeros((n,1))
    B = 0

    cost_list = []
    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)

        cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        dW = (1/m)*np.dot(X, (A-Y).T)
        dB = (1/m)*np.sum(A-Y)

        W = W - learning_rate*dW
        B = B - learning_rate*dB

        cost_list.append(cost)

        if(i%(iterations/10) == 0):
            print("Cost after ", i, " iterations is:  ", cost)
    return W, B, cost_list

# One-vs-all logistic regression
num_classes = 7
all_logistic_regressions = []
learning_rate = 0.005
iterations = 1000

for i in range(num_classes):
    y_binary = (y_train == i).astype(int)
    print(f'Class: {i}')
    W, B, cost_list = logistic_regression(X_train.T, y_binary, learning_rate, iterations)
    all_logistic_regressions.append((W, B))
    plt.plot(cost_list, label=f'Class {i}')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations for each class')
plt.legend()
plt.savefig('cost_function_convergence_log_reg_ova.png')
plt.clf()

# Test the predictions using 30% of the data
df_test = df_ceo.sample(frac=0.3, random_state=10)

# Perform the same one-hot encoding on test data
encoded_test_features = np.hstack([one_hot_encode_with_unique_values(df_test, col, unique_values_dict[col]) for col in categorical_features])
numeric_test_data = df_test[numeric_features].values
normalized_test_numeric_data = (numeric_test_data - mean_numeric) / std_numeric  # Normalize using train data stats
X_test = np.hstack([encoded_test_features, normalized_test_numeric_data])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])  # Add intercept term

y_test = df_test['price_category'].values

def predict(X, all_log_models):
    m = X.shape[1]
    predictions = np.zeros((m, len(all_log_models)))
    for i, (W, B) in enumerate(all_log_models):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        predictions[:, i] = A
    return np.argmax(predictions, axis=1)

y_pred_ova = predict(X_test.T, all_logistic_regressions)

accuracy_ova = np.mean(y_pred_ova == y_test)
print(f'One-vs-All Accuracy: {accuracy_ova * 100:.2f}%')

# Multinomial logistic regression
def multinomial_logistic_regression(X, Y, learning_rate, iterations, num_classes):
    m = X.shape[1]
    n = X.shape[0]

    W = np.zeros((num_classes, n))
    B = np.zeros((num_classes, 1))

    cost_list = []
    for i in range(iterations):
        Z = np.dot(W, X) + B
        A = np.exp(Z - np.max(Z, axis=0))  # Subtract max for numerical stability
        A = A / np.sum(A, axis=0)

        cost = - (1/m) * np.sum(Y * np.log(A))

        dW = (1/m) * np.dot((A - Y), X.T)
        dB = (1/m) * np.sum(A - Y, axis=1, keepdims=True)

        W = W - learning_rate * dW
        B = B - learning_rate * dB

        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            print("Cost after ", i, " iterations is:  ", cost)
    return W, B, cost_list

# Convert y_train to one-hot encoded matrix for multinomial logistic regression
y_train_one_hot = np.zeros((num_classes, X_train.shape[0]))
for i, val in enumerate(y_train):
    y_train_one_hot[val, i] = 1

# Train multinomial logistic regression model
print('Training Multinomial Logistic Regression Model')
W_multinomial, B_multinomial, cost_list_multinomial = multinomial_logistic_regression(X_train.T, y_train_one_hot, learning_rate, iterations, num_classes)

plt.plot(cost_list_multinomial, label='Multinomial Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations for Multinomial Logistic Regression')
plt.legend()
plt.savefig('cost_function_convergence_log_reg_mul.png')

# Corrected predict function for multinomial logistic regression
def predict_multinomial(X, W, B):
    Z = np.dot(W, X) + B
    A = np.exp(Z - np.max(Z, axis=0))  # Subtract max for numerical stability
    A = A / np.sum(A, axis=0)

    predicted_class = np.argmax(A, axis=0)

    return predicted_class


y_pred_multinomial = predict_multinomial(X_test.T, W_multinomial, B_multinomial)

accuracy_multinomial = np.mean(y_pred_multinomial == y_test)
print(f'Multinomial Logistic Regression Accuracy: {accuracy_multinomial * 100:.2f}%')

# Save the models and parameters
with open('models.pkl', 'wb') as f:
    pickle.dump({
        'all_logistic_regressions': all_logistic_regressions,
        'W_multinomial': W_multinomial,
        'B_multinomial': B_multinomial,
        'mean_numeric': mean_numeric,
        'std_numeric': std_numeric,
        'unique_values_dict': unique_values_dict
    }, f)
