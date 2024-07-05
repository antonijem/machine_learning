import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from kmeans_model import KMeansClustering

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

# Fit K-Means model
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X_train)

# Compute price ranges for each cluster
price_ranges = {}
for cluster in range(kmeans.k):
    cluster_prices = df_train['price'][labels == cluster]
    price_ranges[cluster] = (cluster_prices.min(), cluster_prices.max())

print("Price ranges for each cluster:", price_ranges)

# Save the model and parameters
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump({
        'kmeans': kmeans,
        'unique_values_dict': unique_values_dict,
        'mean_numeric': mean_numeric,
        'std_numeric': std_numeric
    }, f)