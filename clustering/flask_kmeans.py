from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from kmeans_model import KMeansClustering

app = Flask(__name__)

# Load the trained model and parameters
with open('kmeans_model.pkl', 'rb') as f:
    data = pickle.load(f)
    kmeans = data['kmeans']
    unique_values_dict = data['unique_values_dict']
    mean_numeric = data['mean_numeric']
    std_numeric = data['std_numeric']


def one_hot_encode_with_unique_values(df, column, unique_values):
    one_hot_matrix = np.zeros((df.shape[0], len(unique_values)))
    for i, unique_value in enumerate(unique_values):
        one_hot_matrix[:, i] = (df[column] == unique_value).astype(float)
    return one_hot_matrix


@app.route('/')
def home():
    return render_template('index3.html', unique_values=unique_values_dict)


@app.route('/predict', methods=['POST'])
def predict():
    author = request.form['author']
    category = request.form['category']
    publisher = request.form['publisher']
    cover = request.form['cover']
    format_ = request.form['format']
    year = float(request.form['year'])
    pages = float(request.form['pages'])

    # Create a DataFrame with the input data
    input_df = pd.DataFrame({
        'author': [author],
        'category': [category],
        'publisher': [publisher],
        'cover': [cover],
        'format': [format_],
        'year': [year],
        'pages': [pages]
    })

    # Perform one-hot encoding
    encoded_input_features = np.hstack(
        [one_hot_encode_with_unique_values(input_df, col, unique_values_dict[col]) for col in
         ['author', 'category', 'publisher', 'cover', 'format']])

    # Normalize numeric features
    numeric_input_data = input_df[['year', 'pages']].values
    normalized_input_numeric_data = (numeric_input_data - mean_numeric) / std_numeric

    # Combine all features
    X_input = np.hstack([encoded_input_features, normalized_input_numeric_data])

    # Predict the cluster
    cluster = kmeans.fit(X_input)

    return render_template('index3.html', prediction_text=f'Predicted cluster: {cluster[0]}', unique_values=unique_values_dict)


if __name__ == '__main__':
    app.run(debug=True)
