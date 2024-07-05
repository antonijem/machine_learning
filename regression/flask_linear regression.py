from flask import Flask, request, render_template
import  numpy as np
import pickle

app = Flask(__name__)

# Load the saved model data
with open('model_data.pkl', 'rb') as f:
    model_data = pickle.load(f)

coeff = model_data['coeff']
mean_numeric = model_data['mean_numeric']
std_numeric = model_data['std_numeric']
mean_y_train = model_data['mean_y_train']
std_y_train = model_data['std_y_train']
unique_values_dict = model_data['unique_values_dict']

def one_hot_encode_with_unique_values(value, unique_values):
    return (unique_values == value).astype(float)

@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values_dict)

@app.route('/predict', methods=['POST'])
def predict():
    author = request.form['author']
    category = request.form['category']
    publisher = request.form['publisher']
    cover = request.form['cover']
    format = request.form['format']
    year = float(request.form['year'])
    pages = float(request.form['pages'])

    # Validate inputs
    if (author not in unique_values_dict['author'] or
        category not in unique_values_dict['category'] or
        publisher not in unique_values_dict['publisher'] or
        cover not in unique_values_dict['cover'] or
        format not in unique_values_dict['format']):
        return render_template('index.html', error_message='Invalid input. Please select valid options.', unique_values=unique_values_dict)

    # One-hot encode the categorical features
    encoded_features = []
    for col in ['author', 'category', 'publisher', 'cover', 'format']:
        encoded_features.append(one_hot_encode_with_unique_values(locals()[col], unique_values_dict[col]))

    encoded_features = np.hstack(encoded_features)

    # Normalize the numeric features
    numeric_features = np.array([year, pages])
    normalized_numeric_features = (numeric_features - mean_numeric) / std_numeric

    # Combine all features
    X_new = np.hstack([encoded_features, normalized_numeric_features])
    X_new = np.hstack([1, X_new])  # Add intercept term

    # Predict the price
    predicted_price_normalized = np.dot(X_new, coeff)
    predicted_price = predicted_price_normalized * std_y_train + mean_y_train  # Reverse normalization

    return render_template('index.html', prediction_text=f'Predicted Price: {predicted_price:.2f}rsd', unique_values=unique_values_dict)

if __name__ == '__main__':
    app.run(debug=True)
