from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model data
with open('models.pkl', 'rb') as f:
    model_data = pickle.load(f)

all_logistic_regressions = model_data['all_logistic_regressions']
W_multinomial = model_data['W_multinomial']
B_multinomial = model_data['B_multinomial']
mean_numeric = model_data['mean_numeric']
std_numeric = model_data['std_numeric']
unique_values_dict = model_data['unique_values_dict']

def one_hot_encode_with_unique_values(value, unique_values):
    return (unique_values == value).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define a mapping of class labels to price ranges
price_ranges = {
    0: 'Less than 500',
    1: '501 - 1000',
    2: '1001 - 1500',
    3: '1501 - 3000',
    4: '3001 - 10000',
    5: '10001 - 15000',
    6: 'More than 15000'
}

@app.route('/')
def home():
    return render_template('index2.html', unique_values=unique_values_dict)

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form['model']
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
        return render_template('index2.html', error_message='Invalid input. Please select valid options.', unique_values=unique_values_dict)

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

    if model_choice == 'one_vs_all':
        # One-vs-All Logistic Regression
        predictions = np.zeros(len(all_logistic_regressions))
        for i, (W, B) in enumerate(all_logistic_regressions):
            Z = np.dot(W.T, X_new) + B
            A = sigmoid(Z)
            predictions[i] = A
        predicted_class = np.argmax(predictions)
    else:
        # Multinomial Logistic Regression
        Z = np.dot(W_multinomial, X_new) + B_multinomial
        Z = Z - np.max(Z, axis=0, keepdims=True)  # Numerical stability
        exp_Z = np.exp(Z)
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        # Debug: Log the intermediate values
        print("Intermediate Z values:", Z)
        print("Intermediate A values:", A)
        print("Sum of A values:", np.sum(A, axis=0))
        print("Predicted class before validation:", np.argmax(A, axis=0))

        predicted_class = np.argmax(A, axis=0)

        # Ensure predicted_class is within valid range
        predicted_class = int(predicted_class[0])  # Convert to integer from array

        if predicted_class not in price_ranges:
            print(f'Invalid predicted class: {predicted_class}')
            return render_template('index2.html', error_message='Unexpected prediction error. Please try again.', unique_values=unique_values_dict)

    # Map the predicted class to its price range
    predicted_price_range = price_ranges[predicted_class]

    return render_template('index2.html', prediction_text=f'Predicted Price Range: {predicted_price_range}', unique_values=unique_values_dict)


if __name__ == '__main__':
    app.run(debug=True)
