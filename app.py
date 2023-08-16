import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data
    data = request.get_json()
    X = np.array(data['input'])

    # Make predictions
    y = model.predict(X)

    # Return the predictions
    return jsonify({'output': y.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)