import json
import pickle
import logging

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    logging.debug("Rendering home page.")
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    logging.debug(f"Received data for prediction: {data}")
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    logging.debug(f"Transformed data: {new_data}")
    output = regmodel.predict(new_data)
    logging.debug(f"Prediction result: {output[0]}")
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    logging.debug(f"Form data received for prediction: {data}")
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    logging.debug(f"Transformed data for prediction: {final_input}")
    output = regmodel.predict(final_input)[0]
    logging.debug(f"Prediction result: {output}")
    return render_template("home.html", prediction_text=f"The House price prediction is {output}")

if __name__ == "__main__":
    logging.debug("Starting Flask application.")
    app.run(debug=True)
