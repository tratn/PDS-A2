import os
import pickle
import pandas as pd
import numpy as np

from flask import Flask, jsonify, request, render_template
import numpy as np

MODEL_PATH = "lightgbm_model.pkl"
with open(MODEL_PATH, "rb") as rf:
    clf = pickle.load(rf)

# Init the app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# App health check
@app.route("/healthcheck", methods=["GET", "POST"])
def healthcheck():
    msg = (
        "This is a sentence to check if the server is running"
    )
    return jsonify({"message": msg})

# predict function
def predict_function(clf):   
    url = 'X_test.csv'
    X_test = pd.read_csv(url, sep=',')
    y_pred = clf.predict(X_test)  

    return y_pred


# Predict function api
@app.route("/predict", methods=['POST'])
def predict():
    predictions = predict_function(clf)

    result = {
        'prediction': list(predictions)
    }

    return jsonify(result)


# evaluate function
def evaluate_function(model):
    # separate features / label column here:
    url_X = 'X_test.csv'
    X_test = pd.read_csv(url_X, sep=',')

    url_y = 'y_test.csv'
    y_test = pd.read_csv(url_y, sep=',', dtype={"ConvertedComp": "category"})
                    
    # predict
    y_pred = clf.predict(X_test)
    
    # evaluate
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    
    return accuracy, precision


# evaluate function api
@app.route("/evaluate", methods=['POST'])
def evaluate():
    accuracy, precision = evaluate_function(clf)

    result = {
        'accuracy': accuracy,
        'precision': precision
    }
    
    return jsonify(result)

# main
if __name__ == '__main__':
    app.run(debug=True)