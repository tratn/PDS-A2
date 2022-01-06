import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template

final_model = pickle.load(open('model.pkl', 'rb'))
X_test = pd.read_csv('X_test.csv', dtype={"YearsCodePro": "category", "YearsCode": "category", "OpSys": "category", "Employment": "category",
                                          "Country": "category", "CompFreq": "category", "Hobbyist": "category", "MainBranch": "category", "EdLevel": "category", "JobSat": "category"})
y_test = pd.read_csv('y_test.csv', dtype={"ConvertedComp": "category"})


app = Flask(__name__)
y_pred = final_model.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))


@app.route('/')
def home():
    return "Hello World"


if __name__ == "main":
    app.run(debug=True)


# print(y_test.info())
