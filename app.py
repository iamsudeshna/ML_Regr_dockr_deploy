import pickle
import os
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#load the model
model=pickle.load(open('regression_model.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api', methods=['POST'])     #for postman testing
# def predict_api():
#     data=request.json['data']
#     print("\n data: ", data)

#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     print("\n new data:", new_data)

#     newer_values = [1.0] + new_data.flatten().tolist()  # add the missing value at start
#     print("\n newer values:", newer_values)

#     output=model.predict(np.array(newer_values).reshape(1, -1))
#     print(output[0])

#     return jsonify(output[0])


@app.route('/predict_api', methods=['GET'])
def predict_api():
    MedInc = float(request.args.get("MedInc"))
    HouseAge = float(request.args.get("HouseAge"))
    AveRooms = float(request.args.get("AveRooms"))
    AveBedrms = float(request.args.get("AveBedrms"))

    # Combine into array
    data = [MedInc, HouseAge, AveRooms, AveBedrms]
    new_data = scalar.transform(np.array(data).reshape(1, -1))
    newer_values = [1.0] + new_data.flatten().tolist()
    output = model.predict(np.array(newer_values).reshape(1, -1))

    return jsonify({"prediction": float(output[0])})
#url for testing: http://127.0.0.1:5000/predict_api?MedInc=7.2574&HouseAge=53&AveRooms=8.388136&AveBedrms=1.073446

@app.route('/predict')
def predict():
    MedInc = float(request.args.get("MedInc"))
    HouseAge = float(request.args.get("HouseAge"))
    AveRooms = float(request.args.get("AveRooms"))
    AveBedrms = float(request.args.get("AveBedrms"))

    # Your existing scaler + model code
    values = [MedInc, HouseAge, AveRooms, AveBedrms]
    new_data = scalar.transform(np.array(values).reshape(1, -1))
    newer = [1.0] + new_data.flatten().tolist()
    pred = model.predict(np.array(newer).reshape(1, -1))

    return jsonify({"prediction": float(pred[0])})


if __name__=="__main__":
    app.run(debug=True)

