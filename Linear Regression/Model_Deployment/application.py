## Importing the libraries
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## Initializing Flask
application = Flask(__name__)
app = application

## Import ridge regression and standard scaler pickle
ridge_model = pickle.load(open(r'E:\udemy\Machine_Learning\Linear Regression\Model_Building\ridge.pkl','rb'))
standard_scaler = pickle.load(open(r'E:\udemy\Machine_Learning\Linear Regression\Model_Building\scaler.pkl','rb'))

## Creating routes - welcome page
@app.route("/")
def index():
    return render_template('index.html')

## Re-routing page to prediction page
@app.route('/predict data',methods = ['GET','POST'])
def predict_datapoint():
    if request.method =='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Convert it to a DataFrame using the original column names          
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', results = result[0])
    
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host ='127.0.0.1', port = 5000, debug = True)