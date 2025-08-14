import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn.preprocessing import StandardScaler




application=Flask(__name__)
app=application

ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open('models/scaler.pkl',"rb"))
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        input_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(input_scaled)
        return render_template('home.html', results=result[0])
    else:
        return render_template("home.html")
   


if __name__ =="__main__":
    application.run(host="0.0.0.0")
    ## here 0.0.0.0 means it is getting mapped to y local ip address
    ## our web application should interact with ridge.pickle and standard.pickle

## import ridge regresssor and standard scaler pickle
## agar index.html template folder mai nahi hai toh file load nahi hogi






     